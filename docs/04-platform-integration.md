# 平台对接矩阵

目标：控制面不改代码，就能接管不同的训练/推理框架。做法是定义一个
**统一作业契约**，各平台只写一个薄适配层。

## 1. 统一作业契约

### 1.1 进程契约

监督器是作业的父进程，负责启动、停止、重启。要求：

- 只启动**一个**入口进程（torchrun / deepspeed / vLLM / 自定义 launcher），
  由它自己 fan-out 出 rank；
- 停止时先发 `SIGTERM`（Linux）或写控制文件（演练/Windows），
  超时 `grace_s` 后强制清理进程树；
- 重启时通过 argv/env 传达"从检查点恢复"和新的拓扑。

### 1.2 心跳契约（JSON，每步/每 1s）

```json
{"rank": 0, "world_size": 2, "step": 137, "step_time_s": 0.42,
 "loss": 1.87, "state": "running|saving|done|failed",
 "node": "gpu-node-15", "ts": 1794488000.1, "pid": 12345}
```

写入 `<run>/hb/attempt-<n>/rank-<i>.json`，`<run>/hb/current` 指向当前尝试目录。
`state=failed` 必须带 `reason`。控制面据此算 `progress_age_s`、`step_skew_ratio`。

### 1.3 控制契约

| 文件 | 含义 |
| --- | --- |
| `graceful` | 保存检查点后退出（退出码 42） |
| `force_exit` | 立即退出（退出码 130），用于清场 |
| `die.<rank>` | 模拟/触发该 rank 硬件崩溃（退出码 9） |
| `hang.<rank>` | 该 rank 卡死（不推进、不退出） |
| `throttle.<rank>` | 该 rank 限频（内容为倍数） |

### 1.4 检查点契约

```text
<ckpt>/<job>/ckpt-00000137/
    state.pt | state.json      # 框架自定义状态
    meta.json                  # step/world_size/sha256/ts，由 CheckpointStore 写
```

框架只需要提供 `write_state(dir)` 回调，原子性、校验、保留策略由存储层负责。

## 2. 对接矩阵

| 平台 | 接入点 | 需要的信号 | 恢复方式 | 现状 |
| --- | --- | --- | --- | --- |
| PyTorch DDP / torchrun | `RANK/WORLD_SIZE/LOCAL_RANK`；elastic rendezvous | 步耗时、步数、loss、检查点 | 原地重启 / 迁移 / 缩容（世界大小可变） | ✅ 本机 2 rank 实测（gloo）；4090 走 NCCL |
| DeepSpeed | `deepspeed --num_gpus N` + `deepspeed.checkpoint` | 同上 + ZeRO stage | 同 DDP；检查点用 `save_checkpoint(tag=step)` | ⚠️ 契约已对齐，待真机 |
| Megatron-LM | `torchrun` + `--save/--load` 路径 | 迭代数、`iteration-time`、检查点 | 同 DDP；注意 TP/PP 并行度变化需要重分片 | ⚠️ 待真机 |
| MindSpeed（昇腾） | `torch_npu` + `torchrun` | NPU 利用率、HCCL 超时、HBM | 同 DDP；HCCL 需设置 `HCCL_EXEC_TIMEOUT` | ⚠️ 与昇腾采集器联动 |
| vLLM | `/health`、`/metrics`、进程 | p99、队列、KV cache 命中、显存 | 摘流 → 重启 → 回填 | ⚠️ 契约已定义；`model-gateway` 可承接摘流 |
| SGLang | `/health`、`/metrics` | 同上 | 同上 | ⚠️ 同上 |
| Kubernetes / AIJob | Pod 注解、`kubectl delete/apply`、cordon | Pod 状态、GPU 归属 | 删除 Pod 由控制器重建；或改 CR 的 worldSize | ⚠️ 与 `ai-job-Operator` 的 CR 对接 |
| GPUStack | Worker `/metrics` | 实例/显存/请求 | 重启实例 | ✅ 指标已在 `alertment` 链路里 |
| 裸机 systemd | `systemctl restart` | 心跳文件 | 重启 unit | ✅ 契约可用 |

## 3. 各平台适配点（代码位置）

没有为每个平台写一整套胶水，而是把差异收敛到三处：

```text
JobSpec.argv              # 启动命令（torchrun / deepspeed / vllm serve ...）
JobSpec.spawn_argv()      # 重启时如何改 world_size / node / resume
Probe（心跳读取）          # 平台自己上报 or 控制面抓 Prometheus
```

### 3.1 torchrun / DDP

- 生产：`torchrun --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=...`
- 本机：`demo/torch_launcher.py` 用 `FileStore` 做 rendezvous，避开 Windows 的
  libuv/TCPStore 限制；两者对训练代码完全等价。
- 缩容：改 `--nproc_per_node` 或 `--nnodes`；DDP 的检查点需要保证
  `world_size` 变化可读（本项目记录 `world_size` 并要求显式兼容判断）。

### 3.2 DeepSpeed

```python
# 训练脚本里
ckpt = client.save_checkpoint(step_dir, tag=f"step{step}")   # 紧急保存
client.load_checkpoint(step_dir, tag=..., load_module_strict=True)
```

`CheckpointStore.save()` 的 `write_state` 回调里调用 `client.save_checkpoint`，
其余（原子性、保留、RPO 计算）不变。

### 3.3 vLLM（推理）

```text
网关（model-gateway）权重置 0  ->  等待在途请求  ->  kill/重启实例
->  /health 通过 3 次  ->  权重恢复  ->  观察 p99 5 分钟
```

对应策略 `drain_restart`；`verify` 为 `svc_healthy` + `latency_slo`。
推理没有检查点语义，所以 `drain_restart` 不要求 `checkpoint_available`。

### 3.4 Kubernetes / AIJob

- 训练作业的"迁移"在 K8s 里 = 改 nodeSelector/affinity 后重建 Pod；
- "缩容" = 改 `AIJob.spec.worldSize`（需要 CR 支持）；
- "隔离节点" = `kubectl cordon` + 打污点，并写 `aiops.io/quarantine=true` 注解；
- 控制面可以作为 sidecar（同 Pod）或集群级 controller 运行，二者共用同一套
  `Detector/Policy/Supervisor`。

## 4. 能力边界（必须写进交付说明）

1. 目前只在 2 卡规模验证；`shrink` 到 1 卡之后不自动扩回。
2. DeepSpeed/Megatron/vLLM 是**契约级**对接，没有在真实集群跑过；
   真机验证清单见 docs/05 §5。
3. AMD/昇腾采集器用真实格式样本验证了解析，但没有连接过物理设备。
4. K8s 适配层没有直接调用 API Server（避免误操作），只定义了动作语义；
   接入时需要补一份带 RBAC 的 executor。
