# 推理任务的故障迁移（端到端）

目标：**推理服务出错 → 报错码 → 翻译成统一故障码 → 上报 → 自动保存进度 →
迁到另一台机器续跑 → 验证恢复结果**。

和训练作业不同，推理没有"训练步"和"模型检查点"可以续；这里的"进度"是
**已完成的请求/样本**与**已产出的结果**。所以保存的不是权重，而是**作业状态**：
处理到哪一条、结果写到哪一行、还有多少没做。

## 1. 组成

```text
        控制面（任意有 SSH 通道的机器）
        ┌──────────────────────────────────────────────┐
        │ demo/inference/controller.py                 │
        │  检测 → 翻译(UFC) → 告警 → 保存 → 搬运 → 验证 │
        └───────┬───────────────────────────┬──────────┘
                │ SSH                       │ SSH
        ┌───────▼────────┐          ┌───────▼────────┐
        │ 源节点          │          │ 目标节点        │
        │ inference/worker│  state   │ inference/worker│
        │  GPU0 :18000    │ ───────► │  GPU1 :18001    │
        └────────────────┘          └────────────────┘
```

- `demo/inference/worker.py`：推理服务。可续跑、有健康检查/指标、可注入真实 CUDA 错误。
- `aiops/remote/ssh.py`：远端执行与文件搬运（迁移的物理通道）。
- `demo/inference/controller.py`：状态机与度量。

## 2. 状态模型（"保存进度"到底保存什么）

```text
<run-dir>/state/
  progress.json     {"done": 400, "total": 1200, "node": "c-gpu0", "ts": ...}
  results.jsonl     每行一条结果 {"i": 123, "next": 4567, "node": ...}
  ckpt/ckpt-NNNNNN.json   每 N 个样本一份提交标记
```

约定：

1. 每处理完一个 batch，先追加结果、`fsync`，再原子写 `progress.json`；
2. 恢复时按 `progress.done` **截断结果文件**，保证"不重不漏"；
3. 迁移只搬运 `state/` 目录（本实验约 17KB），不搬运模型——权重在目标机上加载；
4. 完整性校验：前 `done` 条必须是 `0..done-1` 且唯一，总行数不超过 `done + batch`
   （最后一批可能正在写）。

## 3. 检测与翻译

控制面每 0.5s 采一次：

| 信号 | 判定 |
| --- | --- |
| `exit_code`（worker 或 wrapper 写） | 非 0/42 即失败 |
| 进程是否存活（`kill -0`） | 消失即失败 |
| `/health` | 非 200 且心跳停摆即失败 |
| 心跳 `ts` | 超过 `hang_s` 未更新即挂起 |
| 日志尾部 | 取 `FATAL` / `CUDA error` / `out of memory` 行作为证据 |

翻译走**同一套 UFC 目录**，不是另写一套规则。本轮注入的是真实 CUDA 错误：

```text
[worker] FATAL AcceleratorError: CUDA error: device-side assert triggered
  -> UFC-GPU-002 非法显存访问 / 上下文异常（S2）
```

映射项（`catalog.yaml`，`confidence: community`）：

```yaml
- vendor: generic
  codes: ["CUDA_DEVICE_ASSERT", "CUDA_ILLEGAL_ACCESS"]
  patterns: ['CUDA error:.*(device-side assert|illegal memory access|unspecified launch failure)']
```

容器里通常看不到 `dmesg` 的 Xid，**应用侧 CUDA 错误文本就是主要证据**——这也是为什么
目录里要留 `generic` 厂商位。

## 4. 上报

产物：`alert.json`（结构化）与 `alert.txt`（飞书/群消息风格）。字段：

```json
{"ufc":"UFC-GPU-002","name":"…","severity":"S2","source_node":"c-gpu0",
 "evidence":["[worker] FATAL AcceleratorError: CUDA error: device-side assert triggered"],
 "runbook":["保留崩溃现场…","从最近检查点重启…"],"recommended_strategy":["emergency_checkpoint","migrate_job"]}
```

`alert.json` 可以直接喂给 `alertment` 的飞书 adapter（字段与既有告警链路兼容）。

## 5. 迁移与验证

```text
故障 → 尽力保存（活着就优雅停，死了用最后提交的进度）
     → 从源机取回 state/ → 传到目标机 state/
     → 源节点写隔离标记（quarantine-<node>.json）
     → 目标机 --resume 启动，从 done 继续
     → 验证：进度推进 ≥ verify_samples、/health=200、同码错误不再出现、完整性通过
```

注意"尽力保存"：进程被 SIGKILL / CUDA 上下文损坏时**来不及保存**，只能退回到
最后一个 batch 的提交点，RPO 就是那一批的量级（本实验 0–8 个样本）。

## 6. 实测（2×4090 双卡机，GPU0 → GPU1）

| 指标 | 值 |
| --- | --- |
| 故障 | 真实 CUDA `device-side assert` → `UFC-GPU-002`（S2） |
| 故障时进度 | 400 / 1200 样本 |
| 保存点 | 400（同一 batch 边界） |
| 丢失样本（RPO） | **0** |
| MTTD（故障→识别） | 1.77s |
| 保存耗时 | 8.9s（含优雅停止等待） |
| MTTR（故障→验证通过） | 10.88s |
| 源机/目标机吞吐 | 19.0 / 25.2 samples/s（目标机测量窗口短，偏高） |
| 完整性 | 432 条已提交、下标 0..431 连续、无重复 |
| 搬运 | 8 个文件 / 17.6KB |

原始产物：`results/inference-migration/{alert.json,timeline.jsonl,result.json,summary.md}`。

## 7. 只能拿到一台双卡机，够不够？

够验证整条链路，但**不能验证"整机宕机"**：

| 场景 | 双卡机（本次） | 两台机器 |
| --- | --- | --- |
| 单卡故障 / CUDA 上下文损坏 | ✅ GPU0→GPU1 | ✅ |
| 单卡 OOM / 驱动异常 | ✅ | ✅ |
| 推理进程崩溃、卡死 | ✅ | ✅ |
| 状态跨节点搬运（SSH/SFTP） | ✅（同一台机，两个目录） | ✅（真正跨机） |
| 整机宕机 / 断网 / 内核崩溃 | ❌ 同机无处可迁 | ✅ |
| 跨机网络带宽对迁移耗时的影响 | ❌ | ✅ |

代码本身不区分"另一个 GPU"和"另一台机器"：控制器只认 `hosts` 配置。换成两台机器时，
只改 `.local/hosts.yaml` 里的 `host/port/gpu`，controller 与 worker 一行不用动。

## 8. 运行

```bash
# 1) 两台（或两个 GPU）上部署代码
#    /root/aiops 下需要有 demo/inference/worker.py

# 2) 配置
cp demo/inference/hosts.example.yaml .local/hosts.yaml   # 填 host/port/password/gpu

# 3) 跑
python -m demo.inference.controller --config .local/hosts.yaml \
    --out results/inference-migration

# 参数：job.total / job.qps / fault.mode(device-assert|cuda-oom) / fault.at
#      run.grace_s / run.hang_s / run.verify_samples
```

## 9. 接到真实的推理平台

worker 只是一个"最小可续跑推理服务"，替换成 vLLM/SGLang 时保留四件事即可：

1. `/health` 与 `/metrics`（vLLM 自带）；
2. 进度落盘：离线批推理用 `done` 偏移；在线服务用"请求去重表/在途请求数"；
3. 优雅停止：`SIGTERM` 时停止接新请求、等在途结束、落一次进度；
4. 启动参数里接受 `--resume`（或 `--start-offset`）。

在线服务的"迁移"还要配合网关摘流（`model-gateway` 权重置 0 → 等在途 → 迁移 → 回填），
否则切换瞬间会有请求打到已经停掉的实例。

## 10. 局限

- 目标机加载模型的时间没有算进"保存"，但对 RTO 有影响；大模型应预热或做镜像缓存。
- 状态搬运是"全量覆盖"，不支持增量/断点续传；大状态（KV cache 快照）需要对象存储。
- 只验证了"单次迁移"；没有验证目标机再次故障时的二次迁移/回迁策略。
- 告警目前落盘 + 控制台，没有接真实 webhook（接口已留）。
