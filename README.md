# aiops —— 异构 AI 集群自动运维探索原型

围绕一条主线：**发现硬件故障 → 翻译厂商故障 → 定位受影响作业 → 自动保存进度 →
迁移或重启作业 → 验证恢复结果**。

当前只用 2 张消费级卡（本地 RTX 5060 8GB 开发，目标环境 2×RTX 4090）做探索，
所以设计上把"能在小规模验证的"和"必须上真机才能验证的"严格分开：

| 能力 | 现在能验证吗 | 怎么验证 |
| --- | --- | --- |
| 跨厂商统一故障码（UFC） | ✅ | 48 个故障码、35 个厂商原始码、70 条日志正则；NVIDIA 真机日志 + AMD/昇腾真实格式回放 |
| 多厂商遥测采集 | ✅ / ⚠️ | NVIDIA `nvidia-smi -q -x` 用真机输出；AMD `rocm-smi --json`、昇腾 `npu-smi` 用真实格式样本 |
| 硬件故障检测与生命周期 | ✅ | 单测 + 8 个场景回放（含真实 DDP 作业） |
| 慢节点识别 | ✅ | 步耗时稳健离群 + 微基准交叉判定，4090 上可注入限频复现 |
| ≥2 种动态恢复策略 | ✅ 6 种 | 原地重启 / 迁移 / 疏散 / 缩容 / 摘流重启 / 降载 |
| 作业级 Checkpoint 重启 | ✅ | 原子写 + hash 校验 + 版本保留；仿真与真实 PyTorch DDP 各跑通一次 |
| 对接主流训练/推理平台 | ⚠️ | torchrun/DDP 已实测；DeepSpeed/Megatron/vLLM/K8s 给出接入契约与适配点，待真机 |
| 跨厂商码的"官方确认" | ❌ | AMD/昇腾映射标了 `assumed`/`community`，上机后必须按厂商手册回填 |

## 五个子系统

```text
aiops/faultcode/   统一故障码目录（catalog.yaml）+ 规范化翻译（normalizer）
aiops/collect/     三厂商采集器 + 离线回放（fixtures 带真实样本）
aiops/detect/      指标规则检测、慢节点识别、观测质量门控
aiops/policy/      6 种恢复策略 + 决策表（可审计的 Plan）
aiops/supervisor/  检查点存储、作业状态机、动作执行、恢复验证、MTTD/MTTR/RPO 度量
demo/              仿真训练器、真实 DDP 训练器、场景回放器
```

核心约定：**故障码只描述语义，检测规则与动作都在目录里声明**；控制面所有动作
必须绑定可检查的后验条件（`verify`），否则不允许自动执行。

## 快速开始

```bash
python -m pip install pyyaml          # 唯一必需依赖
python -m unittest discover -s tests  # 56 个单测

python -m aiops catalog stats
python -m aiops translate --vendor nvidia --code 79
python -m aiops diagnose --vendor ascend --file aiops/collect/fixtures/slog_ascend.txt
python -m aiops catalog matrix --vendor-matrix     # 跨厂商对照表
python -m aiops catalog prometheus                 # 导出 PrometheusRule

python -m aiops demo demo/scenarios/*.yaml --out results
cat results/report.md
```

真实 PyTorch DDP 作业（本机 CPU/gloo 双 rank 已跑通；4090 上自动切 NCCL）：

```bash
python -m aiops demo demo/scenarios/1*.yaml --out results
```

## 实验结果（本机实测，2026-11-12）

| 场景 | 故障 | 恢复策略 | 平均停机 | MTTD | MTTR | RPO（紧急保存后） | 仅周期检查点会丢 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 00-baseline | 无 | — | — | — | — | — | 误动作 0 次 |
| 01-xid79-evacuate | Xid 79 | evacuate → 迁到备机 | 0.11s | 0.0s | 0.66s | 0 步 | 0 步 |
| 02-hang-restart | rank 卡死 | restart_in_place | 6.05s | 0.06s | 6.64s | 2 步 | 2 步 |
| 03-straggler-shrink | 限频 6× | shrink（去慢 rank） | 0.36s | 0.55s | 0.90s | 0 步 | 2 步 |
| 04-ecc-evacuate | Xid 48 | evacuate | 0.11s | 0.0s | 0.67s | 0 步 | 5 步 |
| 05-no-ckpt-failfast | 无检查点崩溃 | fail_fast + 升级 | — | — | — | — | 不盲目重启 |
| 10-torch-baseline | 无 | — | — | — | — | — | 真实 DDP 无误动作 |
| 11-torch-xid79 | Xid 79（真实 DDP） | evacuate → 迁到备机 | 7.06s | 0.0s | 11.91s | 0 步 | 4 步 |

结论与局限见 [docs/05-experiments.md](docs/05-experiments.md)。一句话：**这条链路是通的，
而且"紧急检查点"把 RPO 从"一个保存周期"降到了 0；真正的难点在真机上的厂商码确认、
NCCL 超时尺度、以及大规模下的误报率。**

## 真实 4090 实测（2026-11-13）

两台租用实例（**1 卡/节点**，RTX 4090 24GB，torch 2.12.1+cu130）上用真实进程 + 真实信号重跑：

| 场景 | 注入 | 结果 | 停机 | MTTR | RPO |
| --- | --- | --- | --- | --- | --- |
| 30 baseline | 无 | 200 步跑完，0 误动作 | — | — | — |
| 31 SIGKILL | `kill -9` | restart_in_place | 0.00s | 3.93s / 4.09s | 7 / 6 步 |
| 32 SIGTERM | 优雅终止 | restart_in_place + 紧急保存 | 0.00s | 3.61s / 4.22s | **0 步** |
| 33 SIGSTOP | 冻结进程（类 GPU 挂死） | restart_in_place + 强杀进程组 | 4.01s | 8.23s / 8.49s | 20 步 |

同一套代码在两台机器上结果一致（差异 <5%，策略选择相同）；同型号 4090 之间的
吞吐差异有 4%–12%，说明慢节点阈值必须按机器校准。

双卡机（节点 C，2×4090 / PCIe / 无 NVLink，DDP 实测 6.25ms/步、all-reduce 56.2 GB/s）：

| 场景 | 注入 | 结果 | 停机 | MTTR | RPO |
| --- | --- | --- | --- | --- | --- |
| 40 baseline | 无 | 200 步，0 误动作 | — | — | — |
| 41 Xid 43 + kill rank1 | SIGKILL + Xid 43 | **shrink 2→1** | 4.01s | 7.87s | **0 步**（仅周期检查点会丢 20 步） |
| 42 单卡降频 | rank1 前向 ×8 | **shrink 2→1** | 1.56s | 5.63s | **0 步**（会丢 17 步） |

真实 DDP 里总步耗时会被最慢的卡拖平，所以慢节点比较用的是 per-rank 前向时间
（`forward_s`），而不是总 step time。

真机还暴露了三个仿真掩盖不了的缺陷（优雅退出被误判成训练完成、强杀留下持有 GPU 的
孤儿 rank、换环境后的 torchrun/CRLF 坑），详见 [docs/06-4090-results.md](docs/06-4090-results.md)。

## 文档

- [统一故障码规范](docs/01-unified-fault-code.md)：命名、分级、字段、映射方法与置信度、扩展开流程
- [故障监控与处理标准](docs/02-monitoring-standard.md)：检测 → 定位 → 决策 → 执行 → 验证，以及观测降级时的安全边界
- [动态恢复策略](docs/03-recovery-strategies.md)：6 种策略、决策表、升级阶梯、检查点语义
- [平台对接矩阵](docs/04-platform-integration.md)：torchrun/DDP、DeepSpeed、Megatron、vLLM、K8s
- [实验记录（仿真 + CPU DDP）](docs/05-experiments.md)：场景、指标、结论、下一步
- [真实 4090 实测](docs/06-4090-results.md)：基线、故障注入、两台机器一致性、真机暴露的缺陷

## 与既有仓库的关系

| 仓库 | 本项目的复用点 |
| --- | --- |
| [`alertment`](https://github.com/oyyyyy61/alertment) | 告警链路（Prometheus → Alertmanager → 飞书）；UFC 码可作为告警标签 |
| [`Prometheus`](https://github.com/oyyyyy61/Prometheus) | 检查点/迁移实验、K8s 恢复控制器、昇腾 NPU 指标与 exporter |
| [`ai-job-Operator`](https://github.com/oyyyyy61/ai-job-Operator) | AIJob/Volcano 作业对象，迁移与缩容最终落在 CR 上 |
| [`model-gateway`](https://github.com/oyyyyy61/model-gateway) | 推理侧摘流/回填的负载均衡入口 |

## 后续（真实 4090 已完成第一轮）

本轮已在两台 1 卡 4090 上验证了真实进程的故障处置与检查点恢复。还没做的：

1. **多卡 DDP 的卡间偏斜**：租用实例是 1 卡/节点且跨节点无直连网络，需要自有双卡机；
2. **硬件降频注入**：容器内 `nvidia-smi -lgc` 无权限，需要 host 权限或自有机器；
3. **ECC/NVLink/HCCS/IB 物理故障**：4090 无 ECC/NVLink，需 A100/H100/910B；
4. **检查点恢复后的精度等价性**：固定随机种子对比 loss 曲线；
5. 把 AMD/昇腾的 `assumed` 映射逐条替换成厂商手册里的原始 Error Code。
