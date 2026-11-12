# 故障监控与处理标准 v1.0

本标准回答四个问题：**采什么、怎么判、谁来动、怎么算恢复**。
它约束的是控制面行为，不约束具体集群规模。

## 1. 五段式处理链路

```text
采集 ──> 检测 ──> 定位 ──> 决策 ──> 执行 ──> 验证
 │        │        │        │        │        │
 统一指标  UFC 码   受影响对象  策略 Plan  动作序列  后验条件
```

每一段都必须留下可回溯的记录（`timeline.jsonl`），否则事后无法回答
"为什么在第 137 秒把作业迁走了"。

### 1.1 采集

统一指标命名 `对象.指标`，与 Prometheus 的 snake_case 一一对应：

| 对象 | 指标示例 | 来源 |
| --- | --- | --- |
| gpu | `gpu.util` `gpu.temp_c` `gpu.ecc_uncorrected` `gpu.row_remap_pending` `gpu.pcie_replay_total` `gpu.nvlink_crc_rate` | nvidia-smi XML / DCGM / rocm-smi / npu-smi |
| node | `node.ready` `node.disk_used_ratio` `node.mem_avail_ratio` `node.psu_ok` `node.driver_ok` | node-exporter / kubelet / IPMI |
| job | `job.step` `job.step_time_s` `job.progress_age_s` `job.loss_nan` `job.ckpt_age_s` `job.step_skew_ratio` | 训练框架钩子 / 心跳文件 / Prometheus |
| svc | `svc.health` `svc.queue_depth` `svc.p99_latency_s` `svc.kv_hit_ratio` | vLLM metrics / 网关 |

采集频率：硬件 15s，作业 1–5s，推理 5s。**所有采集器必须能离线回放**
（本项目用 `fixtures/` + `ReplayCollector`），否则故障剧本无法回归。

### 1.2 检测

三条硬规则：

1. **去抖**：`for_samples` 连续成立才算故障。默认 S1=1、S2≥3、S3≥5。
2. **迟滞**：`clear_samples` 连续不成立才恢复，且一般 ≥ `for_samples`。
3. **冷却**：同一 `(故障码, 对象)` 在 `cooldown_s` 内不重复触发动作。

事件型（厂商码）与指标型可并存：精确码命中时不再采纳文本正则推断
（否则 `Xid 74` 行里的 "Fatal" 会把它误判成驱动崩溃）。

### 1.3 定位

定位的产出是 `Target(node, gpu, job, service)`：

| scope | 定位到 | 处置单位 |
| --- | --- | --- |
| gpu | 单卡 | 隔离卡、迁移作业 |
| node | 整机 | cordon + 疏散 |
| job | 作业 | 重启/缩容 |
| service | 推理实例 | 摘流重启 |
| observability | 采集链路 | 降级为只读 |

作业与卡的关联来自：Pod 注解（`aiops.io/gpu-uuid`）、DCGM 的 pod 标签、
或训练框架上报的 `CUDA_VISIBLE_DEVICES`。

## 2. 观测质量是控制状态，不是监控项

这是本标准里最容易被忽略、也最容易出事的一条。

| 状态 | 判定 | 允许动作 |
| --- | --- | --- |
| OK | 数据新鲜、指标完整 | 全部 |
| DEGRADED | 指标缺失 >20%，或来源冲突 | 只读 + 可逆（保存、摘流、降载、重试单步） |
| STALE | 超过 3 个采集周期未更新 | 仅告警 + 紧急保存 |
| BLIND | 超过 8 个采集周期未更新 | 仅告警/升级，**动作一律拒绝** |

实现见 `aiops/detect/telemetry.py`，决策层的降级动作会带
`degraded_by_telemetry=true` 写入 Plan，便于复盘。

## 3. 处理标准（按等级的时限）

| 等级 | 检测时限 | 动作时限 | 验证时限 | 人工介入 |
| --- | --- | --- | --- | --- |
| S1 | 30s | 60s | 5min | 动作后立即建单 |
| S2 | 2min | 10min | 15min | 连续 2 次自动失败 |
| S3 | 10min | 下一维护窗口 | 1h | 不需要 |
| S4 | 记录 | 不需要 | — | 月度评审 |

未在规定时限内完成检测/动作/验证，必须产生一条"超时"事件并升级。

## 4. 动作词表

动作是**受控词汇**，策略只能从这里选，执行器只认这些名字：

| 动作 | 语义 | 前置条件 |
| --- | --- | --- |
| `emergency_checkpoint` | 在 `grace_s` 内落一份检查点 | 检查点目录可写 |
| `graceful_restart` | 同节点同卡重启进程 | 有检查点 |
| `migrate_job` | 迁到备用节点/健康卡 | 有备机 + 有检查点 |
| `evacuate_job` | 清空节点并禁止回迁 | 有备机 |
| `shrink_world_size` | 去掉坏 rank，缩容续跑 | 有检查点 + world_size>1 |
| `quarantine_gpu` / `quarantine_node` | 标记不可调度 | 仅标记，不动作业 |
| `fence_node` | 强制隔离（电源/网络） | 需带外权限 |
| `drain_instance` / `restart_instance` | 推理摘流/重启 | 实例健康检查可用 |
| `reduce_load` / `reduce_batch` | 降载不重启 | 可调参数 |
| `retry_step` | 只重试当前步 | 幂等 step |
| `rollback_driver` | 回滚驱动/固件 | 有版本矩阵 |
| `notify_only` / `escalate_human` | 只读 | 无 |

`quarantine_*` 与 `fence_node` 是"防止二次伤害"的动作：前者只是调度层标记，
后者会真正断电/断网，必须人工确认后由带外系统执行。

## 5. 验证（强制）

动作执行后进入 `RECOVERING`，只有同时满足以下条件才算恢复：

1. 作业进度推进超过 `max(3 步, 检查点之后 5 秒)`；
2. 触发故障的同码故障在验证窗口内未复发；
3. 该作业的吞吐/微基准回到基线 85% 以上（训练类）或 p99 回到 SLO 内（推理类）。

任何一条不满足 → 记录 `verified=false`，按升级阶梯加重策略；重试耗尽后
`fail_fast` 并升级人工。**禁止在验证失败时原地反复重启**。

## 6. 值班 runbook 模板

每个 UFC 码都必须有可执行步骤，格式：

```text
1. 确认（哪条命令、看什么字段）
2. 止损（隔离/摘流/保存，谁有权执行）
3. 恢复（迁移/重启/缩容，前置条件）
4. 验证（看哪个指标、持续多久）
5. 收尾（工单、证据留存、复发观察窗口）
```

示例（UFC-MEM-001 不可纠正 ECC）：

```text
1. nvidia-smi -q -d ECC 确认 volatile/aggregate 的 uncorrectable 计数
2. 冻结该卡：标记不可调度；把作业迁走（需检查点）
3. 不要原地重试——数据可能已被污染
4. 迁移后确认 step 推进、ECC 计数不再增长、微基准达标
5. 建硬件工单，保留 dmesg / nvidia-bug-report；观察 24h 是否复发
```

## 7. 与告警链路的关系

UFC 码是告警的**标签**而不是替代品：

```text
aiops_ufc_active{code="UFC-MEM-001", severity="S1", node="gpu-node-15"} == 1
```

`python -m aiops catalog prometheus` 直接生成 PrometheusRule，交给
`alertment` 既有的 Alertmanager → 飞书链路；告警正文里带 `runbook` 第一条，
值班不需要再翻文档找入口。

告警分级与静默：S1 立即通知 + 电话；S2 群内通知；S3 日报聚合；S4 周报。
演练窗口用 `suppressed` 状态显式标注，避免"静默期间真故障被吞"。
