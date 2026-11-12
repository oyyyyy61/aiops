# 统一故障码（UFC）规范 v1.0

## 1. 为什么要统一

同一个物理故障，三家厂商给出三种表达：

| 现象 | NVIDIA | AMD | 昇腾 |
| --- | --- | --- | --- |
| 卡从总线上消失 | `Xid 79` + `GPU has fallen off the bus` | `amdgpu: device lost` | `npu-smi` Health=Alarm + `device lost` |
| 显存不可纠正错误 | `Xid 48/94/95/140` | RAS UE | HBM ECC 多比特错误 |
| 片间互联故障 | `Xid 74` / NVLink CRC | XGMI error | HCCS link down |

如果告警、工单、恢复策略各自认一套码，就会出现"同一个故障三种处置"。
UFC 的作用是**给故障一个与厂商无关的稳定标识**，厂商码只作为它的输入。

## 2. 命名

```text
UFC-<DOMAIN>-<NNN>
```

`DOMAIN` 三位大写，`NNN` 三位序号：

| DOMAIN | 含义 | 典型内容 |
| --- | --- | --- |
| GPU | 算力引擎 | Xid 43/79、AI Core 异常、GPU reset |
| MEM | 显存 / HBM | ECC、行重映射、OOM |
| LNK | 片内互联 | NVLink、HCCS、XGMI、PCIe |
| NET | 节点间网络 | IB/RoCE、NCCL/HCCL 超时 |
| PWR | 供电 | power brake、PSU |
| THM | 温度散热 | 过温、降频、风扇 |
| DRV | 驱动固件 | 驱动崩溃、版本漂移 |
| NOD | 主机/节点 | NotReady、磁盘、内存、宕机 |
| JOB | 训练作业 | 挂起、NaN、检查点、慢节点 |
| INF | 推理服务 | 健康检查、队列、KV cache |
| OBS | 观测链路 | exporter、指标陈旧、指标冲突 |

规则：

1. **序号在域内单调递增，永不复用**；废弃的码保留并标 `status: deprecated`。
2. 一个语义只对应一个码；厂商差异写进 `vendor_map`，不新增码。
3. 码本身不含节点/卡的信息，作用对象由运行期 `Target(node, gpu, job, service)` 携带。

## 3. 字段定义

```yaml
- id: UFC-MEM-001
  name: 显存不可纠正 ECC 错误
  domain: MEM
  severity: S1                 # S1..S4
  scope: gpu                   # gpu|node|job|service|cluster|observability
  summary: 一句话说明危害
  actions: [emergency_checkpoint, evacuate_job, quarantine_gpu, escalate_human]
  checkpoint: emergency        # none|periodic|emergency|both
  verify: [ecc_clear, bench_pass]      # 动作后的强制后验条件
  recurrence: {max_retries: 0, quarantine_after: 1, backoff_s: 0}
  signals:                     # 指标型检测规则（可选）
    - {metric: gpu.ecc_uncorrected, op: gt, value: 0, for_samples: 1}
  vendor_map:                  # 厂商侧输入（至少一条，或至少一条 signals）
    - vendor: nvidia
      codes: ["48", "94", "95", "140"]
      patterns: ['Xid.*\b(48|94|95|140)\b', 'Uncorrectable ECC']
      confidence: verified
  runbook: [ ... ]             # 值班可执行步骤
  tags: [hardware, fatal]
```

`aiops/faultcode/models.py` 在构造时会校验：编号格式、域与编号一致、动作词表、
验证项词表、S1/S2 必须绑定 `verify`、S1 重试次数上限、厂商码不与其它码冲突。

### 为什么 `verify` 是强制的

"动作发出去了"不等于"恢复了"。没有后验条件，控制面就无法区分
"重启成功"和"重启后立刻又挂"，最终会变成重启风暴。所以 S1/S2 的自动动作
必须声明至少一条可检查的后验条件：

`step_advance` / `ckpt_advance` / `bench_pass` / `ecc_clear` / `fault_absent` /
`svc_healthy` / `latency_slo` / `node_ready`

## 4. 分级与处理预算

| 等级 | 含义 | 检测预算 | 允许自动动作 | 说明 |
| --- | --- | --- | --- | --- |
| S1 | 作业已失败或节点不可用，数据完整性受威胁 | ≤30s | 是（但通常 0 重试） | 掉卡、不可纠正 ECC、PSU、节点宕机 |
| S2 | 作业即将失败或性能严重受损 | ≤120s | 是 | 卡级可疑、通信超时、挂起、检查点失败 |
| S3 | 性能降级或可观测性受损 | ≤600s | 计划性 | 可纠正 ECC、PCIe 降速、散热降频 |
| S4 | 记录与趋势 | — | 否 | 版本漂移 |

注意：**等级不是严重性形容词，而是"允许多快、多强地动手"的授权**。

## 5. 生命周期

```text
open ──ack──> verifying ──> resolved
  │                └── 验证失败 ──> 升级（策略阶梯）
  └── suppressed（演练窗口）
```

- 去重键：`(故障码, 作用对象)`；同一键复发不会刷屏，而是累加证据。
- 事件型故障（厂商码）没有"自动恢复"信号，用 `event_ttl_s`（默认 900s）兜底关闭；
  动作执行时会显式 `resolve` 触发故障，复发才算新故障。
- 指标型故障用 `clear_samples` 连续不成立才关闭，避免抖动。

## 6. 厂商映射方法与置信度

映射只能来自三类证据，必须显式标注：

| 置信度 | 要求 | 当前用途 |
| --- | --- | --- |
| `verified` | 厂商手册或真机日志确认 | 允许触发一切自动动作 |
| `community` | 公开 issue / 内核日志格式 | 允许触发，但 S1 需人工确认 |
| `assumed` | 按语义推断 | **只允许只读动作**，上机后必须回填 |

当前目录：NVIDIA 12 条码映射（`verified` 为主）、AMD 5 条（`community`）、
昇腾 7 条（`assumed`，类别名而非官方原始码）。这条边界必须写进交付说明，
否则"统一故障码"会被误读成"已经拿到三家厂商的官方编码表"。

### 翻译顺序

1. 精确厂商码命中（最高优先，一个码只属于一个 UFC）；
2. 正则匹配（仅在无精确码命中时采用，避免一行日志被翻成多个语义）；
3. 指标阈值（与以上两者可并存，作为证据补充）。

生成跨厂商对照表：

```bash
python -m aiops catalog matrix --vendor-matrix
```

## 7. 新增一个故障码的流程

1. 先判断是不是已有语义：能扩 `vendor_map` 就不新增码；
2. 在 `catalog.yaml` 追加条目（序号取域内最大值 +1）；
3. 写 `signals`（指标可测）和/或 `vendor_map`（日志可测），至少一条；
4. 写 `actions`（只能取动作词表）与 `verify`（S1/S2 必填）；
5. 写 `runbook`（值班照着能做）；
6. `python -m aiops catalog stats` 必须通过校验；
7. 补一条翻译单测（真实日志行）与（可选）一条策略单测；
8. 上真机后把 `assumed` 改成 `verified/community`，并在版本号上 +0.1。

## 8. 版本与兼容

- 目录版本在 `catalog.yaml: version`；新增码不破坏兼容，修改语义或等级属于破坏性变更。
- 告警规则由 `python -m aiops catalog prometheus` 生成，禁止手改导出的 YAML。
