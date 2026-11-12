# vLLM 推理任务的迁移与回迁（实测）

上一轮（docs/07）用的是自写的小推理 worker；这一轮换成**真实 vLLM 引擎**，
并且做**两次迁移**（源 → 备 → 回源），重点回答两个问题：

1. 迁移/回迁之后，**结果会不会偏移**？
2. 迁移后的**效果是否理想**，代价在哪？

结论先说：**没有偏移**（同一 prompt 在两张卡上逐字一致，回放 5/5 匹配），
但代价是**一次引擎重启约 40 秒**，MTTR 几乎全部由它决定。

## 1. 环境

| 项 | 值 |
| --- | --- |
| 机器 | 2×RTX 4090（PCIe，无 NVLink） |
| vLLM | 0.33.0 |
| PyTorch | 2.14.0+cu130（vLLM 自动安装，替换了镜像里的 2.12） |
| 模型 | Qwen2.5-0.5B-Instruct（`/root/models/qwen2.5-0.5b`） |
| 服务 | `vllm.entrypoints.openai.api_server`，`--gpu-memory-utilization 0.45 --enforce-eager` |
| 采样 | `temperature=0, seed=0`（确定性） |

## 2. 结构

```text
controller（本机）
  ├── SSH ──> c-gpu0：demo.inference.vllm_worker
  │              └── 子进程：vLLM api_server（:18200）
  └── SSH ──> c-gpu1：demo.inference.vllm_worker
                 └── 子进程：vLLM api_server（:18201）
```

worker 负责：

- 拉起/守护 vLLM 引擎，等 `/health` 200；
- 通过 `/v1/chat/completions` 逐条处理任务（本轮 120 条算术请求，5 QPS 限速）；
- 每完成一条就原子提交 `progress.json` + 追加 `results.jsonl`；
- **每次启动都做三项一致性检查**：
  - `probe`：固定 prompt 的输出哈希；
  - `calibration`：10 道算术题的正确率（衡量"效果"）；
  - `replay`：随机重放 5 条已完成样本，逐字比对（衡量"是否偏移"）；
- 故障注入 `engine-kill`：直接 `SIGKILL` vLLM 子进程，模拟推理引擎崩溃。

## 3. 结果（两跳，含回迁）

| 跳 | 方向 | 故障码 | 故障时进度 | 保存点 | RPO | MTTD | MTTR | 验证 | 回放 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | c-gpu0 → c-gpu1 | `UFC-INF-001` | 40 | 40 | 0 | 1.65s | 46.11s | ✅ | 5/5 |
| 2（回迁） | c-gpu1 → c-gpu0 | `UFC-INF-001` | 79 | 80 | 0 | 0.77s | 45.82s | ✅ | 5/5 |

- 总 RPO：**0 条**（每条请求结束即提交，崩溃只影响在途的那一条）；
- 总恢复时延：91.9s；总墙钟约 150s（含 3 次引擎冷启动）；
- 两次故障都被翻译成 `UFC-INF-001 推理实例健康检查失败`（证据：`Connection refused`）。

## 4. 结果有没有偏移？

三条独立证据，全部指向"没有"：

| 检查 | 结果 |
| --- | --- |
| probe 输出哈希（3 次启动：c-gpu0 / c-gpu1 / c-gpu0） | `670d9743542c` ×3，**完全一致** |
| replay：在目标卡上重算已完成样本，与源卡结果逐字比对 | 第一跳 5/5、回迁 5/5，**全一致** |
| 校准集正确率（10 题算术） | c-gpu0 0.6 → c-gpu1 0.6 → c-gpu0 0.6，**稳定** |

为什么能逐字一致：同型号 GPU（都是 4090）、同一份权重、`temperature=0`、固定 `seed`、
相同 prompt 模板；vLLM 的调度不会改变单条请求的贪心结果。

**但这不是普遍结论**：换不同架构的卡（如 4090 → A100）、换 vLLM/驱动版本、
或开 `temperature>0`，都可能出现差异。要判定"是否偏移"，必须像这里一样把
**校验集 + 回放**做进流程，而不是假设"同模型就一样"。

## 5. 效果是否理想？

- 校准集正确率 0.6 —— 这是 0.5B 模型本身的能力上限附近，**迁移前后没有下降**；
  它在这里的作用是"相对基线"，不是绝对质量指标。
- 120 条任务全部完成，结果下标 0..119 连续、无重复（完整性校验通过）。
- 真正不理想的是**恢复时延**：单跳 MTTR ~46s，其中引擎启动 + 模型加载约占 40s，
  搬运状态只要 3 个文件（~4KB）与 2 秒。

启示：

1. 推理迁移的瓶颈是**目标端冷启动**，不是状态搬运；
2. 要压 MTTR，需要**预热/常驻备用实例**或镜像缓存，而不是优化传输；
3. 若要连 KV cache 一起迁，代价和复杂度会显著上升（见 §7）。

## 6. 这轮踩到的坑（都已修进代码）

| 问题 | 现象 | 处理 |
| --- | --- | --- |
| `ninja` 不在 PATH | vLLM flashinfer JIT 报 `FileNotFoundError: ninja` | 启动时把 conda bin 前置到 PATH |
| flashinfer 首次 JIT | 首次启动要几分钟编 sampling 核 | 设 `VLLM_USE_FLASHINFER_SAMPLER=0`，改用 PyTorch 采样（对 greedy 无影响） |
| 两节点共用 vLLM 端口 | 回迁时新引擎被另一节点的清理逻辑杀掉 | 每个节点独立 `vllm_port` |
| 引擎残留占显存 | 新引擎报 "Free memory ... less than desired" | 启动前 `nvidia-smi --query-compute-apps` 清理 + 引擎独立进程组 |
| 控制文件残留 | 新 worker 读到上一轮的 `force_exit` 秒退 | 启动前清空 `ctl/` |
| 校验竞态 | 服务推进太快，`rows` 与 `committed` 差 >1 被误判 | 只要求"已提交部分连续且唯一"，允许在途一段 |

## 7. 与生产 vLLM 的差距

- **KV cache 没有迁**：本轮是"重放请求"，不是"搬走缓存"。真要迁，需要把
  prefix cache/KV block 导出到对象存储或直连传输，成本高一个量级。
- **没有网关摘流**：真实在线服务要先在网关把权重置 0、等在途请求结束再停实例，
  否则切换瞬间会丢请求；本项目里由控制面"停→搬→起"近似。
- **没有压测**：5 QPS 是为了让故障点可控，不代表生产吞吐。
- 只验证了 0.5B 模型；7B/70B 的冷启动时间会显著拉长 MTTR。

## 8. 复现

```bash
# 模型与 vLLM 需先装好（AutoDL 镜像 + aliyun/tuna 镜像源）
python -m demo.inference.controller --config .local/hosts_vllm.yaml \
    --out results/vllm-migration
```

配置要点：`job.backend: vllm`、`job.model`、每个节点独立的 `worker_port`/`vllm_port`、
`run.failback: true` 与 `run.failback_fault`（第二跳的注入）。
