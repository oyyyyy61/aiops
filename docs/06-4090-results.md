# 真实 4090 实测（2026-11-13）

这是把原型从"仿真闭环"推到**真实 GPU + 真实进程 + 真实信号**的一轮实验。
先说结论：控制闭环在真机上成立，并且真机暴露了三个仿真掩盖不了的缺陷（见 §5）。

## 1. 环境

三台租用实例（AutoDL）：A/B 是 1 卡/节点，C 是双卡机（本文 §4 单独讲 C）：

| 节点 | 容器 | GPU | 驱动 | Python | torch | 访问方式 |
| --- | --- | --- | --- | --- | --- | --- |
| A | autodl-container-8bd74a93… | RTX 4090 24GB（23.5 GiB 可用） | 580.76.05 | 3.12.3 | 2.12.1+cu130 | ssh -p 30782 |
| B | autodl-container-321d48b5… | RTX 4090 24GB | 595.71.05 | 3.12.3 | 2.12.1+cu130 | ssh -p 21868 |
| C | autodl-container-8f1a459e… | **RTX 4090 ×2**（PCIe NODE，无 NVLink） | 595.71.05 | 3.12.3 | 2.12.1+cu130 | ssh -p 31141 |

约束（直接决定了能做到什么程度）：

- **每台只有 1 张卡**，且容器内看不到对端，跨节点没有直连 L3（AutoDL NAT）；
  所以"2 卡 DDP"在本轮不可得，多 rank 行为仍以本机 CPU/gloo 双 rank 验证。
- 容器内 `nvidia-smi -lgc` 返回 `rc=4`（无 CAP_SYS_ADMIN），**不能锁频/改功耗**，
  硬件降频这条注入路径只能在自有 4090 机器上做。
- RTX 4090 无 ECC、无 NVLink：ECC/NVLink 路径仍然只有回放证据。

## 2. 基线（同一套代码在两台机器上跑）

`python -m experiments.baseline_4090`，batch=256，60 step 预热后统计：

| 形状 | 参数 | A step（p50/p95） | A steps/s | B step（p50/p95） | B steps/s |
| --- | --- | --- | --- | --- | --- |
| 512×3 | 0.59M | 0.85 / 0.99 ms | 1182.7 | 0.87 / 0.91 ms | 1150.4 |
| 1024×4 | 3.28M | 0.94 / 0.99 ms | 1060.5 | 0.99 / 1.01 ms | 1011.5 |
| 2048×4 | 12.86M | 1.05 / 1.16 ms | 953.5 | 1.10 / 1.12 ms | 909.3 |

检查点开销（`CheckpointStore` 原子写 + torch.save）：

| 参数 | 文件大小 | A 保存 | A 加载 | B 保存 | B 加载 |
| --- | --- | --- | --- | --- | --- |
| 3.28M | 12.5 MB | 31.6 ms | 10.2 ms | 30.4 ms | 9.4 ms |
| 12.86M | 49.1 MB | 118.7 ms | 41.5 ms | 110.8 ms | 37.1 ms |

两个观察：

1. **同型号不同机器的差异有 4%–12%**（4096 GEMM：A 2.49ms vs B 2.79ms）。
   这意味着慢节点阈值不能按"同型号就该一样"来定，至少留 20%–30% 余量，
   或者用每机自身基线而不是跨机中位数。
2. **检查点开销相对小模型是致命的**：12.86M 模型存一次 ≈ 113 个 step 的计算量。
   所以场景实验里把 step 做成 14ms（batch=8192）、保存间隔 20 步，
   让"保存开销 / 训练时间"落在 ~30% 的可观测区间；这也说明**大模型必须异步/分片保存**。

## 3. 故障处置（真实信号注入）

四个场景都用真实进程 + 真实信号，不用仿真训练器：

| 场景 | 注入 | 结果 | 停机 | MTTR | RPO |
| --- | --- | --- | --- | --- | --- |
| 30 baseline | 无 | 200 步跑完，0 故障、0 误动作 | — | — | — |
| 31 SIGKILL | `kill -9`（模拟掉卡/被内核杀） | restart_in_place | 0.00s | 3.93s(A) / 4.09s(B) | 7(A) / 6(B) 步 |
| 32 SIGTERM | 优雅终止（模拟抢占） | restart_in_place，紧急保存 | 0.00s | 3.61s(A) / 4.22s(B) | **0 步** |
| 33 SIGSTOP | 冻结进程（模拟 GPU 挂死） | restart_in_place，强杀进程组 | 4.01s | 8.23s(A) / 8.49s(B) | 20 步 |

三点值得记下来：

1. **紧急保存把 RPO 从 7 步降到 0 步**（31 vs 32），代价是停机窗口多等一次保存；
   这是"值不值得等 `grace_s`"的定量依据。
2. **SIGSTOP 是最接近 GPU 挂死的形态**：进程还在、显存还占着、心跳停了，
   应用层收不到任何信号，只能靠 `progress_age` 发现；此时**无法紧急保存**，
   RPO 就等于一个检查点间隔（20 步）。
3. 冻结进程无法自行退出，必须强杀**整个进程组**，否则 launcher 死了、rank 还挂着 GPU
   （见 §5 的缺陷 2）。

### 3.1 SIGKILL 时间线（A）

```text
step 47  注入 SIGKILL（python -m demo.torch_launcher 的 rank-0）
+0.00s   控制面检测到进程退出 rc=247（= -9），合成 UFC-JOB-001
+0.00s   决策 restart_in_place，读取最近检查点 step 40（RPO=7）
+0.00s   拉起新进程，加载 state.pt（41.5ms）
+3.9s    验证通过（step > 43，故障未复发）
```

### 3.2 SIGSTOP 时间线（A）

```text
step 40  注入 SIGSTOP；心跳停在 step 40，进程与显存都还在
+8.0s    job.progress_age_s 连续 2 次 > 8s，判定 UFC-JOB-001（演练阈值 8s）
+8.0s    请求优雅停止（控制文件），进程已被冻结读不到
+11.0s   超过 grace=3s，写 force_exit + 强杀进程组
+11.0s   从检查点 step 20 恢复（RPO=20），重启
+15.0s   验证通过
```

## 4. 两台机器的一致性

同一套场景在 A、B 上各跑一遍，策略选择、恢复结果完全一致，只有毫秒级差异：

| 指标 | A | B | 差异 |
| --- | --- | --- | --- |
| SIGKILL MTTR | 3.93s | 4.09s | +4% |
| SIGTERM MTTR | 3.61s | 4.22s | +17% |
| SIGSTOP MTTR | 8.23s | 8.49s | +3% |
| 恢复成功率 | 3/3 | 3/3 | 一致 |

这说明策略层对机器性能差异不敏感（决策用的是故障语义与检查点状态，不是绝对耗时），
但**检测阈值**必须按机器校准——这正是 §2 的结论。

重复性：在 A 上把 SIGKILL 场景连跑 3 次，RPO 分别为 6 / 7 / 8 步
（取决于杀进程落在两个检查点之间的位置），MTTR 3.88–4.00s（波动 <3%），
三次都通过验证。

## 4bis. 双卡 4090（节点 C）：真实 NCCL 作业

C 是 2×4090、`nvidia-smi topo -m` 显示两卡之间是 `NODE`（走 PCIe，**没有 NVLink**），
这正好是"便宜双卡机"的真实形态。

### 4bis.1 基线

`python -m demo.torch_launcher --world-size 2 --module experiments.ddp_bench`：

| 指标 | 值 |
| --- | --- |
| 模型 | 12.86M 参数（batch 256，2 rank DDP/NCCL） |
| 步耗时 | 6.25 ms（p50）/ 6.62 ms（p95） |
| 单次 all-reduce（51MB） | 0.91 ms |
| 有效带宽（busbw） | 56.2 GB/s |

对比单卡同模型 1.05ms/步：**双卡 DDP 每步多出来的 5ms 主要是 Python/DDP 与 PCIe 通信开销**，
all-reduce 本身只占 15%。所以"双卡加速比"在这种小模型上几乎为 1——慢节点诊断要看
per-rank 前向时间，而不是总步耗时。

### 4bis.2 故障处置

| 场景 | 注入 | 结果 | 停机 | MTTR | RPO | 仅周期检查点会丢 |
| --- | --- | --- | --- | --- | --- | --- |
| 40 baseline | 无 | 200 步，0 故障 | — | — | — | — |
| 41 Xid 43 + kill rank1 | `SIGKILL` + Xid 43 | **shrink 2→1** | 4.01s | 7.87s | **0 步** | 20 步 |
| 42 单卡降频 | rank1 前向 ×8 | **shrink 2→1** | 1.56s | 5.63s | **0 步** | 17 步 |

两个结论：

1. **没有备机时，"缩容到 1 卡"是正确的处置**。双卡机坏一张卡，把作业重启回 2 卡只会
   再挂一次；丢掉坏 rank 继续跑，比反复重启划算。场景 41 里 rank 1 被 SIGKILL 后，
   rank 0 立刻卡在 NCCL all-reduce 里，控制面必须**强制清理整个进程组**才能重启。
2. **真实 DDP 里"总步耗时"会被最慢的卡拖平**，两边的 `step_time` 几乎一样，
   跨 rank 比总耗时是无效信号。所以训练器额外上报 `forward_s`（纯前向时间），
   在线比较的是 per-rank 前向时间——这是本轮为此改的代码。

### 4bis.3 由此修掉的两个假阳性

双卡实验里，慢节点检测出现了两次误报，都是"阈值在小尺度下失真"：

1. rank 被 kill 的瞬间，它最后一条心跳被当成"慢 rank"（1.35x）→ 误报 JOB-007。
   修复：心跳超过 2s 未更新的 rank 不参与偏斜比较。
2. 步耗时只有 1ms 时，"1.35 倍"只是 0.35ms 抖动。修复：除比值外要求
   **绝对增量 ≥2ms**，并且参照系用最小 rank 而不是中位数（2 个 rank 时中位数上限 2.0x）。

修复后重跑：41 场景从 2 条故障降为 1 条（只剩 Xid），42 场景的慢节点仍能稳定检出。

## 5. 真机暴露的三个缺陷

仿真环境里这三个都不会暴露，是这轮实验最实在的产出：

1. **"优雅退出"被当成"训练完成"**。训练器收到 SIGTERM 后写 `state=done`，
   控制面看到 done 就判定作业完成，根本不进入恢复流程（场景 32 第一版 0 故障）。
   修复：优雅停止写 `state=stopped`，只有真正跑完才写 `done`。
   教训：**作业状态机里"正常结束"和"被中断"必须是两个状态**。
2. **强杀只杀了 launcher，留下持有 GPU 的孤儿 rank**。SIGSTOP 的 rank 杀不掉，
   launcher 退出后它继续占着 24GB 显存。修复：POSIX 用 `start_new_session` +
   `killpg`，Windows 用 `taskkill /T /F`，并在强杀前写 `force_exit` 给看门狗。
3. **换环境就换坑**：容器里没有 `python3`（要显式用 `/root/miniconda3/bin/python`）；
   `torchrun --standalone` 在 Windows 的 torch 构建上因 libuv/TCPStore 起不来
   （本机改用 `FileStore` 启动器，Linux 上 torchrun 仍可用）；
   上传的 bash 脚本带 CRLF 会直接语法错误（已在工具链里统一 LF）。

## 6. 本轮没有验证的

诚实清单：

- 8 卡以上的规模化行为（双卡已验证，见 §4bis）；
- 时钟锁定/功耗限制导致的硬件降频（容器内无权限，只能用品模型用"多跑几次前向"模拟）；
- ECC、NVLink、HCCS、IB 的物理故障（4090 无 ECC/NVLink，且没有第二台异构设备）；
- 大规模（>8 卡）下的误报率与策略收敛性；
- 检查点恢复后的**精度等价性**（只验证了步数推进与故障不复发）。

## 7. 复现命令

```bash
# 两台机器上（把 repo 放到 /root/aiops）
/root/miniconda3/bin/python -m unittest discover -s tests
/root/miniconda3/bin/python -m experiments.baseline_4090 --steps 40 --out results/4090-baseline.json
/root/miniconda3/bin/python -m aiops demo 'demo/scenarios/3*.yaml' --out results
cat results/report.md
```
