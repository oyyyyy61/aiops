# 推理任务跨机迁移：infer-demo

- 故障：`UFC-GPU-002` 非法显存访问 / 上下文异常（S2）
- 源机：c-gpu0 → 目标机：c-gpu1
- 故障时已完成：400 样本；保存点：400
- 丢失样本（RPO）：0
- 检测时延（MTTD）：1.77s；保存耗时：8.9s
- 恢复时延（MTTR）：10.88s；总耗时：36.6s
- 源机吞吐：19.04 samples/s；目标机吞吐：25.23 samples/s
- 完整性：{'ok': True, 'rows': 440, 'committed': 432, 'contiguous': True, 'duplicates': False, 'in_flight': 8, 'first': 0, 'last': 439}
- 验证：通过

## 时间线

| 时间 | 事件 | 详情 |
| --- | --- | --- |
| 1794503799.64 | connect | source=c-gpu0 target=c-gpu1 |
| 1794503801.468 | worker_start | c-gpu0 pid=8698 resume=False |
| 1794503810.509 | baseline | 源机吞吐 19.0 samples/s |
| 1794503825.342 | fault_detected | 进程退出 rc=3（已完成 400 样本）；[worker] FATAL AcceleratorError: CUDA error: device-side assert triggered |
| 1794503825.343 | translated | UFC-GPU-002 非法显存访问 / 上下文异常 [S2] |
| 1794503825.704 | state_saved | done=400 |
| 1794503831.554 | state_transferred | 8 个文件 / 17618 字节: state/progress.json, state/results.jsonl, state/ckpt/ckpt-000064.json, state/ckpt/ckpt-000128.json, st |
| 1794503831.882 | quarantine | c-gpu0 已标记隔离 |
| 1794503832.161 | worker_start | c-gpu1 pid=8920 resume=True |
| 1794503834.416 | resumed | c-gpu1 从 400 继续 |
| 1794503836.225 | verified | done=432 integrity={'ok': True, 'rows': 440, 'committed': 432, 'contiguous': True, 'duplicates': False, 'in_flight': 8,  |
| 1794503836.563 | target_stopped | 演示结束后停止目标机 worker |
