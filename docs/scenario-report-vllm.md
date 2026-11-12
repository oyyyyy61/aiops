# 推理任务迁移报告：vllm-demo（vllm）

- 迁移次数：2（c-gpu0->c-gpu1 → c-gpu1->c-gpu0）
- 故障码：UFC-INF-001、UFC-INF-001
- 总丢失样本（RPO）：0
- 总恢复时延：91.93s / 总耗时 151.4s
- 验证：全部通过

## 各跳明细

| 跳 | 方向 | 故障码 | 故障进度 | 保存点 | RPO | MTTD | MTTR | 验证 | 回放一致 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | c-gpu0->c-gpu1 | UFC-INF-001 | 40 | 40 | 0 | 1.65s | 46.11s | ✅ | 5/5 |
| 2 | c-gpu1->c-gpu0 | UFC-INF-001 | 79 | 80 | 0 | 0.77s | 45.82s | ✅ | 5/5 |

## 结果一致性（是否偏移）

- 各节点 probe 哈希：['670d9743542c', '670d9743542c', '670d9743542c']
- probe 完全一致：True
- 校准集准确率：[{'node': 'c-gpu0', 'accuracy': 0.6}, {'node': 'c-gpu1', 'accuracy': 0.6}, {'node': 'c-gpu0', 'accuracy': 0.6}]
- 回放比对：[{'node': 'c-gpu0', 'matched': 0, 'checked': 0}, {'node': 'c-gpu1', 'matched': 5, 'checked': 5}, {'node': 'c-gpu0', 'matched': 5, 'checked': 5}]
- 回放全部一致：True
