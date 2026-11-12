# 自动运维场景实验报告

生成时间：2026-11-12 21:09:37

| 场景 | 故障数 | 恢复 | 成功 | 平均停机(s) | 平均 MTTD(s) | 平均 MTTR(s) | 平均 RPO(步) | 仅周期检查点 RPO | 期望达标 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00-baseline | 0 | 0 | 0 | None | None | None | None | None | ✅ |
| 01-xid79-evacuate | 1 | 1 | 1 | 0.11 | 0.0 | 0.7 | 0.0 | 0.0 | ✅ |
| 02-hang-restart | 2 | 1 | 1 | 6.05 | 0.06 | 6.64 | 2.0 | 2.0 | ✅ |
| 03-straggler-shrink | 1 | 1 | 1 | 0.36 | 0.55 | 0.91 | 0.0 | 2.0 | ✅ |
| 04-ecc-evacuate | 1 | 1 | 1 | 0.11 | 0.0 | 0.71 | 0.0 | 5.0 | ✅ |
| 05-no-ckpt-failfast | 1 | 0 | 0 | None | None | None | None | None | ✅ |
| 10-torch-baseline | 0 | 0 | 0 | None | None | None | None | None | ✅ |
| 11-torch-xid79 | 1 | 1 | 1 | 7.06 | 0.0 | 12.02 | 0.0 | 4.0 | ✅ |

合计：8 个场景，5 次恢复，5 次通过验证。
