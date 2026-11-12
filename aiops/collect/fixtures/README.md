# Fixture 来源说明

这个目录里的样本分两类，**必须分清**，否则"验证过"会被误读成"真机跑过"。

## 真实设备输出

| 文件 | 来源 | 说明 |
| --- | --- | --- |
| `nvidia_smi_q.xml` | 本机 RTX 5060，驱动 610.88 / CUDA 13.3 实跑 `nvidia-smi -q -x` | 结构真实，但消费级卡无 ECC，相关字段为 `N/A` |
| `dmesg_nvidia_xid.txt` | 按真实内核打印格式整理（Xid 13/31/43/48/74/79/92/94/95/109） | 用于验证 Xid 解析与映射，不是某一次真实故障的完整日志 |

## 合成样本（结构真实、数值构造）

| 文件 | 用途 | 备注 |
| --- | --- | --- |
| `nvidia_smi_q_a100_ecc.xml` | 由 `scripts/make_fixtures.py` 从真实 RTX XML 改写而来 | 把 ECC/行重映射/温度等字段改成 A100 档位，用于验证 ECC 解析路径 |
| `amd_rocm_smi.json` | `rocm-smi --showallinfo --json` 的字段结构（MI300X） | 键名与真实输出一致，数值为构造 |
| `dmesg_amd.txt` | amdgpu 内核日志常见措辞（ring timeout / RAS UE / XGMI） | 用于验证 AMD 翻译 |
| `ascend_npu_smi_info_910b.txt` | `npu-smi info` 的两行式表格（910B2 ×4） | 表格结构与真实一致，数值构造 |
| `ascend_npu_smi_health_alarm.txt` | `npu-smi info -t health` 的 Alarm 输出 | Error Code 为示意值 |
| `ascend_npu_smi_usages.txt` | `npu-smi info -t usages` 键值输出 | 用于验证 usage 解析 |
| `slog_ascend.txt` | 昇腾 slog 的典型错误行（HCCL timeout / HBM ECC / HCCS） | 用于验证昇腾翻译 |

## 重要限制

1. RTX 4090 / 5060 **不支持 ECC，也没有 NVLink**。这两类故障在 2 卡阶段
   只能靠回放验证；A100/H100/910B 到货后才能做物理注入。
2. AMD / 昇腾的 `vendor_map` 置信度为 `community` / `assumed`，
   真机上必须用厂商手册里的原始 Error Code 回填，再改成 `verified`。
3. 重新生成 A100 样本：`python scripts/make_fixtures.py`。
