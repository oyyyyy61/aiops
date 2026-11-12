#!/usr/bin/env bash
# 2x4090 环境的一键验证脚本（Linux）。
#
#   bash scripts/run_2x4090.sh              # 装依赖 + 单测 + 全部场景
#   bash scripts/run_2x4090.sh --no-install # 跳过依赖安装
#
# 注意：脚本会用到 nvidia-smi，但不会主动改驱动/断电；限频注入需要 root。
set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_INSTALL=0
[[ "${1:-}" == "--no-install" ]] && SKIP_INSTALL=1

if [[ $SKIP_INSTALL -eq 0 ]]; then
  python -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install -U pip
  python -m pip install -r requirements.txt
  # 4090 是 Ada（sm_89），cu124 及以上均可
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
fi

echo "== 硬件 =="
nvidia-smi --query-gpu=index,name,memory.total,ecc.mode.current --format=csv || true

echo "== 单元测试 =="
python -m unittest discover -s tests -v

echo "== 故障码目录 =="
python -m aiops catalog stats

echo "== 仿真场景（无 GPU 也可跑）=="
python -m aiops demo demo/scenarios/0*.yaml --out results

echo "== 真实 DDP 场景（2 卡 NCCL）=="
# torchrun 在 Linux 上可用；训练器检测到 CUDA 会自动切 NCCL
python -m aiops demo demo/scenarios/1*.yaml --out results

echo "== 报告 =="
cat results/report.md

cat <<'EOF'

下一步（需要 root / 谨慎操作）：
  1. 慢节点：nvidia-smi -i 1 -lgc 300,300   # 锁住卡 1 频率
     然后重跑 03-straggler-shrink，观察是否选 shrink
  2. 掉卡：systemctl restart nvidia-persistenced 或卸载/重载内核模块
     （会中断该机所有 GPU 作业，务必在可牺牲节点上做）
  3. 用真实 Xid 日志回填 UFC 映射并更新 confidence
EOF
