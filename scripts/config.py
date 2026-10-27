"""v2 共用設定. data/m1, data/d1 是 macOS 替身 (alias), Python 無法跟隨, 所以直接指到實際路徑.
雲端 (GitHub Actions) 沒有 v1 的原始資料夾: RAW 不存在時, 只用 data/live/ (orb_live.py refresh 從 HF 下載)。"""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RAW = Path(os.environ.get("ORB_RAW", ROOT.parent / "quant_trading" / "data" / "raw"))   # d1/ m1/ meta/
PROC = ROOT / "data" / "processed"; PROC.mkdir(parents=True, exist_ok=True)
STATS = ROOT / "stats"; STATS.mkdir(parents=True, exist_ok=True)
HAS_RAW = (RAW / "d1").exists()           # 本機研究用; 雲端為 False

HF_REPO = "raywu918daytrade/daytrade"     # HF dataset
HF_PREFIX = "db/orb"                      # 本策略在 HF 的資料夾
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{HF_PREFIX}"

COST = 0.0023            # 現股當沖來回成本 (手續費折讓後 + 交易稅 0.15%), 與 v1 一致
MIN_DV20 = 1e8           # 流動性: 前 20 日均成交金額 ≥ 1 億
MIN_PRICE = 10.0         # 昨收 ≥ 10 元
EXPLORE_END = "2023-12-31"   # 探索期 2020-01 ~ 2023-12, 驗證期 2024-01 ~
