"""診斷 09:02 大盤跳空: 比對 D1 昨收 (HF) 與 MIS 的 y (昨收) / o (今開)。用法: python3 scripts/diag_gap.py"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import orb_live as OL
from quotes import make_quotes
pool, last = OL.cmd_pool(argparse.Namespace(date=None))
print("D1 最後日期", last.date(), "流動股", len(pool))
snap = make_quotes().snapshot(pool.stock_id.tolist(), pool.market_type.tolist())
x = pool.merge(snap, on="stock_id", how="left")
x["d1_vs_y"] = x.pc / x.y - 1; x["gap_d1"] = x.o / x.pc - 1; x["gap_y"] = x.o / x.y - 1
print("MIS 日期", snap.mis_d.dropna().mode().tolist(), "時間樣本", snap.mis_t.dropna().head(3).tolist())
print(f"平均跳空 (D1 昨收) {x.gap_d1.mean()*100:.2f}%   (MIS y) {x.gap_y.mean()*100:.2f}%")
print("D1 昨收 / MIS y 分布:\n", x.d1_vs_y.describe())
print(x[["stock_id", "name", "last_date", "pc", "y", "o", "z", "d1_vs_y"]].head(15).to_string(index=False))
d1 = OL.load_d1(since="2026_10")
ids = x.stock_id.head(6).tolist()
print("D1 最近 4 天 (前 6 檔):")
print(d1[d1.stock_id.isin(ids)].sort_values(["stock_id", "date"]).groupby("stock_id").tail(4)[["stock_id", "date", "open", "high", "low", "close", "volume"]].to_string(index=False))
