#!/usr/bin/env python3
"""檢查 env 檔裡的 Hugging Face token: 屬於哪個帳號、權限, 能不能寫入 raywu918daytrade/daytrade。不會印出 token 值。
用法: python3 scripts/hf_check.py <env 檔> [<env 檔> ...]"""
import sys
from pathlib import Path

TARGET = "raywu918daytrade/daytrade"


def read_env(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        if line.startswith("export "): line = line[7:]
        k, v = line.split("=", 1); out[k.strip()] = v.strip().strip("\"'")
    return out


def main():
    from huggingface_hub import HfApi
    files = sys.argv[1:] or [str(Path(__file__).resolve().parents[2] / n) for n in ["daytrade.env", "HF.env"]]
    for f in files:
        if not Path(f).exists(): print(f"{f}: 找不到"); continue
        for k, v in read_env(f).items():
            if "HF" not in k.upper() or not v or not v.startswith("hf_"): continue
            try:
                w = HfApi(token=v).whoami(); at = w.get("auth", {}).get("accessToken", {}); role = at.get("role")
                if role == "write": can = "可以 (write token)" if w.get("name") == TARGET.split("/")[0] or TARGET.split("/")[0] in [o.get("name") for o in w.get("orgs", [])] else "不行 (帳號不同)"
                elif role == "fineGrained":
                    sc = (at.get("fineGrained") or {}).get("scoped", [])
                    ok = any(s.get("entity", {}).get("name") in (TARGET, TARGET.split("/")[0]) and any("write" in p for p in s.get("permissions", [])) for s in sc)
                    can = "可以 (fine-grained 有開寫入)" if ok else "不行 (fine-grained 沒開這個 dataset 的寫入)"
                else: can = "不行 (唯讀)"
                print(f"{Path(f).name}:{k} → 帳號 {w.get('name')} | 角色 {role} | 寫入 {TARGET}: {can}")
            except Exception as e:
                print(f"{Path(f).name}:{k} → 無效或無法連線 ({type(e).__name__})")


if __name__ == "__main__":
    main()
