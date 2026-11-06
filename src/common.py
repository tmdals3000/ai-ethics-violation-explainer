"""설정 읽기와 경로. 다른 모든 모듈이 여기서 설정을 가져간다."""
import os
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    cfg = yaml.safe_load(open(path or ROOT / "configs/baseline.yaml", encoding="utf-8"))
    # .env 에서 HF_TOKEN 등을 읽는다 (토큰은 환경 변수로만 사용)
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    # HF 라이브러리를 import 하기 전에 캐시 위치를 고정한다
    os.environ.setdefault("HF_HOME", cfg["paths"]["hf_home"])
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    p = cfg["paths"]
    cfg["_raw"] = Path(p["raw_dir"])
    cfg["_data"] = Path(p["data_dir"])
    cfg["_ta"] = Path(p["ta_dir"])
    cfg["_res"] = ROOT / p["results_dir"]
    for d in (cfg["_data"], cfg["_res"]):
        d.mkdir(parents=True, exist_ok=True)
    return cfg


def read_csv_safe(path, **kw):
    """결과 파일이 없거나 비어 있으면(결과 0행) 빈 표를 돌려준다."""
    import pandas as pd
    try:
        return pd.read_csv(path, **kw)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def check_disk(min_gb=15, path="C:/"):
    """남은 디스크가 min_gb 미만이면 멈춘다 (사용자 요청)."""
    free = shutil.disk_usage(path).free / 1e9
    print(f"[disk] 남은 공간 {free:.1f} GB")
    if free < min_gb:
        raise SystemExit(f"디스크 {free:.1f} GB < {min_gb} GB. 멈추고 보고합니다.")
    return free
