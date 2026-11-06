"""Baseline 전체 흐름 실행기.

예) python scripts/run.py --steps data,p1,judge,p2,p3,p4
    python scripts/run.py --steps p2 --allow-test   # test 지표까지 (명시할 때만)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import check_disk, load_config  # noqa: E402

STEPS = ["data", "p1", "judge", "p2", "p3", "p4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-test", action="store_true", help="test split 지표 계산 허용")
    ap.add_argument("--models", default=None, help="판정 모델 일부만 (쉼표 구분)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    cfg["_allow_test"] = a.allow_test
    timings = []
    for step in a.steps.split(","):
        check_disk()
        t0 = time.time()
        print(f"\n===== {step} =====")
        if step == "data":
            from src import data
            data.prepare(cfg)
        elif step == "p1":
            from src import retrieval
            retrieval.run(cfg)
        elif step == "judge":
            from src import judges
            judges.run(cfg, a.models.split(",") if a.models else None)
        elif step == "p2":
            from src import metrics
            metrics.run(cfg)
        elif step == "p3":
            from src import diagnose
            diagnose.run(cfg)
        elif step == "p4":
            from src import effects
            effects.run(cfg)
        else:
            raise SystemExit(f"모르는 단계: {step}")
        timings.append((step, round(time.time() - t0, 1)))
        print(f"[time] {step}: {timings[-1][1]} s")
    check_disk()
    log = cfg["_res"] / "timings.csv"
    new = not log.exists()
    with open(log, "a", encoding="utf-8") as f:
        if new:
            f.write("timestamp,step,seconds,dev_sample_per_source\n")
        for s, sec in timings:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{s},{sec},{cfg['data']['dev_sample_per_source']}\n")


if __name__ == "__main__":
    main()
