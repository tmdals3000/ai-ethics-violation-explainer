"""실험 기록 표 형식 테스트."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("record_run", Path(__file__).parents[1] / "scripts/record_run.py")
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)


def test_fmt_rounds_half_up_and_keeps_text():
    assert rr.fmt(0.5155) == "0.516" and rr.fmt(0.6695) == "0.670"
    assert rr.fmt(2256) == "2,256" and rr.fmt("n/a") == "n/a" and rr.fmt(float("nan")) == "-"
