"""실험 결과를 results/experiments.csv 에 기록하고, 같은 값으로 README '실험 결과' 표를 만든다.

  python scripts/record_run.py record --version baseline-v0 --date 2026-11-08 --commit 1e2c384
  python scripts/record_run.py readme --version baseline-v0     # README 의 해당 버전 블록을 다시 쓴다

experiments.csv 는 행을 지우지 않고 쌓는다 (version·run_id 로 구분). README 표는 이 파일에서만 만든다.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, load_config, read_csv_safe  # noqa: E402

import pandas as pd  # noqa: E402

RUN_ID = {"llama_guard": "B-LG", "qwen3guard": "B-QG", "shieldgemma": "B-SG", "team": "T1"}
SRC = {"do_not_answer": "DNA", "jigsaw": "Jigsaw", "prosocial": "PD"}
CAUSE = {"retrieval": "규칙 찾기", "label_check": "라벨 확인", "rule_text": "규칙 문장", "judge_model": "판정 모델",
         "unclassified": "unclassified"}
COLS = ["version", "run_id", "date", "commit", "split", "table", "source", "model", "metric", "value", "note"]

# 실행 시간: 로그에서 잰 값 (results/model_status.csv, results/timings.csv, 실행 로그)
TIMES = [
    ("R1", "all-roberta-large-v1", "similarity 2,356 텍스트 × 81 노드", None, 93.4, "timings.csv p1"),
    ("B-LG", "llama_guard", "판정", 2355, 91, "실행 로그"),
    ("B-QG", "qwen3guard", "판정", 2206, 170, "실행 로그 (DNA dev 전체 첫 계산)"),
    ("B-SG", "shieldgemma", "판정", 15507, 3380, "5번 나눠 실행한 로그 합산(15+7+18+14+2.4분), 근사"),
    ("T1", "team", "판정", 15507, 1359.4, "model_status.csv (길이 기반 배치 적용 후)"),
]


def build_rows(cfg, version, date, commit):
    res, rows = cfg["_res"], []

    def add(run_id, table, source, model, metric, value, note=""):
        rows.append([version, run_id, date, commit, "dev", table, source, model, metric, value, note])

    # 실험 설정
    th = read_csv_safe(res / "team_thresholds.csv")
    tau, agg = cfg["retrieval"]["tau"], cfg["judge"]["aggregation"]
    for mo, rid in RUN_ID.items():
        spec = cfg["judge"]["models"][mo]
        add(rid, "setup", "", mo, "hf_id", spec["hf_id"])
        add(rid, "setup", "", mo, "prompt_version", "team_v1" if mo == "team" else "official")
        if mo == "team" and len(th):
            t = th[th.prompt_version == "team_v1"]
            add(rid, "setup", "", mo, "threshold", " / ".join(f"{SRC[s]} {v:g}" for s, v in zip(t.source, t.per_source_threshold)))
            add(rid, "setup", "", mo, "threshold_rule", "출처별 dev F1(위반) 최대, 격자 0.01")
        else:
            add(rid, "setup", "", mo, "threshold", f"{cfg['judge']['default_threshold']:g}")
            add(rid, "setup", "", mo, "threshold_rule", "고정 0.5")
        add(rid, "setup", "", mo, "tau", f"{tau:g}" if spec["uses_rules"] else "- (규칙 안 받음)")
        add(rid, "setup", "", mo, "aggregation", f"상위 {cfg['judge']['top_k_candidates']}후보 {agg}" if spec["uses_rules"] else "- (1회 판정)")
    # P1
    for r in read_csv_safe(res / "p1/metrics.csv").itertuples():
        for k in ["n", "top1_acc", "top3_acc", "efficiency_mean", "n_boundary"]:
            v = getattr(r, k, None)
            if pd.notna(v):
                add("R1", "p1", r.source, "all-roberta-large-v1", k, v, f"tau={tau:g}")
    # P2
    m = read_csv_safe(res / "p2/metrics_by_source.csv")
    for r in m[(m.condition == "base") & (m.split == "dev")].itertuples():
        for k in ["n", "n_pos", "roc_auc", "pr_auc", "f1", "kappa_human"]:
            add(RUN_ID[r.model], "p2", r.source, r.model, k, getattr(r, k), r.note if isinstance(r.note, str) else "")
    # 모델 간 κ
    for r in read_csv_safe(res / "p2/pairwise_kappa.csv").itertuples():
        add(f"{RUN_ID[r.model_a]}×{RUN_ID[r.model_b]}", "kappa_pair", r.source, f"{r.model_a}|{r.model_b}", "kappa", r.kappa)
    # P3
    s = read_csv_safe(res / "p3/cause_summary.csv")
    n = {(r.source, r.model, r.cause): r.n for r in s.itertuples()}
    for src in sorted(s.source.unique()):
        for mo in RUN_ID:
            for c in CAUSE:
                rule_free = c == "retrieval" and not cfg["judge"]["models"][mo]["uses_rules"]
                add(RUN_ID[mo], "p3", src, mo, c, "n/a" if rule_free else n.get((src, mo, c), 0),
                    "규칙을 받지 않는 모델에는 적용 안 함" if rule_free else "")
    # 실행 시간
    for rid, mo, what, n, sec, note in TIMES:
        add(rid, "time", "", mo, "seconds", sec, f"{what}; n_forward={n}; {note}" if n else f"{what}; {note}")
    return pd.DataFrame(rows, columns=COLS)


def record(cfg, a):
    p = cfg["_res"] / "experiments.csv"
    old = read_csv_safe(p)
    if len(old) and (old.version == a.version).any():
        raise SystemExit(f"{a.version} 은 이미 기록되어 있습니다. 이전 결과는 지우지 않으므로 새 version 을 쓰세요.")
    new = build_rows(cfg, a.version, a.date, a.commit)
    pd.concat([old, new], ignore_index=True)[COLS].to_csv(p, index=False)
    print(f"{p}: {len(new)}행 추가 ({a.version})")


def fmt(v):
    """정수는 천 단위 쉼표, 소수는 셋째 자리 사사오입 (0.5155 → 0.516)."""
    from decimal import ROUND_HALF_UP, Decimal
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return "-"
    if f.is_integer():
        return f"{int(f):,}"
    return str(Decimal(str(v)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def table(df, index, columns, order_cols=None, raw=False):
    p = df.pivot_table(index=index, columns=columns, values="value", aggfunc="first")
    if order_cols:
        p = p[[c for c in order_cols if c in p.columns]]
    head = "| " + " | ".join(list(p.index.names) + list(p.columns)) + " |"
    sep = "|" + "---|" * (len(p.index.names) + len(p.columns))
    lines = [head, sep]
    for idx, row in p.iterrows():
        idx = idx if isinstance(idx, tuple) else (idx,)
        lines.append("| " + " | ".join([str(i) for i in idx] + [str(v) if raw else fmt(v) for v in row]) + " |")
    return "\n".join(lines)


def readme_block(cfg, version):
    e = read_csv_safe(cfg["_res"] / "experiments.csv", keep_default_na=False)  # "n/a" 를 빈 값으로 바꾸지 않게
    e = e[e.version == version].copy()
    if e.empty:
        raise SystemExit(f"experiments.csv 에 {version} 이 없습니다.")
    e["출처"] = e.source.map(SRC).fillna("")
    date, commit = e.date.iloc[0], e.commit.iloc[0]
    st = e[e.table == "setup"].assign(run_id=lambda x: x.run_id)
    setup = table(st.rename(columns={"run_id": "run_id"}), ["run_id"], "metric",
                  ["hf_id", "prompt_version", "threshold", "threshold_rule", "tau", "aggregation"], raw=True).replace(
        "| run_id | hf_id |", "| run_id | 모델 |")
    setup_lines = setup.split("\n")
    setup_lines[0] = setup_lines[0].rstrip(" |") + " | 커밋 |"
    setup_lines[1] += "---|"
    setup_lines[2:] = [ln + f" `{commit}` |" for ln in setup_lines[2:]]
    p1 = table(e[e.table == "p1"], ["출처"], "metric", ["n", "top1_acc", "top3_acc", "efficiency_mean", "n_boundary"])
    p2 = table(e[e.table == "p2"], ["출처", "run_id"], "metric", ["n", "n_pos", "roc_auc", "pr_auc", "f1", "kappa_human"])
    kp = table(e[e.table == "kappa_pair"], ["run_id"], "출처", ["DNA", "Jigsaw", "PD"])
    p3e = e[e.table == "p3"].assign(metric=lambda x: x.metric.map(CAUSE))
    p3 = table(p3e, ["출처", "run_id"], "metric", list(CAUSE.values()))
    tm = e[e.table == "time"]
    tml = ["| run_id | 작업 | 시간 | 비고 |", "|---|---|---|---|"]
    for r in tm.itertuples():
        sec = float(r.value)
        tml.append(f"| {r.run_id} | {r.model} | {sec / 60:.1f}분 ({sec:,.0f}초) | {r.note} |")
    return f"""<!-- results:{version} -->
### {version.replace('baseline-v', 'Baseline v')} (dev, {date})

> ⚠ **모든 수치는 dev 결과다. test 는 아직 보지 않았다.** 검증 상태: `IS만` (임계값·τ 를 정한 dev 로 평가).
> 표본: DNA dev 전체 2,256건, Jigsaw·ProsocialDialog dev 각 50건. split·규칙 트리는 임시([ASSUMPTIONS](docs/ASSUMPTIONS.md)).
> 원자료: `results/experiments.csv` (version = `{version}`). 이 표는 `python scripts/record_run.py readme --version {version}` 로 만든다.

**실험 설정**

{chr(10).join(setup_lines)}

**P1 규칙 찾기** (R1: all-roberta-large-v1)

{p1}

**P2 판정 비교** (base 조건)

{p2}

**모델 간 Cohen's κ** (base 조건 예측)

{kp}

**P3 원인별 잘못된 판정 건수** (규칙 찾기는 규칙을 받는 모델에만 적용)

{p3}

**실행 시간** (RTX 3060 8GB)

{chr(10).join(tml)}
<!-- /results:{version} -->"""


def readme(cfg, a):
    p = ROOT / "README.md"
    s = p.read_text(encoding="utf-8")
    block = readme_block(cfg, a.version)
    start, end = f"<!-- results:{a.version} -->", f"<!-- /results:{a.version} -->"
    if start in s:
        s = s[:s.index(start)] + block + s[s.index(end) + len(end):]
    else:
        anchor = "<!-- results:new -->"
        if anchor not in s:
            raise SystemExit("README 에 '<!-- results:new -->' 자리표시가 없습니다.")
        s = s.replace(anchor, anchor + "\n\n" + block)  # 새 버전은 위에 쌓인다
    p.write_text(s, encoding="utf-8")
    print(f"README.md 의 {a.version} 블록을 갱신했습니다.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["record", "readme"])
    ap.add_argument("--version", required=True)
    ap.add_argument("--date")
    ap.add_argument("--commit")
    a = ap.parse_args()
    cfg = load_config()
    record(cfg, a) if a.cmd == "record" else readme(cfg, a)
