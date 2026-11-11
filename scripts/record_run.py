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

# 실행 시간: 로그에서 잰 값. baseline-v0 은 여러 번 나눠 돌린 로그를 손으로 합산한 값이라 그대로 적어 둔다.
# 그 뒤 version 은 results/timings.csv (마지막 실행) 와 results/model_status.csv 에서 읽는다 (times_from_files).
TIMES_V0 = [
    ("R1", "all-roberta-large-v1", "similarity 2,356 텍스트 × 81 노드", None, 93.4, "timings.csv p1"),
    ("B-LG", "llama_guard", "판정", 2355, 91, "실행 로그"),
    ("B-QG", "qwen3guard", "판정", 2206, 170, "실행 로그 (DNA dev 전체 첫 계산)"),
    ("B-SG", "shieldgemma", "판정", 15507, 3380, "5번 나눠 실행한 로그 합산(15+7+18+14+2.4분), 근사"),
    ("T1", "team", "판정", 15507, 1359.4, "model_status.csv (길이 기반 배치 적용 후)"),
]


def times_from_files(cfg):
    """마지막 실행의 단계별 시간(timings.csv)과 모델별 새 forward 수·시간(model_status.csv). 캐시에 있던 판정은 다시 계산하지 않는다."""
    res, out = cfg["_res"], []
    tm = read_csv_safe(res / "timings.csv")
    if len(tm):
        last = tm.groupby("step").tail(1)
        for r in last.itertuples():
            out.append((f"step:{r.step}", r.step, "단계 전체 (마지막 실행)", None, float(r.seconds), f"timings.csv {r.timestamp}; {r.dev_sample_per_source}"))
    st = read_csv_safe(res / "model_status.csv")
    for r in st[st.status == "ok"].itertuples() if len(st) else []:
        out.append((RUN_ID.get(r.model, r.model), r.model, "판정 (캐시에 없던 것만 새로 계산)", int(r.n_new_forward), float(r.seconds),
                    f"model_status.csv; 전체 프롬프트 {int(r.n_prompts):,}"))
    return out


def build_rows(cfg, version, date, commit):
    res, rows = cfg["_res"], []
    # 임계값 기준이 바뀌면 run_id 도 바꾼다 (exp-v1: 팀 판정기 κ 기준 → T1-κ)
    RID = RUN_ID | ({"team": "T1-κ"} if version.startswith("exp") else {})

    def add(run_id, table, source, model, metric, value, note=""):
        rows.append([version, run_id, date, commit, "dev", table, source, model, metric, value, note])

    # 실험 설정
    th = read_csv_safe(res / "team_thresholds.csv")
    tau, agg = cfg["retrieval"]["tau"], cfg["judge"]["aggregation"]
    for mo, rid in RID.items():
        spec = cfg["judge"]["models"][mo]
        add(rid, "setup", "", mo, "hf_id", spec["hf_id"])
        add(rid, "setup", "", mo, "prompt_version", "team_v1" if mo == "team" else "official")
        if mo == "team" and len(th):
            t = th[th.prompt_version == "team_v1"]
            add(rid, "setup", "", mo, "threshold", " / ".join(f"{SRC[s]} {v:g}" for s, v in zip(t.source, t.per_source_threshold)))
            crit = t.criterion.iloc[0] if "criterion" in t and len(t) else "f1"
            add(rid, "setup", "", mo, "threshold_rule", f"출처별 dev {'κ(사람 라벨)' if crit == 'kappa' else 'F1(위반)'} 최대, 격자 0.01")
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
            add(RID[r.model], "p2", r.source, r.model, k, getattr(r, k), r.note if isinstance(r.note, str) else "")
    # 모델 간 κ
    for r in read_csv_safe(res / "p2/pairwise_kappa.csv").itertuples():
        add(f"{RID[r.model_a]}×{RID[r.model_b]}", "kappa_pair", r.source, f"{r.model_a}|{r.model_b}", "kappa", r.kappa)
    # P3
    s = read_csv_safe(res / "p3/cause_summary.csv")
    n = {(r.source, r.model, r.cause): r.n for r in s.itertuples()}
    for src in sorted(s.source.unique()):
        for mo in RID:
            for c in CAUSE:
                rule_free = c == "retrieval" and not cfg["judge"]["models"][mo]["uses_rules"]
                add(RID[mo], "p3", src, mo, c, "n/a" if rule_free else n.get((src, mo, c), 0),
                    "규칙을 받지 않는 모델에는 적용 안 함" if rule_free else "")
    # P4 (있으면): 처음 보는 규칙 AUC 하락폭·회복폭, 설정을 바꿨을 때 뒤집힌 판정 수
    adr = read_csv_safe(res / "p4/auc_drop_recovery.csv")
    for r in adr[adr.group_by == "source"].itertuples() if len(adr) else []:
        for k in ["auc_base", "auc_rule_excluded", "auc_rule_only", "auc_rule_plus_examples", "auc_drop",
                  "recovery_rule_only", "recovery_rule_plus_examples"]:
            add(RID[r.model], "p4_auc", r.group, r.model, k, getattr(r, k))
    import json
    for f in sorted((res / "rule_proposals").glob("*.json")):
        pr = json.loads(f.read_text(encoding="utf-8"))
        kind = pr.get("name", "").split("_")[0]
        if kind not in ("threshold", "priority", "tau"):
            continue  # 규칙 개선안(proposal_*) 은 실험 version 에서 따로 기록
        mo = pr["change"]["model"]
        key = {"threshold": "threshold", "priority": "aggregation", "tau": "tau"}[kind]
        what = f"{ {'threshold': '임계값', 'priority': '집계', 'tau': 'τ'}[kind]} → {pr['change'][key][1]}"
        for k in ["n_flipped", "n_fixed", "n_broken"]:
            add(RID[mo], "p4_flip", "", mo, k, pr[k], what)
    if version.startswith("exp"):
        exp_rows(res / "exp_v1", res, add)
    # 실행 시간
    for rid, mo, what, n, sec, note in (TIMES_V0 if version == "baseline-v0" else times_from_files(cfg)):
        add(rid, "time", "", mo, "seconds", sec, f"{what}; n_forward={n}; {note}" if n else f"{what}; {note}")
    return pd.DataFrame(rows, columns=COLS)


def exp_rows(d, res, add):
    """exp-v1 실험 결과 (scripts/exp_v1.py) 를 long format 으로."""
    b1 = read_csv_safe(d / "b1_prompt_compare.csv")
    for r in b1.itertuples():
        for k in ["n", "n_pos", "roc_auc", "pr_auc", "threshold", "f1", "kappa", "pred_pos"]:
            add(r.run_id, "b1_prompt", r.source, r.prompt_version, k, getattr(r, k), r.threshold_rule)
    for r in read_csv_safe(d / "b1_by_action.csv").itertuples():
        add(r.run_id, "b1_action", "do_not_answer", r.prompt_version, r.action, r.mean_p, f"n={r.n}, 위반={r.n_harmful}")
    for r in read_csv_safe(d / "b1_harm_recall.csv").itertuples():
        add(r.run_id, "b1_recall", "do_not_answer", r.prompt_version, r.group, f"{r.caught}/{r.n_pos} ({r.recall:.2f})", f"임계값 {r.threshold:g}")
    th = read_csv_safe(res / "team_thresholds.csv")
    for r in th.itertuples() if "criterion" in th else []:
        add("T1-κ", "b2_threshold", r.source, r.prompt_version, f"{r.criterion} 기준 (최종)", f"{r.per_source_threshold:g} (점수 {r.per_source_dev_score:.3f})")
        add("T1-κ", "b2_threshold", r.source, r.prompt_version, f"{r.alt_criterion} 기준 (비교)", f"{r.alt_per_source_threshold:g} (점수 {r.alt_per_source_dev_score:.3f})")
    b3, fl = read_csv_safe(d / "b3_rule_compare.csv"), read_csv_safe(d / "b3_flips.csv")
    for r in b3.itertuples():
        for k in ["roc_auc", "pr_auc", "threshold", "f1", "kappa", "pred_pos"]:
            add(r.run_id, "b3_rule", r.source, r.model, k, getattr(r, k), r.rule_version)
    for r in fl.itertuples():
        for k in ["wrong_R0", "wrong_R1", "n_fixed", "n_broken"]:
            add(r.run_id, "b3_flip", "do_not_answer", r.model, k, getattr(r, k))
    for name, f in [("B1", "b1_times.csv"), ("B3", "b3_times.csv")]:
        for r in read_csv_safe(d / f).itertuples():
            what = f"{name} {r.run_id}" + (f" {r.rule_version}" if hasattr(r, "rule_version") else "")
            add(r.run_id, "time", "", what, "seconds", r.seconds,
                f"{name} rule_only (캐시에 없던 것만 새로 계산); n_forward={r.n_new_forward}; 전체 프롬프트 {r.n_prompts:,}")
    for r in read_csv_safe(d / "b4_harm_errors.csv").itertuples():
        add(RUN_ID.get(r.model, r.model) if r.model != "team" else "T1-κ", "b4_harm", "do_not_answer", r.model, r.types_of_harm,
            f"{r.wrong}/{r.n} ({r.wrong / r.n:.2f})", f"놓친 위반 {r.missed}/{r.n_pos}")


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


def exp_block(e):
    """exp-v1 표: B1 프롬프트, 응답 유형별 평균 확률, 유해 표현 없는 유형 재현율, B2 임계값, B3 규칙 문장, B4 위해 유형별 오류."""
    out = []

    def sec(title, df, index, cols=None, raw=False):
        if len(df):
            out.append(f"**{title}**\n\n" + table(df, index, "metric", cols, raw=raw) + "\n")
    b1 = e[e.table == "b1_prompt"].assign(임계값_기준=lambda x: x.note)
    sec("B1 팀 프롬프트 비교 (rule_only, R0 규칙 문장)", b1, ["출처", "run_id", "임계값_기준"],
        ["n", "n_pos", "roc_auc", "pr_auc", "threshold", "f1", "kappa", "pred_pos"])
    sec("B1 DNA 응답 유형별 평균 위반 확률 (rule_only)", e[e.table == "b1_action"], ["run_id"])
    sec("B1 DNA 위해 유형별 재현율 (잡은 위반/위반, κ 최대 임계값)", e[e.table == "b1_recall"], ["run_id"], raw=True)
    sec("B2 팀 판정기 임계값 기준 (T1, base)", e[e.table == "b2_threshold"], ["출처"], raw=True)
    b3 = e[e.table == "b3_rule"].assign(규칙=lambda x: x.note)
    sec("B3 규칙 문장 R0 vs R1 (DNA, rule_only)", b3, ["run_id", "규칙"], ["roc_auc", "pr_auc", "threshold", "f1", "kappa", "pred_pos"])
    sec("B3 고쳐진 판정 / 새로 틀린 판정 (R0 → R1)", e[e.table == "b3_flip"], ["run_id"], ["wrong_R0", "wrong_R1", "n_fixed", "n_broken"])
    b4 = e[e.table == "b4_harm"].rename(columns={"metric": "위해 유형", "run_id": "metric"})
    sec("B4 DNA 위해 유형별 잘못된 판정 (잘못된/전체, base)", b4, ["위해 유형"], raw=True)
    return "\n".join(out) + ("\n" if out else "")


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
    n_src = e[(e.table == "p2") & (e.metric == "n")].groupby("출처").value.first()
    sample = ", ".join(f"{k} {int(float(v)):,}건" for k, v in n_src.items())
    p4a = e[e.table == "p4_auc"]
    p4 = ""
    if len(p4a):
        p4 = "**P4 처음 보는 규칙** (정답 규칙이 있는 텍스트, 하락폭 = base − 규칙 제외, 회복폭 = 조건 − 규칙 제외)\n\n" + table(
            p4a, ["출처", "run_id"], "metric", ["auc_base", "auc_rule_excluded", "auc_rule_only", "auc_rule_plus_examples",
                                                "auc_drop", "recovery_rule_only", "recovery_rule_plus_examples"]) + "\n\n"
    p4f = e[e.table == "p4_flip"].assign(변경=lambda x: x.note)
    if len(p4f):
        p4 += "**P4 설정 변경 시 뒤집힌 판정** (dev 전체 출처 합계, 맞게 = fixed, 틀리게 = broken)\n\n" + table(
            p4f, ["run_id", "변경"], "metric", ["n_flipped", "n_fixed", "n_broken"]) + "\n\n"
    ex = exp_block(e) if version.startswith("exp") else ""
    tm = e[e.table == "time"]
    tml = ["| run_id | 작업 | 시간 | 비고 |", "|---|---|---|---|"]
    for r in tm.itertuples():
        sec = float(r.value)
        tml.append(f"| {r.run_id} | {r.model} | {sec / 60:.1f}분 ({sec:,.0f}초) | {r.note} |")
    return f"""<!-- results:{version} -->
### {version.replace('baseline-v', 'Baseline v')} (dev, {date})

> ⚠ **모든 수치는 dev 결과다. test 는 아직 보지 않았다.** 검증 상태: `IS만` (In-Sample: 임계값·τ 를 정한 dev 로 평가해 낙관적).
> 표본(dev): {sample}. split·규칙 트리는 임시([ASSUMPTIONS](docs/ASSUMPTIONS.md)).
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

{p4}{ex}**실행 시간** (RTX 3060 8GB)

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
