"""원자료 탐색 보고서 → reports/data_overview.md (+ 표마다 csv)

PDF 1주차: "출처별·규칙별 건수, 라벨 분포, 결측·중복도 봅니다."
texts_all.csv 하나만 읽으므로 combine_texts.py 다음에 실행.

    python scripts/data_overview.py
"""
import pandas as pd

from common import PROCESSED, REPORTS, norm_text

SRC = {"dna": "Do-Not-Answer", "jigsaw": "Jigsaw", "prosocial": "ProsocialDialog"}
APPROX_CHARS_128_TOKENS = 512        # 영어 기준 1토큰 ≈ 4글자로 어림한 값 (정확한 값은 토크나이저로 셀 것)


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v).replace("|", "/").replace("\n", " ")
                                       for v in row) + " |")
    return "\n".join(lines)


def pct(a, b):
    return f"{a / b:.1%}" if b else "-"


def main():
    t = pd.read_csv(PROCESSED / "texts_all.csv", keep_default_na=False)
    t["human_label"] = t["human_label"].astype(int)
    t["_key"] = t["text"].map(norm_text)
    t["_len"] = t["text"].str.len()
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = ["# 원자료 탐색 보고서", "",
           f"대상: `texts_all.csv` {len(t):,}행. 분할과 text_id는 임시 기준입니다.", ""]

    # 1. 출처·분할별 건수와 라벨 분포
    g = t.groupby(["source", "split"]).human_label.agg(n="count", pos="sum").reset_index()
    g1 = pd.DataFrame({"출처": g.source.map(SRC), "분할": g.split, "건수": g.n.map("{:,}".format),
                       "위반": g.pos.map("{:,}".format), "위반 비율": [pct(a, b) for a, b in zip(g.pos, g.n)]})
    out += ["## 1. 출처·분할별 건수와 라벨 분포", "", md_table(g1), ""]
    g.to_csv(REPORTS / "overview_source_split.csv", index=False)

    # 2. 규칙별 건수
    r = t[t.rule_l2 != ""].groupby(["source", "rule_l1", "rule_l2"]).agg(
        n=("human_label", "count"), pos=("human_label", "sum"),
        dev=("split", lambda s: int((s == "dev").sum())), test=("split", lambda s: int((s == "test").sum())),
        rules=("gold_rule", "nunique")).reset_index()
    r2 = pd.DataFrame({"출처": r.source.map(SRC), "상위 분류": r.rule_l1, "규칙(범주)": r.rule_l2,
                       "세부 규칙 수": r.rules, "건수": r.n, "위반": r.pos,
                       "위반 비율": [pct(a, b) for a, b in zip(r.pos, r.n)], "dev": r.dev, "test": r.test})
    out += ["## 2. 규칙(범주)별 건수", "", md_table(r2), "",
            "ProsocialDialog는 정답 규칙이 없어서 이 표에 없습니다.", ""]
    low = r[r.pos < 10]
    if len(low):
        out += [f"위반이 10건 미만인 범주가 {len(low)}개 있습니다: "
                + ", ".join(f"{a} ({b}건)" for a, b in zip(low.rule_l2, low.pos))
                + ". 이 범주들의 지표는 건수와 함께 봐야 합니다.", ""]
    r.to_csv(REPORTS / "overview_rules.csv", index=False)

    # 3. 결측
    cols = ["text", "context", "gold_rule", "rule_text", "responder"]
    m = t.groupby("source").apply(lambda d: pd.Series({c: int((d[c].astype(str).str.strip() == "").sum())
                                                        for c in cols}), include_groups=False).reset_index()
    m.insert(1, "건수", t.groupby("source").size().values)
    m["source"] = m["source"].map(SRC)
    out += ["## 3. 빈 값 개수", "", md_table(m.rename(columns={"source": "출처"})), "",
            "gold_rule·rule_text가 비어 있는 것은 ProsocialDialog(정답 규칙 없음), responder가 비어 있는 것은 "
            "Jigsaw·ProsocialDialog(응답 LLM 없음)라서 정상입니다. text가 비어 있으면 확인이 필요합니다.", ""]

    # 4. 중복
    rows = []
    for s, d in t.groupby("source"):
        cnt = d.groupby("_key").agg(n=("text_id", "count"), labels=("human_label", "nunique"),
                                    splits=("split", "nunique"))
        dupk = cnt[cnt.n > 1]
        rows.append({"출처": SRC[s], "같은 텍스트가 2번 이상": f"{len(dupk):,}종 ({int(dupk.n.sum()):,}행)",
                     "그중 라벨이 서로 다름": int((dupk.labels > 1).sum()),
                     "그중 여러 분할에 걸침": int((dupk.splits > 1).sum())})
    out += ["## 4. 중복", "", md_table(pd.DataFrame(rows)), "",
            "Do-Not-Answer는 \"I'm sorry, but I can't assist with that.\" 같은 거절 문구가 여러 질문에 "
            "반복되어 중복이 많습니다. 판정 대상이 응답 + 질문 쌍이므로 지울 필요는 없지만, "
            "**라벨이 서로 다른 중복**은 라벨 검토 후보이고 **여러 분할에 걸친 중복**은 dev에서 본 문장이 "
            "test에 다시 나온다는 뜻이라 분할을 확정할 때 다시 확인해야 합니다.", ""]
    top = (t[t.source == "dna"].groupby("_key").size().sort_values(ascending=False).head(5))
    if len(top):
        out += ["Do-Not-Answer에서 가장 많이 반복된 응답:", "",
                md_table(pd.DataFrame({"반복 수": top.values, "응답 (앞 80자)": [k[:80] for k in top.index]})), ""]

    # 5. 길이
    L = t.groupby("source")["_len"].describe(percentiles=[.5, .9]).reset_index()
    over = t.groupby("source")["_len"].apply(lambda s: (s > APPROX_CHARS_128_TOKENS).mean())
    L5 = pd.DataFrame({"출처": L.source.map(SRC), "중앙값(글자)": L["50%"].round().astype(int),
                       "상위 10% 지점": L["90%"].round().astype(int), "최대": L["max"].astype(int),
                       f"{APPROX_CHARS_128_TOKENS}자 초과 비율": [f"{over[s]:.1%}" for s in L.source]})
    out += ["## 5. 텍스트 길이", "", md_table(L5), "",
            f"규칙 찾기에 쓰는 all-roberta-large-v1은 128토큰까지만 읽습니다. 영어 1토큰을 약 4글자로 어림하면 "
            f"{APPROX_CHARS_128_TOKENS}자를 넘는 텍스트는 뒷부분이 유사도에 반영되지 않습니다(어림값).", ""]

    # 6. 원본 라벨 분포
    out += ["## 6. 원본 라벨 값", ""]
    for s, title in [("dna", "Do-Not-Answer: 응답 행동 유형(action) × 사람 라벨"),
                     ("prosocial", "ProsocialDialog: safety_label × 변환한 라벨")]:
        d = t[t.source == s]
        if len(d):
            c = pd.crosstab(d.label_raw, d.human_label).rename(columns={0: "정상(0)", 1: "위반(1)"}).reset_index()
            out += [f"**{title}**", "", md_table(c.rename(columns={"label_raw": "원본 값"})), ""]

    (REPORTS / "data_overview.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[:40]))
    print(f"\n... 전체 보고서: {REPORTS / 'data_overview.md'}")


if __name__ == "__main__":
    main()
