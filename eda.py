"""
데이터셋 살피기 (EDA) — 데이터 3종의 구조·분포·품질 확인

실행: python eda.py
입력: texts.csv (preprocess.py 결과), data_en.csv (DNA 응답 유형 action 을 가져오려고)
출력: results/eda/*.csv  (원문 없이 숫자만 → GitHub 에 올려도 됨)
"""
import os
import numpy as np
import pandas as pd

os.makedirs("results/eda", exist_ok=True)
SRC = {"do_not_answer": "Do-Not-Answer", "jigsaw": "Jigsaw", "prosocial": "ProsocialDialog"}
t = pd.read_csv("texts.csv", keep_default_na=False)
t["n_words"] = t["text"].astype(str).str.split().str.len()
t["n_words_q"] = t["question"].astype(str).str.split().str.len()


def save(df, name):
    df.to_csv(f"results/eda/{name}.csv", index=False)
    return df


# 1) 출처 × 분할: 건수, 위반 수·비율, 판정 표본 수
s1 = t.groupby(["source", "split"]).agg(n=("text_id", "size"), n_violation=("human_harmful", "sum"),
                                         violation_rate=("human_harmful", "mean"), n_sample=("sample", "sum")).reset_index()
print("① 출처 × 분할\n", save(s1.round(3), "source_split").to_string(index=False))

# 2) 품질: 결측(빈 글), 중복(같은 글이 여러 번), 그룹 누수(같은 그룹이 dev·test 에 갈라짐)
rows = []
for s, g in t.groupby("source"):
    dup = g.duplicated("text", keep=False)
    leak = g.groupby("group_id")["split"].nunique().gt(1).sum()
    rows.append({"source": s, "n": len(g), "empty_text": int((g["text"].str.strip() == "").sum()),
                 "duplicate_text_rows": int(dup.sum()), "duplicate_texts_label_conflict":
                     int(g[dup].groupby("text")["human_harmful"].nunique().gt(1).sum()),
                 "groups_split_across_dev_test": int(leak)})
print("\n② 품질 (결측·중복·누수)\n", save(pd.DataFrame(rows), "quality").to_string(index=False))

# 3) 글 길이 (단어 수)
s3 = t.groupby("source")["n_words"].describe(percentiles=[.5, .9, .99]).reset_index().round(1)
print("\n③ 판정할 글 길이 (단어 수)\n", save(s3, "text_length").to_string(index=False))

# 4) DNA: 규칙(대·중분류)별 건수와 위반 비율 (dev)
dna = t[(t.source == "do_not_answer")]
s4 = dna.groupby(["rule_l1", "rule_l2"]).agg(n=("text_id", "size"), n_violation=("human_harmful", "sum"),
                                              violation_rate=("human_harmful", "mean")).reset_index()
print("\n④ DNA 규칙별 (dev+test)\n", save(s4.round(3), "dna_by_rule").to_string(index=False))
s4b = dna.groupby("rule_l3").agg(n=("text_id", "size"), n_violation=("human_harmful", "sum")).reset_index()
print(f"   소분류 {len(s4b)}개 · 건수 최소 {s4b.n.min()} / 최대 {s4b.n.max()} · 위반 0건인 소분류 {int((s4b.n_violation == 0).sum())}개")
save(s4b, "dna_by_rule_l3")

# 5) DNA: 응답 LLM × 응답 유형(action)
#    action (Do-Not-Answer 논문 기준): 0 거절 · 1 질문의 전제를 반박 · 2 양쪽 관점 제시
#    3 위험을 알리고 조심스럽게 답(면책 문구) · 4 모른다/정확히 답할 수 없음 · 5 지시를 그대로 따름
ACTION = {0: "0 거절", 1: "1 전제 반박", 2: "2 양쪽 관점", 3: "3 면책 후 답변", 4: "4 모름·불확실", 5: "5 그대로 따름"}
raw = pd.read_csv("data_en.csv")
llms = ["GPT4", "ChatGPT", "Claude", "ChatGLM2", "llama2-7b-chat", "vicuna-7b"]
amap = {f"dna-{i}-{m}": a for m in llms for i, a in zip(raw["id"], raw[f"{m}_action"])}
dna = dna.assign(action=dna.text_id.map(amap).map(ACTION))
s5 = dna.groupby("response_llm").agg(n=("text_id", "size"), violation_rate=("human_harmful", "mean"),
                                     words_median=("n_words", "median")).reset_index()
print("\n⑤ DNA 응답 LLM별\n", save(s5.round(3), "dna_by_llm").to_string(index=False))
s5b = dna.groupby("action").agg(n=("text_id", "size"), n_violation=("human_harmful", "sum"),
                                violation_rate=("human_harmful", "mean")).reset_index()
print("\n⑥ DNA 응답 유형별 (사람 라벨 기준 위반 비율)\n", save(s5b.round(3), "dna_by_action").to_string(index=False))
save(pd.crosstab(dna.response_llm, dna.action).reset_index(), "dna_llm_x_action")

# 6) Jigsaw: 규칙별·서브레딧 상위
jig = t[t.source == "jigsaw"]
if len(jig):
    s6 = jig.groupby(["split", "rule_l2"]).agg(n=("text_id", "size"), violation_rate=("human_harmful", "mean")).reset_index()
    print("\n⑦ Jigsaw 규칙별\n", save(s6.round(3), "jigsaw_by_rule").to_string(index=False))
    s6b = jig.groupby("question").agg(n=("text_id", "size"), violation_rate=("human_harmful", "mean")) \
        .sort_values("n", ascending=False).head(15).reset_index().rename(columns={"question": "subreddit"})
    save(s6b.round(3), "jigsaw_top_subreddits")
    print(f"   서브레딧 {jig.question.nunique()}개 (상위 15개는 results/eda/jigsaw_top_subreddits.csv)")

# 7) ProsocialDialog: 표본의 맥락 유무, 길이
pdl = t[t.source == "prosocial"]
if len(pdl):
    print(f"\n⑧ ProsocialDialog 표본 {len(pdl)}개: 이전 대화 맥락이 있는 글 {int((pdl.question != '').sum())}개, "
          f"위반 {int(pdl.human_harmful.sum())}개")
print("\n완료! results/eda 폴더를 확인하세요.")
