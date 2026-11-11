"""
1단계: 데이터 3종 전처리 → texts.csv (텍스트 1개 = 1행, 공통 형식)

실행: python preprocess.py
입력 (demo 폴더):
  data_en.csv            Do-Not-Answer 원본
  jigsaw/train.csv       Jigsaw ACRC (Kaggle 에서 받은 것)
  splits.csv             (있으면) 지난번에 만든 DNA dev/test 분할을 그대로 씀
  ProsocialDialog 는 Hugging Face 에서 자동으로 받음 (pip install datasets 필요)
출력:
  texts.csv              모든 텍스트 (원문 포함 → GitHub 에 올리지 않음)
  jigsaw_examples.csv    Jigsaw 각 행에 딸린 위반·비위반 예시 (규칙+예시 조건용 후보)

공통 열:
  text_id, source, split, sample(판정 대상 표본이면 1), mode(response=챗봇 응답 판정 / prompt=사용자 글 판정),
  question(질문·맥락), text(판정할 글), human_harmful(1=위반), response_llm,
  gold_l1, gold_l2, gold_l3 (정답 규칙 노드 id, 없으면 빈칸), group_id
"""
import os
import re
import numpy as np
import pandas as pd

SEED = 42
SAMPLE_PER_SOURCE = {"jigsaw": 50, "prosocial": 50}   # Do-Not-Answer 는 dev 전체
# ProsocialDialog 안전 라벨 → 위반(1)/정상(0). 조교 데이터 설명서가 오면 바꿔야 함
PD_LABEL_MAP = {"__casual__": 0, "__possibly_needs_caution__": 0,
                "__probably_needs_caution__": 1, "__needs_caution__": 1, "__needs_intervention__": 1}
rng = np.random.default_rng(SEED)


def stratified_sample(g, n):
    """라벨 비율을 유지하면서 n개 뽑기"""
    if len(g) <= n:
        return g.index
    parts = [x.sample(max(1, round(len(x) * n / len(g))), random_state=SEED) for _, x in g.groupby("human_harmful")]
    return pd.concat(parts).sample(min(n, sum(len(p) for p in parts)), random_state=SEED).index


# ---------------------------------------------------------------
# 1) Do-Not-Answer: 응답 6개를 한 행씩 펼침 (지난번과 같은 순서·같은 ID)
# ---------------------------------------------------------------
llms = ["GPT4", "ChatGPT", "Claude", "ChatGLM2", "llama2-7b-chat", "vicuna-7b"]
raw = pd.read_csv("data_en.csv")
parts = []
for llm in llms:
    p = pd.DataFrame({
        "text_id": "dna-" + raw["id"].astype(str) + "-" + llm,
        "source": "do_not_answer", "mode": "response",
        "question": raw["question"], "text": raw[f"{llm}_response"].fillna(""),
        "human_harmful": raw[f"{llm}_harmful"].astype(int), "response_llm": llm,
        "rule_l1": raw["risk_area"], "rule_l2": raw["types_of_harm"],
        "specific_harms": raw["specific_harms"], "group_id": "dna-q" + raw["id"].astype(str),
    })
    parts.append(p)
dna = pd.concat(parts, ignore_index=True)
dna["rule_l3"] = dna["specific_harms"].str.split(":").str[0].str.strip()
dna["rule_l3_desc"] = dna["specific_harms"].str.split(":", n=1).str[1].str.strip().fillna(dna["rule_l3"])
dna["gold_l1"] = "L1:" + dna["rule_l1"]
dna["gold_l2"] = "L2:" + dna["rule_l2"]
dna["gold_l3"] = "L3:" + dna["rule_l3"]

# 분할: 지난번 splits.csv 가 있으면 그대로 (팀과 같은 dev), 없으면 질문 단위 50:50
if os.path.exists("splits.csv"):
    old = pd.read_csv("splits.csv")
    dna["split"] = dna["text_id"].map(dict(zip(old["text_id"], old["split"])))
    print("Do-Not-Answer 분할: 기존 splits.csv 사용")
if "split" not in dna or dna["split"].isna().any():
    qids = dna["group_id"].unique()
    dev_q = set(rng.choice(qids, size=len(qids) // 2, replace=False))
    dna["split"] = np.where(dna["group_id"].isin(dev_q), "dev", "test")
    print("Do-Not-Answer 분할: 무작위 50:50")
dna["sample"] = (dna["split"] == "dev").astype(int)

# ---------------------------------------------------------------
# 2) Jigsaw: 댓글 1개 = 1행. 정답 규칙 = 그 행의 rule. 예시 4개는 따로 뗌
#    같은 댓글 본문이 dev 와 test 에 갈라지지 않게 본문 단위로 나눔
# ---------------------------------------------------------------
jig = None
if os.path.exists("jigsaw/train.csv"):
    j = pd.read_csv("jigsaw/train.csv")
    rule_names = {r: r.split(":")[0].split(".")[0].strip() for r in j["rule"].unique()}
    jig = pd.DataFrame({
        "text_id": "jig-" + j["row_id"].astype(str), "source": "jigsaw", "mode": "prompt",
        "question": "r/" + j["subreddit"].astype(str), "text": j["body"].astype(str),
        "human_harmful": j["rule_violation"].astype(int), "response_llm": "",
        "rule_l1": "Reddit community rules", "rule_l2": j["rule"].map(rule_names),
        "rule_l3": "", "rule_l3_desc": j["rule"],
        "gold_l1": "J1:Reddit community rules", "gold_l2": "J2:" + j["rule"].map(rule_names), "gold_l3": "",
    })
    key = jig["text"].str.strip().str.lower()
    keys = key.unique()
    dev_k = set(rng.choice(keys, size=len(keys) // 2, replace=False))
    jig["group_id"] = "jig-" + pd.factorize(key)[0].astype(str)
    jig["split"] = np.where(key.isin(dev_k), "dev", "test")
    dev = jig[jig["split"] == "dev"]
    jig["sample"] = 0
    jig.loc[stratified_sample(dev, SAMPLE_PER_SOURCE["jigsaw"]), "sample"] = 1
    ex = []
    for _, r in j.iterrows():
        for col, lab in [("positive_example_1", 1), ("positive_example_2", 1),
                         ("negative_example_1", 0), ("negative_example_2", 0)]:
            ex.append({"origin_text_id": f"jig-{r['row_id']}", "rule": r["rule"], "label": lab, "text": r[col]})
    pd.DataFrame(ex).to_csv("jigsaw_examples.csv", index=False)
    print(f"Jigsaw: {len(jig)}행, 규칙 {j['rule'].nunique()}개, 예시 후보 {len(ex)}개")
else:
    print("※ jigsaw/train.csv 가 없어서 Jigsaw 는 건너뜀")

# ---------------------------------------------------------------
# 3) ProsocialDialog: 사용자 발화 1개 = 1행 (판정 대상), 이전 대화는 맥락으로
#    영어만 남기고, 안전 라벨을 위반/정상으로 바꿈. 정답 규칙은 없음
#    용량을 줄이려고 Hugging Face 의 validation(=dev) 만 받아서 표본 50개만 씀
# ---------------------------------------------------------------
pdl = None
try:
    from datasets import load_dataset
    ds = load_dataset("allenai/prosocial-dialog", split="validation").to_pandas()
    ds = ds.sort_values(["dialogue_id", "response_id"])
    hist, rows = {}, []
    for r in ds.itertuples():
        h = hist.get(r.dialogue_id, [])
        rows.append({"text_id": f"pd-validation-{r.dialogue_id}-{r.response_id}", "text": r.context,
                     "question": "\n".join(h[-4:]), "safety_label": r.safety_label,
                     "group_id": f"pd-{r.dialogue_id}"})
        hist[r.dialogue_id] = h + [r.context, r.response]
    pdl = pd.DataFrame(rows)

    def ascii_ratio(s):
        letters = re.findall(r"[^\W\d_]", str(s))
        return 1.0 if not letters else sum(c.isascii() for c in letters) / len(letters)

    before = len(pdl)
    pdl = pdl[pdl["text"].map(ascii_ratio) >= 0.9].copy()
    pdl["human_harmful"] = pdl["safety_label"].map(PD_LABEL_MAP).astype(int)
    pdl = pdl.assign(source="prosocial", mode="prompt", split="dev", response_llm="",
                     rule_l1="", rule_l2="", rule_l3="", rule_l3_desc="", gold_l1="", gold_l2="", gold_l3="")
    pdl = pdl.loc[stratified_sample(pdl, SAMPLE_PER_SOURCE["prosocial"])].assign(sample=1)
    print(f"ProsocialDialog: validation {before}행 중 영어 {int((pdl.shape[0]))}개 표본 사용 "
          f"(라벨: {pdl['safety_label'].value_counts().to_dict()})")
except Exception as e:
    print("※ ProsocialDialog 를 받지 못해 건너뜀:", str(e)[:150])

# ---------------------------------------------------------------
# 4) 합쳐서 저장
# ---------------------------------------------------------------
cols = ["text_id", "source", "split", "sample", "mode", "question", "text", "human_harmful", "response_llm",
        "rule_l1", "rule_l2", "rule_l3", "rule_l3_desc", "gold_l1", "gold_l2", "gold_l3", "group_id"]
texts = pd.concat([x[cols] for x in [dna, jig, pdl] if x is not None], ignore_index=True)
texts.to_csv("texts.csv", index=False)
texts[["text_id", "source", "split"]].to_csv("splits.csv", index=False)

print("\n출처별 건수 (분할·판정 표본):")
print(texts.groupby(["source", "split"]).agg(n=("text_id", "size"), 표본=("sample", "sum"),
                                             위반=("human_harmful", "sum")).to_string())
print("\ntext_id 중복 없음:", texts["text_id"].is_unique)
print("저장 완료: texts.csv")
