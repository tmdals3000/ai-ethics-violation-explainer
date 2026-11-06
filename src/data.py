"""1단계: 원본 3종 → 공통 스키마, 규칙 트리, split, dev 표본, EDA."""
import re

import numpy as np
import pandas as pd

from src import adapters

SCHEMA = ["text_id", "source", "split", "text", "context", "human_label",
          "gold_rule_id", "gold_path", "response_llm"]
DNA_LLMS = ["GPT4", "ChatGPT", "Claude", "ChatGLM2", "llama2-7b-chat", "vicuna-7b"]


# ---------- 원본 읽기 ----------
def load_dna(raw_dir):
    """Do-Not-Answer: 질문 1행에 응답 6개 → 응답 1개 = 1행으로 펼친다."""
    d = pd.read_csv(raw_dir / "dna/datasets/data_en.csv")
    rows = []
    for _, r in d.iterrows():
        for llm in DNA_LLMS:
            rows.append({
                "text_id": f"dna-{r['id']}-{llm}", "source": "do_not_answer",
                "text": r[f"{llm}_response"], "context": r["question"],
                "human_label": int(r[f"{llm}_harmful"]), "response_llm": llm,
                "group_id": f"dna-q{r['id']}",
                "risk_area": r["risk_area"], "types_of_harm": r["types_of_harm"],
                "specific_harms": r["specific_harms"],
            })
    return pd.DataFrame(rows)


def load_jigsaw(raw_dir):
    """Jigsaw ACRC train.csv. 행마다 딸린 예시 4개는 따로 떼어 예시 후보로 만든다."""
    p = raw_dir / "jigsaw/train.csv"
    if not p.exists():
        return None, None
    d = pd.read_csv(p)
    texts = pd.DataFrame({
        "text_id": "jig-" + d["row_id"].astype(str), "source": "jigsaw",
        "text": d["body"], "context": "r/" + d["subreddit"].astype(str),
        "human_label": d["rule_violation"].astype(int), "response_llm": "",
        "group_id": "jig-" + d["row_id"].astype(str), "rule": d["rule"],
    })
    ex = []
    for _, r in d.iterrows():
        for col, lab in [("positive_example_1", 1), ("positive_example_2", 1),
                         ("negative_example_1", 0), ("negative_example_2", 0)]:
            ex.append({"origin_text_id": f"jig-{r['row_id']}", "rule": r["rule"],
                       "label": lab, "text": r[col]})
    return texts, pd.DataFrame(ex)


def _ascii_ratio(s):
    letters = re.findall(r"[^\W\d_]", s)
    return 1.0 if not letters else sum(c.isascii() for c in letters) / len(letters)


def load_prosocial(cfg):
    """ProsocialDialog: 사용자 발화(context 필드)가 판정 대상. 이전 턴은 맥락으로 붙인다."""
    from datasets import load_dataset
    ds = load_dataset("allenai/prosocial-dialog")
    lab = cfg["data"]["prosocial_label_map"]
    out = []
    for hf_split, split in [("train", "train"), ("validation", "dev"), ("test", "test")]:
        d = ds[hf_split].to_pandas().sort_values(["dialogue_id", "response_id"])
        prev = {}
        for r in d.itertuples():
            hist = prev.get(r.dialogue_id, [])
            out.append({
                "text_id": f"pd-{hf_split}-{r.dialogue_id}-{r.response_id}", "source": "prosocial",
                "split": split, "text": r.context, "context": "\n".join(hist[-4:]),
                "human_label": lab[r.safety_label], "response_llm": "",
                "group_id": f"pd-{hf_split}-{r.dialogue_id}", "safety_label": r.safety_label,
            })
            prev[r.dialogue_id] = hist + [r.context, r.response]
    df = pd.DataFrame(out)
    keep = df["text"].map(_ascii_ratio) >= cfg["data"]["english_min_ascii_ratio"]
    print(f"[prosocial] 영어 필터로 {int((~keep).sum())}건 제외")
    return df[keep]


# ---------- 규칙 트리 ----------
def build_tree(dna, jig):
    """대(level1)·중(level2)·소(level3) 노드표. rule_text 가 임베딩·판정에 쓰는 규칙 문장."""
    nodes = []
    tri = dna[["risk_area", "types_of_harm", "specific_harms"]].drop_duplicates()
    ra_id = {v: f"R{i+1}" for i, v in enumerate(sorted(tri["risk_area"].unique()))}
    th_id = {v: f"H{i+1:02d}" for i, v in enumerate(sorted(tri["types_of_harm"].unique()))}
    sh_id = {}
    for ra, i in ra_id.items():
        nodes.append({"node_id": i, "level": 1, "parent_id": "", "name": ra, "rule_text": ra, "source": "do_not_answer"})
    for (ra, th), _ in tri.groupby(["risk_area", "types_of_harm"]):
        nodes.append({"node_id": th_id[th], "level": 2, "parent_id": ra_id[ra], "name": th, "rule_text": th, "source": "do_not_answer"})
    for k, (th, sh) in enumerate(sorted(set(zip(tri["types_of_harm"], tri["specific_harms"])))):
        sid = f"S{k+1:02d}"
        sh_id[(th, sh)] = sid
        name, _, desc = sh.partition(":")
        rule = (f'"{name.strip()}" ({th}): The chatbot shall not provide content that complies with '
                f"or assists requests in this category. Scope: {desc.strip() or name.strip()}")
        nodes.append({"node_id": sid, "level": 3, "parent_id": th_id[th], "name": name.strip(), "rule_text": rule, "source": "do_not_answer"})
    jig_id = {}
    if jig is not None:
        nodes.append({"node_id": "J", "level": 1, "parent_id": "", "name": "Community rules (Reddit)",
                      "rule_text": "Community rules (Reddit)", "source": "jigsaw"})
        for i, rule in enumerate(sorted(jig["rule"].unique())):
            jig_id[rule] = f"J{i+1}"
            nodes.append({"node_id": f"J{i+1}", "level": 2, "parent_id": "J", "name": rule.split(":")[0],
                          "rule_text": rule, "source": "jigsaw"})
    # 정답 규칙과 경로를 텍스트에 붙인다
    dna["gold_rule_id"] = [sh_id[(t, s)] for t, s in zip(dna["types_of_harm"], dna["specific_harms"])]
    dna["gold_path"] = [f"{ra_id[a]}>{th_id[t]}>{g}" for a, t, g in zip(dna["risk_area"], dna["types_of_harm"], dna["gold_rule_id"])]
    if jig is not None:
        jig["gold_rule_id"] = jig["rule"].map(jig_id)
        jig["gold_path"] = "J>" + jig["gold_rule_id"]
    return pd.DataFrame(nodes), jig_id


# ---------- split ----------
def temporary_split(df, ratio, seed):
    """group_id(DNA는 같은 질문) 단위로 고정 시드 split. 같은 질문의 응답이 dev/test에 섞이지 않게."""
    groups = np.array(sorted(df["group_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n = len(groups)
    a, b = int(n * ratio["train"]), int(n * (ratio["train"] + ratio["dev"]))
    m = {g: "train" for g in groups[:a]} | {g: "dev" for g in groups[a:b]} | {g: "test" for g in groups[b:]}
    return df["group_id"].map(m)


def apply_splits(cfg, df):
    ta, _ = adapters.read(cfg, "splits") if adapters.has_ta(cfg, "splits") else (None, True)
    if ta is not None:
        m = dict(zip(ta["text_id"], ta["split"]))
        hit = df["text_id"].isin(m)
        df.loc[hit, "split"] = df.loc[hit, "text_id"].map(m)
        df.loc[hit, "split_source"] = "ta"
        print(f"[split] 조교 splits.csv 적용 {int(hit.sum())}건")
    return df


def dev_sample(df, n_by_source, stratify, seed, split="dev"):
    """출처별 최대 n건 (n='all' 이면 전체). 라벨 비율 유지."""
    out = []
    for src, g in df[df["split"] == split].groupby("source"):
        n = n_by_source.get(src, 50) if isinstance(n_by_source, dict) else n_by_source
        if n == "all" or len(g) <= int(n):
            out.append(g)
            continue
        n = int(n)
        if stratify and g["human_label"].nunique() > 1:
            frac = n / len(g)
            s = pd.concat([x.sample(max(1, round(len(x) * frac)), random_state=seed)
                           for _, x in g.groupby("human_label")])
            out.append(s.sample(min(n, len(s)), random_state=seed))
        else:
            out.append(g.sample(n, random_state=seed))
    return pd.concat(out)


def make_examples(cfg, df, jig_ex, nodes):
    """rule_plus_examples 조건용 예시 후보.
    Jigsaw: 각 행에 딸린 예시. DNA: train split 응답 중 같은 소분류(없으면 같은 중분류) 것."""
    ex = []
    if jig_ex is not None:
        rid = dict(zip(nodes.loc[nodes.source == "jigsaw", "rule_text"], nodes.loc[nodes.source == "jigsaw", "node_id"]))
        j = jig_ex.assign(rule_id=jig_ex["rule"].map(rid), source="jigsaw", origin="row_attached")
        ex.append(j[["source", "rule_id", "label", "text", "origin_text_id", "origin"]])
    tr = df[(df.source == "do_not_answer") & (df.split == "train")]
    ex.append(pd.DataFrame({"source": "do_not_answer", "rule_id": tr["gold_rule_id"], "label": tr["human_label"],
                            "text": tr["text"], "origin_text_id": tr["text_id"], "origin": "train_split"}))
    return pd.concat(ex, ignore_index=True)


# ---------- EDA ----------
def eda(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    g = df.groupby(["source", "split"])["human_label"]
    pd.DataFrame({"n": g.size(), "n_violation": g.sum(), "violation_rate": g.mean().round(4)}).reset_index() \
        .to_csv(out_dir / "counts_by_source_split.csv", index=False)
    r = df[df.gold_rule_id.astype(str) != ""].groupby(["source", "gold_rule_id"])["human_label"]
    pd.DataFrame({"n": r.size(), "n_violation": r.sum(), "violation_rate": r.mean().round(4)}).reset_index() \
        .to_csv(out_dir / "counts_by_rule.csv", index=False)
    if (df.source == "do_not_answer").any():
        d = df[df.source == "do_not_answer"].groupby("response_llm")["human_label"]
        pd.DataFrame({"n": d.size(), "n_violation": d.sum(), "violation_rate": d.mean().round(4)}).reset_index() \
            .to_csv(out_dir / "counts_by_response_llm.csv", index=False)
    miss = df[SCHEMA].replace("", np.nan).isna().groupby(df["source"].rename("src")).sum()
    miss.reset_index().to_csv(out_dir / "missing.csv", index=False)  # 빈 값 개수 (PD의 gold_rule 은 원래 없음)
    dup = df.groupby("source").agg(n=("text_id", "size"),
                                   dup_text_id=("text_id", lambda s: int(s.duplicated().sum())),
                                   dup_text=("text", lambda s: int(s.duplicated().sum())))
    dup.reset_index().to_csv(out_dir / "duplicates.csv", index=False)
    if "safety_label" in df:
        df[df.source == "prosocial"].groupby(["split", "safety_label", "human_label"]).size() \
            .rename("n").reset_index().to_csv(out_dir / "prosocial_label_mapping.csv", index=False)


# ---------- 전체 ----------
def prepare(cfg):
    raw = cfg["_raw"]
    status = {}
    dna = load_dna(raw)
    jig, jig_ex = load_jigsaw(raw)
    status["jigsaw"] = "ok" if jig is not None else f"missing: {raw / 'jigsaw/train.csv'}"
    nodes, _ = build_tree(dna, jig)
    parts = [dna]
    if jig is not None:
        parts.append(jig)
    for p in parts:
        p["split"] = temporary_split(p, cfg["data"]["split_ratio"], cfg["seed"])
        p["split_source"] = "temporary"
    if "prosocial" in cfg["data"]["sources"]:
        pdl = load_prosocial(cfg)
        pdl["split_source"] = "original_hf"
        pdl["gold_rule_id"] = ""
        pdl["gold_path"] = ""
        parts.append(pdl)
    df = pd.concat(parts, ignore_index=True)
    for c in SCHEMA:
        df[c] = df[c].fillna("") if c != "human_label" else df[c]
    df = apply_splits(cfg, df)
    out = cfg["_data"] / "derived"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "texts.csv", index=False)
    nodes.to_csv(out / "nodes.csv", index=False)
    make_examples(cfg, df, jig_ex, nodes).to_csv(out / "examples.csv", index=False)
    if not adapters.has_ta(cfg, "splits"):
        adapters.write_ours(cfg, "splits", df[["text_id", "split", "split_source"]])
    s = dev_sample(df, cfg["data"]["dev_sample_per_source"], cfg["data"]["stratify_sample"], cfg["seed"])
    s[["text_id", "source"]].to_csv(out / "dev_sample.csv", index=False)
    eda(df, cfg["_res"] / "eda")
    pd.Series(status).rename("status").to_csv(cfg["_res"] / "eda" / "source_status.csv")
    print(df.groupby(["source", "split"]).size().to_string())
    print("dev 표본:", s.groupby("source").size().to_dict(), "위반:", s.groupby("source")["human_label"].sum().to_dict())
    return df, nodes


def load(cfg):
    d = cfg["_data"] / "derived"
    texts = pd.read_csv(d / "texts.csv", keep_default_na=False, na_values=[], low_memory=False)
    texts["human_label"] = texts["human_label"].astype(int)
    nodes = pd.read_csv(d / "nodes.csv", keep_default_na=False)
    nodes["level"] = nodes["level"].astype(int)
    ex = pd.read_csv(d / "examples.csv", keep_default_na=False)
    sample = pd.read_csv(d / "dev_sample.csv")
    return texts, nodes, ex, sample
