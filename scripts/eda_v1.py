"""EDA 보강 (Baseline v1). 기존 results/eda/*.csv 는 그대로 두고 아래 파일만 추가한다. 유해 원문은 쓰지 않는다.

  python scripts/eda_v1.py

  results/eda/dna_action.csv         DNA 응답 유형별 건수·위반 비율 (전체 / dev)
  results/eda/text_length.csv        출처별 길이 분포와 입력 길이 제한에 걸리는 건수
  results/eda/rule_node_length.csv   규칙 노드 문장 길이 (출처 × 단계)
  results/eda/dna_harm_counts.csv    DNA 위험 영역 5개 · 위해 유형 12개 단위 위반 건수
  results/eda/figures/*.png          그래프 (제목에 n)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import data  # noqa: E402
from src.common import load_config  # noqa: E402
from src.judges import MAX_TEXT_CHARS  # noqa: E402
from src.retrieval import embed_input  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"  # 한글 (Windows 기본 글꼴)
plt.rcParams["axes.unicode_minus"] = False
ACTION = {0: "0 거절", 1: "1 반박", 2: "2 양쪽 관점", 3: "3 주의 문구", 4: "4 모름·한계", 5: "5 지시 따름"}
LLMS = ["GPT4", "ChatGPT", "Claude", "ChatGLM2", "llama2-7b-chat", "vicuna-7b"]
SRC = {"do_not_answer": "DNA", "jigsaw": "Jigsaw", "prosocial": "PD"}


def rate_table(df, by):
    g = df.groupby(by)["human_label"]
    return pd.DataFrame({"n": g.size(), "n_violation": g.sum(), "violation_rate": g.mean().round(4)}).reset_index()


def both_splits(df, by):
    """전체와 dev 를 같은 표에 (scope 열로 구분)."""
    return pd.concat([rate_table(df, by).assign(scope="all"), rate_table(df[df.split == "dev"], by).assign(scope="dev")])


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    cfg = load_config()
    texts, nodes, _, _ = data.load(cfg)
    out, fig_dir = cfg["_res"] / "eda", cfg["_res"] / "eda/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dna = texts[texts.source == "do_not_answer"].copy()

    # 1) DNA 응답 유형
    raw = pd.read_csv(cfg["_raw"] / "dna/datasets/data_en.csv")
    act = {f"dna-{r.id}-{l}": int(r[f"{l}_action"]) for _, r in raw.iterrows() for l in LLMS}
    dna["action"] = dna.text_id.map(act).map(ACTION)
    a = both_splits(dna, "action")
    a.to_csv(out / "dna_action.csv", index=False)
    d = a[a.scope == "all"].set_index("action")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(d.index, d.n, color="#9aa5b1", label="전체")
    ax.bar(d.index, d.n_violation, color="#d64545", label="위반")
    for i, (n, v, r) in enumerate(zip(d.n, d.n_violation, d.violation_rate)):
        ax.text(i, n, f"{v}/{n}\n({r:.0%})", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"DNA 응답 유형별 건수와 위반 (n={int(d.n.sum()):,}, 위반 {int(d.n_violation.sum())})")
    ax.set_ylim(0, d.n.max() * 1.25)
    ax.legend()
    save(fig, fig_dir / "dna_action.png")

    # 2) 텍스트 길이와 잘리는 건수
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["retrieval"]["encoder"])
    max_seq = 256  # all-roberta-large-v1 sentence_bert_config.json 의 max_seq_length
    rows, lens = [], {}
    for src, g in texts.groupby("source"):
        for scope, gg in [("all", g), ("dev", g[g.split == "dev"])]:
            n_chars = gg.text.str.len()
            n_tok = pd.Series([len(x) for x in tok([embed_input(cfg, r) for _, r in gg.iterrows()],
                                                     add_special_tokens=True).input_ids])
            rows.append({"source": src, "scope": scope, "n": len(gg),
                         "chars_median": int(n_chars.median()), "chars_p95": int(n_chars.quantile(.95)), "chars_max": int(n_chars.max()),
                         f"judge_text_over_{MAX_TEXT_CHARS}chars": int((n_chars > MAX_TEXT_CHARS).sum()),
                         f"judge_context_over_{MAX_TEXT_CHARS}chars": int((gg.context.str.len() > MAX_TEXT_CHARS).sum()),
                         "embed_tokens_median": int(n_tok.median()), f"embed_over_{max_seq}tokens": int((n_tok > max_seq).sum())})
            if scope == "dev":
                lens[src] = n_chars
    pd.DataFrame(rows).to_csv(out / "text_length.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for src, s in lens.items():
        ax.hist(s.clip(upper=4000), bins=60, histtype="step", linewidth=1.5, label=f"{SRC[src]} (n={len(s):,})")
    ax.axvline(MAX_TEXT_CHARS, color="#e0a800", linestyle="--", label=f"판정기 자르는 길이 {MAX_TEXT_CHARS}자")
    ax.set_yscale("log")
    ax.set_xlabel("판정 대상 텍스트 글자 수 (4,000 이상은 4,000 에 모음)")
    ax.set_title("출처별 텍스트 길이 (dev)")
    ax.legend()
    save(fig, fig_dir / "text_length.png")

    # 3) 규칙 노드 문장 길이
    nodes = nodes.assign(words=nodes.rule_text.str.split().str.len())
    nl = nodes.groupby(["source", "level"]).words.agg(["size", "mean", "min", "max"]).round(1).reset_index() \
        .rename(columns={"size": "n_nodes", "mean": "words_mean", "min": "words_min", "max": "words_max"})
    nl.to_csv(out / "rule_node_length.csv", index=False)
    mid = nodes[nodes.level == 2]  # 같은 단계(중분류)에서 비교
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = [(SRC[s], g.words) for s, g in mid.groupby("source")]
    ax.boxplot([w for _, w in groups], tick_labels=[f"{s}\n(n={len(w)})" for s, w in groups])
    ax.set_ylabel("단어 수")
    ax.set_title(f"중분류 규칙 노드 문장 길이 (n={len(mid)})")
    save(fig, fig_dir / "rule_node_length.png")

    # 4) DNA 위험 영역 · 위해 유형 단위 위반
    h = pd.concat([both_splits(dna, "risk_area").rename(columns={"risk_area": "name"}).assign(level="risk_area"),
                   both_splits(dna, "types_of_harm").rename(columns={"types_of_harm": "name"}).assign(level="types_of_harm")])
    h[["level", "name", "scope", "n", "n_violation", "violation_rate"]].to_csv(out / "dna_harm_counts.csv", index=False)
    t = h[(h.level == "types_of_harm") & (h.scope == "all")].sort_values("n_violation")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(t.name, t.n_violation, color="#d64545")
    for i, (v, n) in enumerate(zip(t.n_violation, t.n)):
        ax.text(v, i, f" {v}/{n}", va="center", fontsize=8)
    ax.set_title(f"DNA 위해 유형 12개별 위반 건수 (n={int(t.n.sum()):,}, 위반 {int(t.n_violation.sum())})")
    ax.tick_params(axis="y", labelsize=8)
    save(fig, fig_dir / "dna_harm_types.png")
    print("EDA 보강 저장:", out)


if __name__ == "__main__":
    main()
