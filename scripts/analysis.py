"""2주차 분석 함수 모음 (PDF '함수 정리': BFS 규칙 찾기, 지표 계산, 수정 효과 재계산 + 원인 규칙 1–4)

스크립트(rule_search.py, build_conditions.py, diagnose.py, compare_conditions.py, whatif.py)와
3주차 뷰가 같은 함수를 쓰도록 여기에 모읍니다. 화면에서 τ·임계값·k를 바꿀 때도 이 함수로 다시 계산합니다.

    tree = Tree(rule_tree_df, "dna")
    leaves, n_cmp, kept, fb = bfs(tree, sim, tau=0.08)                 # 중분류부터 (PDF 2주차)
    leaves, *_ = bfs(tree, sim, tau, exclude={"dna_L2_3"})             # '규칙 제외': 그 가지를 트리에서 뺌
    source_metrics(judg, thresholds)                                   # 출처·모델·조건별 지표
    diagnose(...)                                                      # 원인 규칙 1–4
    flips(p, y, thr_old, thr_new)                                      # 뒤집힌 판정 (맞게/틀리게)
"""
import numpy as np
import pandas as pd

from common import best_f1_threshold, binary_metrics

# ---------------------------------------------------------------- BFS 규칙 찾기
class Tree:
    """rule_tree.csv의 한 출처 부분. level 1 = 대분류, 2 = 중분류, 3 = 소분류 (Jigsaw는 2단계가 규칙)."""

    def __init__(self, tree: pd.DataFrame, source: str):
        t = tree[tree.source == source]
        self.source = source
        self.n_nodes = len(t)
        self.level = dict(zip(t.node_id, t.level.astype(int)))
        self.parent = dict(zip(t.node_id, t.parent_id))
        self.name = dict(zip(t.node_id, t.name))
        self.text = dict(zip(t.node_id, t.node_text))
        self.children = {n: [] for n in t.node_id}
        for n, p in zip(t.node_id, t.parent_id):
            if p:
                self.children[p].append(n)
        self.leaves = [n for n in t.node_id if not self.children[n]]
        self.max_level = max(self.level.values())

    def path(self, node):
        out = []
        while node:
            out.append(node)
            node = self.parent[node]
        return out[::-1]                                   # [L1, L2, (L3)]

    def at_level(self, lv):
        """시작 단계의 노드. 트리가 그보다 얕으면(Jigsaw의 L3) 잎."""
        lv = min(lv, self.max_level)
        return [n for n, l in self.level.items() if l == lv]

    def unit_of(self, leaf):
        """'규칙 제외' 실험 단위 = 중분류(L2). DNA는 위해 유형 12개, Jigsaw는 규칙 2개 (PDF)."""
        p = self.path(leaf)
        return p[1] if len(p) > 1 else p[0]

    def leaves_under(self, node):
        if not self.children[node]:
            return [node]
        return [x for c in self.children[node] for x in self.leaves_under(c)]


def bfs(tree: Tree, sim: dict, tau: float, start_level: int = 2, exclude=(), boost=None):
    """sim: node_id → 유사도.

    start_level=2  PDF 2주차 정의 "τ 이상인 중분류만 남기고 소분류로 내려갑니다" (기본)
    start_level=1  대분류부터 거름 (1주차 구현)
    exclude        트리에서 뺄 노드 (그 아래 가지 전체가 빠짐) → '규칙 제외' 조건
    boost          node_id → 유사도에 더할 값 → P4 '우선순위 바꾸기'. 그 노드와 아래 가지 전체에 더함

    단계마다 τ 이상만 남기고, 하나도 없으면 가장 높은 하나만 남기고 fallback 표시.
    반환: (찾은 잎을 점수 순으로, 비교한 노드 수, 단계별 남긴 노드, fallback 여부)
    """
    ex = set(exclude)
    if boost:                                   # 노드에 더한 값은 그 아래 가지 전체에 적용 (가지를 통째로 앞/뒤로)
        add = {}
        for node, v in boost.items():
            stack = [node]
            while stack:
                n = stack.pop()
                add[n] = add.get(n, 0.0) + v
                stack.extend(tree.children.get(n, []))
        score = lambda n: sim[n] + add.get(n, 0.0)
    else:
        score = lambda n: sim[n]
    frontier = [n for n in tree.at_level(start_level) if n not in ex]
    leaves, kept_by_level, n_cmp, fb = [], {}, 0, False
    while frontier:
        n_cmp += len(frontier)
        keep = [n for n in frontier if score(n) >= tau]
        if not keep:
            keep, fb = [max(frontier, key=score)], True
        kept_by_level[tree.level[keep[0]]] = set(keep)
        nxt = []
        for n in keep:
            kids = [c for c in tree.children[n] if c not in ex]
            (nxt.extend(kids) if kids else (leaves.append(n) if not tree.children[n] else None))
        frontier = nxt
    leaves.sort(key=lambda n: -score(n))
    return leaves, n_cmp, kept_by_level, fb


# ---------------------------------------------------------------- 지표
def best_kappa_threshold(p, y) -> float:
    """사람 라벨과의 Cohen's κ가 최대인 임계값. 전부 위반(또는 전부 정상)으로 판정하면 κ = 0이라
    F1 최대가 '전부 위반'으로 무너지는 경우를 피함. 동률이면 높은 임계값."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    n, pos = len(y), int(y.sum())
    if pos in (0, n):
        return 0.5
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    last = np.r_[ps[1:] != ps[:-1], True]
    tp, fp, thr = tp[last], fp[last], ps[last]
    pred_pos = tp + fp
    po = (tp + (n - pos - fp)) / n
    pe = (pred_pos * pos + (n - pred_pos) * (n - pos)) / n ** 2
    kappa = np.where(pe < 1, (po - pe) / (1 - pe), 0.0)
    return float(thr[int(np.argmax(kappa))])


THRESHOLD_RULES = {"f1": best_f1_threshold, "kappa": best_kappa_threshold}


def thresholds_on_dev(judg: pd.DataFrame, by=("source", "model", "condition"), criterion: str = "f1") -> dict:
    """묶음마다 dev에서 기준(f1 | kappa)이 최대인 임계값. judg 열: split, human_label, p_violation, + by"""
    dev = judg[judg.split == "dev"].dropna(subset=["p_violation"])
    f = THRESHOLD_RULES[criterion]
    return {k: f(g.p_violation, g.human_label) for k, g in dev.groupby(list(by))}


def degenerate(judg: pd.DataFrame, thr: dict, by=("source", "model", "condition"), share=0.95) -> list:
    """임계값이 거의 모두를 한쪽으로 판정하는 묶음 (dev의 share 이상이 위반 또는 정상)."""
    out = []
    dev = judg[judg.split == "dev"].dropna(subset=["p_violation"])
    for k, g in dev.groupby(list(by)):
        r = float((g.p_violation >= thr.get(k, 0.5)).mean())
        if r >= share or r <= 1 - share:
            out.append((k, r))
    return out


def source_metrics(judg: pd.DataFrame, thresholds: dict, by=("source", "model", "condition")) -> pd.DataFrame:
    """출처·모델·조건별 ROC-AUC, PR-AUC, F1, 사람 라벨과의 κ (PDF 지표 정의, scikit-learn).
    thresholds는 thresholds_on_dev 결과 (test에도 dev 임계값을 그대로 씀)."""
    rows = []
    for k, g in judg.dropna(subset=["p_violation"]).groupby(list(by) + ["split"]):
        thr = thresholds.get(k[:-1], 0.5)
        rows.append({**dict(zip(list(by) + ["split"], k)), **binary_metrics(g.p_violation, g.human_label, thr)})
    return pd.DataFrame(rows)


def auc_or_nan(p, y):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, int)
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


# ---------------------------------------------------------------- 4가지 조건 묶기
CONDITIONS = {
    "base": "기본 (BFS 상위 5개 후보마다 판정 → 가장 높은 위반 확률)",
    "excluded": "규칙 제외 (정답 규칙의 중분류를 트리에서 빼고 찾은 상위 5개 → 처음 보는 규칙)",
    "rule_only": "규칙 문장만 (정답 규칙 문장 하나)",
    "rule_example": "규칙 문장 + 예시 (정답 규칙 문장 + 위반·비위반 예시)",
}


def assemble_conditions(cache: pd.DataFrame, cmap: pd.DataFrame) -> pd.DataFrame:
    """후보별 판정(cache) → 텍스트·조건별 위반 확률 (judgments.csv 형식).

    cache: text_id, rule_id, with_examples, variant, p
    cmap : text_id, condition, rank, rule_id, with_examples
    여러 후보로 판정하는 조건(기본, 규칙 제외)은 후보 중 가장 높은 위반 확률 = "어느 규칙이든 어기면 위반".
    """
    keys = ["text_id", "rule_id", "with_examples"]
    grid = cmap.merge(pd.DataFrame({"variant": sorted(cache.variant.unique())}), how="cross")
    m = (grid.merge(cache[keys + ["variant", "p"]], on=keys + ["variant"], how="left")
             .sort_values("p", ascending=False, na_position="last"))
    agg = (m.groupby(["text_id", "condition", "variant"], sort=False)
             .agg(p_violation=("p", "max"), n_rules=("rule_id", "size"),
                  n_missing=("p", lambda s: int(s.isna().sum())), top_rule=("rule_id", "first"))
             .reset_index())
    agg.loc[agg.n_missing > 0, "p_violation"] = np.nan        # 후보 판정이 덜 끝난 조건은 비움
    agg["model"] = agg.variant.replace({"team_v0": "shieldgemma"})   # team_v0 = 기준 ShieldGemma와 같은 값
    return agg.drop(columns="variant")


# ---------------------------------------------------------------- 원인 규칙 1–4 (P3)
CAUSES = {
    1: "규칙 찾기 문제",
    2: "라벨 확인 필요",
    3: "규칙 문장 문제",
    4: "판정 모델 문제",
}
RULE_FREE = ("llamaguard", "qwen3guard")                     # 규칙 문장을 받지 않는 판정 모델


def diagnose(wide: pd.DataFrame, rule_models, k: int = 5) -> pd.DataFrame:
    """잘못된 판정(텍스트 × 모델)마다 원인 하나. 위에서부터 차례로 적용 (PDF).

    wide 열 (텍스트 1행):
      human_label, gold_rank (BFS로 찾은 규칙 중 정답 순위, 없으면 nan)
      pred_{m}            모델 m의 '기본' 조건 판정 (0/1)  ← 진단 대상
      pred_{m}_rule_only  규칙 문장을 받는 모델의 '규칙 문장만' 판정
    rule_models: 규칙 문장을 받는 모델 (shieldgemma, 팀 판정기)
    k: '규칙 찾기 문제'로 볼 상위 k (정답이 상위 k 밖이면 규칙 찾기 문제)

    1 규칙 찾기 문제  규칙을 받는 모델만 해당. 정답 규칙이 상위 k 밖이거나,
                      '규칙 문장만'에서는 맞는데 '기본'에서 틀림 (함께 찾은 다른 규칙 때문에 판정이 바뀜)
    2 라벨 확인 필요  모든 모델(규칙 모델은 '규칙 문장만')이 같은 답이고 사람 라벨만 다름
    3 규칙 문장 문제  규칙을 받는 모델이 정답 규칙 문장을 넣어도 모두 틀림 (규칙 모델의 오판만)
    4 판정 모델 문제  나머지: 정답 규칙 문장을 넣었을 때 일부 모델만 틀림
    """
    rule_models = list(rule_models)
    models = [c[5:] for c in wide.columns if c.startswith("pred_") and not c.endswith("_rule_only")]
    y = wide.human_label.astype(int)
    with_gold = {m: wide[f"pred_{m}_rule_only"] if m in rule_models else wide[f"pred_{m}"] for m in models}
    wrong_gold = pd.DataFrame({m: with_gold[m] != y for m in models})
    all_wrong = wrong_gold.all(axis=1)                         # 모두 같은 답 & 사람과 다름 (이진이라 동치)
    rule_all_wrong = wrong_gold[rule_models].all(axis=1)
    out_of_k = ~(wide.gold_rank <= k)                          # nan(못 찾음)도 True

    rows = []
    for m in models:
        wrong = wide[f"pred_{m}"] != y
        if m in rule_models:
            changed = with_gold[m] == y                        # 정답 규칙만 넣으면 맞음
            c1 = out_of_k | changed
        else:
            c1 = pd.Series(False, index=wide.index)
        cause = np.select([c1, all_wrong, rule_all_wrong & (m in rule_models)], [1, 2, 3], 4)
        why = np.select(
            [c1 & out_of_k, c1, all_wrong, rule_all_wrong & (m in rule_models)],
            [f"정답 규칙이 상위 {k}개 밖", "정답 규칙만 넣으면 맞음 (다른 후보 규칙 때문에 바뀜)",
             "모든 모델이 같은 답, 사람 라벨만 다름", "규칙 문장을 받는 모델이 정답 규칙으로도 모두 틀림"],
            "일부 모델만 틀림")
        d = wide.loc[wrong, ["text_id"]].copy()
        d["model"], d["cause"], d["cause_name"], d["why"] = m, cause[wrong], "", why[wrong]
        d["error"] = np.where(y[wrong] == 1, "FN (놓침)", "FP (과잉)")
        d["all_models_wrong"] = all_wrong[wrong].values      # 순서상 1번이 먼저라도, 라벨 확인 후보인지 뷰에서 보이게
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out["cause_name"] = out.cause.map(CAUSES)
    return out


# ---------------------------------------------------------------- 수정 효과 재계산 (P4)
def flips(p_old, p_new, y, thr_old, thr_new=None) -> dict:
    """설정을 바꿨을 때 뒤집힌 판정. p_new가 같으면 임계값만 바꾼 것.
    반환: 바뀐 수, 맞게 바뀐 수(틀림→맞음), 틀리게 바뀐 수(맞음→틀림), 위치(mask)"""
    thr_new = thr_old if thr_new is None else thr_new
    y = np.asarray(y, int)
    a = (np.asarray(p_old, float) >= thr_old).astype(int)
    b = (np.asarray(p_new, float) >= thr_new).astype(int)
    changed = a != b
    fixed = changed & (b == y)
    broken = changed & (b != y)
    return {"flipped": int(changed.sum()), "fixed": int(fixed.sum()), "broken": int(broken.sum()),
            "mask_fixed": fixed, "mask_broken": broken}


def recompute_base(cache: pd.DataFrame, found: pd.Series, variant: str, top_n: int = 5) -> pd.DataFrame:
    """τ·우선순위를 바꿔 찾은 규칙이 달라졌을 때 '기본' 조건 위반 확률을 판정 캐시로 다시 계산.
    found: text_id → 찾은 잎 목록. 캐시에 없는 (텍스트, 규칙) 쌍은 missing으로 돌려줌 → 판정 후 다시 계산."""
    c = cache[(cache.variant == variant) & (~cache.with_examples)].set_index(["text_id", "rule_id"]).p
    rows, missing = [], []
    for tid, leaves in found.items():
        ps = []
        for r in list(leaves)[:top_n]:
            v = c.get((tid, r))
            (ps.append(v) if v is not None and not pd.isna(v) else missing.append((tid, r)))
        rows.append({"text_id": tid, "p_violation": max(ps) if ps and len(ps) == min(top_n, len(leaves)) else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(missing, columns=["text_id", "rule_id"])
