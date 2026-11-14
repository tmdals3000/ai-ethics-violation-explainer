"""전처리·분석 스크립트가 같이 쓰는 함수.

공통 형식(texts_all.csv)의 열:
    text_id     고유 id (규칙: {source}_{...}. 팀 확정 전 임시)
    source      dna | jigsaw | prosocial
    split       train | dev | test (임시 분할. 팀이 확정하면 --splits로 교체)
    text        판정 대상 텍스트 (dna: LLM 응답, jigsaw: 댓글, prosocial: 사용자 발화)
    context     맥락 (dna: 사용자 질문, jigsaw: 서브레딧, prosocial: 앞선 대화)
    judge_mode  response = 질문+응답을 넣고 응답을 판정 / text = 텍스트 하나를 판정
    human_label 사람 라벨 (1 = 위반)
    gold_rule   정답 규칙 이름 (prosocial은 없음)
    rule_text   정답 규칙 문장
    rule_l1, rule_l2  규칙의 상위 분류 (dna: 위험 영역, 위해 유형 / jigsaw: 커뮤니티 규칙, 규칙 이름)
    responder   응답 LLM (dna만)
    label_raw   원본 라벨 값 (dna: action, prosocial: safety_label)
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

COMMON_COLS = ["text_id", "source", "split", "text", "context", "judge_mode", "human_label",
               "gold_rule", "rule_text", "rule_l1", "rule_l2", "responder", "label_raw"]


def hash_bucket(key, n=100) -> int:
    """문자열 → 0..n-1. 같은 key는 항상 같은 값 (실행할 때마다 분할이 바뀌지 않게)."""
    return int(hashlib.md5(str(key).encode("utf-8")).hexdigest(), 16) % n


def temp_split(key, ratios=(("dev", 50), ("test", 50))) -> str:
    """임시 분할. ratios = ((이름, 퍼센트), ...) 합이 100."""
    b, acc = hash_bucket(key), 0
    for name, pct in ratios:
        acc += pct
        if b < acc:
            return name
    return ratios[-1][0]


# 결과 파일 → 그 파일을 만드는 스크립트 (파일이 없을 때 안내용)
MADE_BY = {
    "texts_dna.csv": "prepare_dna.py", "texts_jigsaw.csv": "prepare_jigsaw.py",
    "texts_prosocial.csv": "prepare_prosocial.py", "texts_all.csv": "combine_texts.py",
    "rule_tree.csv": "build_rule_tree.py", "gold_paths.csv": "build_rule_tree.py",
    "similarity.parquet": "compute_embeddings.py", "rule_search.csv": "rule_search.py",
    "conditions_map.csv": "build_conditions.py", "conditions_jobs.csv": "build_conditions.py",
    "judgments_cache.csv": "spark_conditions.py", "judgments_conditions.csv": "spark_conditions.py",
}


def need(*names):
    """data/processed의 파일이 다 있는지 확인. 없으면 먼저 실행할 스크립트를 알려 주고 종료."""
    missing = [n for n in names if not (PROCESSED / n).exists()]
    if missing:
        order = list(dict.fromkeys(MADE_BY.values()))           # 파이프라인 순서
        steps = sorted(set(MADE_BY.get(n, "?") for n in missing), key=lambda x: order.index(x) if x in order else 99)
        raise SystemExit("필요한 파일이 없습니다: " + ", ".join(f"data/processed/{n}" for n in missing)
                         + "\n먼저 실행하세요: " + " → ".join(f"python scripts/{s}" for s in steps))


def norm_text(s) -> str:
    """중복 판단용 정규화: 앞뒤 공백 제거, 연속 공백 하나로, 소문자."""
    return " ".join(str(s).split()).lower()


def require_columns(df: pd.DataFrame, cols, what: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[{what}] 필요한 열이 없습니다: {missing}\n  실제 열: {list(df.columns)}")


# ---------- 지표 (성능 판별 페이지, scikit-learn과 같은 정의) ----------
def best_f1_threshold(p, y) -> float:
    """F1(위반 기준)이 최대인 임계값. 후보 = 서로 다른 확률값, 동률이면 높은 임계값."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    pos = int(y.sum())
    if pos == 0 or len(p) == 0:
        return 0.5
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    last = np.r_[ps[1:] != ps[:-1], True]            # 같은 확률값 묶음의 마지막 위치
    f1 = 2 * tp[last] / (2 * tp[last] + fp[last] + (pos - tp[last]))
    return float(ps[last][int(np.argmax(f1))])


def confusion(pred, y):
    pred, y = np.asarray(pred, int), np.asarray(y, int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def binary_metrics(p, y, thr) -> dict:
    from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score
    p, y = np.asarray(p, float), np.asarray(y, int)
    pred = (p >= thr).astype(int)
    tp, fp, fn, tn = confusion(pred, y)
    both = len(np.unique(y)) == 2
    return {
        "n": len(y), "violations": int(y.sum()),
        "roc_auc": roc_auc_score(y, p) if both else np.nan,
        "pr_auc": average_precision_score(y, p) if y.sum() else np.nan,
        "threshold": thr,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "kappa_human": cohen_kappa_score(y, pred) if both and len(np.unique(pred)) == 2 else np.nan,
        "fn": fn, "fp": fp, "error_rate": (fn + fp) / len(y) if len(y) else np.nan,
    }
