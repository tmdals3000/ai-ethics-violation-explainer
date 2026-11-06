"""조교 파일(similarity, coords, judgments, splits.csv) 읽기·쓰기 어댑터.

조교 파일이 ta_dir 에 있으면 그것을 읽고, 없으면 우리가 만든 임시 파일을 읽는다.
조교 파일 형식이 공개되면 이 파일의 read_* 함수만 고치면 된다.
반환값의 'is_temporary' 로 임시 데이터 여부를 화면·보고서에 표시한다.
"""
import pandas as pd

# 각 파일의 우리 쪽 표준 컬럼 (조교 형식을 모르는 상태의 임시 정의)
COLUMNS = {
    "splits": ["text_id", "split"],
    "similarity": ["text_id", "node_id", "similarity"],
    "coords": ["text_id", "x", "y"],
    "judgments": ["text_id", "model", "condition", "candidate_rule_id", "prompt_version",
                  "p_violation", "prediction", "threshold"],
}


def _ta_path(cfg, name):
    return cfg["_ta"] / f"{name}.csv"


def _ours_path(cfg, name):
    if name == "judgments":  # 판정 결과는 원문이 없어 results/ 에 둔다
        return cfg["_res"] / "judgments.csv"
    return cfg["_data"] / "derived" / f"{name}.csv"


def read(cfg, name):
    """(DataFrame 또는 None, is_temporary) 반환. 조교 파일 우선."""
    ta = _ta_path(cfg, name)
    if ta.exists():
        df = pd.read_csv(ta)
        missing = set(COLUMNS[name]) - set(df.columns)
        if missing:
            raise ValueError(f"조교 파일 {ta} 에 컬럼 {missing} 이 없습니다. adapters.py 매핑을 고치세요.")
        return df, False
    ours = _ours_path(cfg, name)
    if ours.exists():
        return pd.read_csv(ours, keep_default_na=False), True
    return None, True


def write_ours(cfg, name, df):
    p = _ours_path(cfg, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS[name] + [c for c in df.columns if c not in COLUMNS[name]]].to_csv(p, index=False)
    return p


def has_ta(cfg, name):
    return _ta_path(cfg, name).exists()
