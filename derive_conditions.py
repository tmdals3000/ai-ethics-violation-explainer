"""
'규칙 제외' 조건 만들기 — 이미 있는 '기본' 조건의 후보별 판정으로 계산 (GPU 필요 없음)

PDF 정의: 처음 보는 규칙 실험. 정답 규칙이 규칙집에 없다고 가정하고 판정한다.
          실험 단위 = Do-Not-Answer 위해 유형 12개(중분류), Jigsaw 규칙 2개
계산: 텍스트마다 '기본' 조건의 상위 5개 후보 중
      정답 규칙과 같은 중분류(DNA) / 같은 규칙(Jigsaw)인 후보를 빼고, 남은 후보의 위반 확률 최댓값
      남은 후보가 하나도 없으면 0 (적용할 규칙이 없으니 위반 아님)
근사인 점: 원래는 정답 규칙을 뺀 트리에서 BFS 를 다시 돌려 빈자리를 6·7위 후보로 채워야 하지만,
          그 후보들의 판정이 없어서 남은 후보만 씀 → 실제보다 후보가 적음

실행: python derive_conditions.py
입력: texts.csv, rules.csv, base_candidates.csv
출력: judgments_derived.csv (text_id, judge, condition, p_violation, n_remaining_candidates)
"""
import pandas as pd

t = pd.read_csv("texts.csv", keep_default_na=False)
rules = pd.read_csv("rules.csv")
parent = dict(zip(rules.node_id, rules.parent))
bc = pd.read_csv("base_candidates.csv")

gold_unit = {}      # 텍스트별 '빼야 할 단위' (DNA: 정답 중분류, Jigsaw: 정답 규칙)
for r in t.itertuples():
    if r.source == "do_not_answer" and r.gold_l2:
        gold_unit[r.text_id] = r.gold_l2
    elif r.source == "jigsaw" and r.gold_l2:
        gold_unit[r.text_id] = r.gold_l2

bc = bc[bc.text_id.isin(gold_unit)].copy()
bc["unit"] = bc.cand.map(lambda c: parent.get(c, "") if c.startswith("L3:") else c)
bc["excluded"] = bc.unit == bc.text_id.map(gold_unit)
keep = bc[~bc.excluded]
agg = keep.groupby("text_id").agg(p_violation=("p_violation", "max"), n_remaining_candidates=("cand", "size"))
out = pd.DataFrame({"text_id": sorted(set(bc.text_id))}).merge(agg, left_on="text_id", right_index=True, how="left")
out["p_violation"] = out.p_violation.fillna(0.0)
out["n_remaining_candidates"] = out.n_remaining_candidates.fillna(0).astype(int)
out["judge"], out["condition"] = "ShieldGemma-2B", "rule_excluded"
out[["text_id", "judge", "condition", "p_violation", "n_remaining_candidates"]].to_csv("judgments_derived.csv", index=False)

src = out.text_id.str.split("-").str[0].map({"dna": "Do-Not-Answer", "jig": "Jigsaw"})
print("'규칙 제외' 판정 수:", src.value_counts().to_dict())
print("남은 후보 수 분포:", out.n_remaining_candidates.value_counts().sort_index().to_dict())
print(f"남은 후보가 없어 0 으로 둔 텍스트: {int((out.n_remaining_candidates == 0).sum())}개")
print("저장 완료: judgments_derived.csv")
