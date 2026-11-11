"""
P3 원인 진단 (analyze.py 와 app.py 가 같이 씀)

잘못된 판정 하나 = (텍스트, 판정 모델) 한 쌍. 모델마다 따로 진단하고, 위에서부터 처음 해당하는 원인 하나만 붙인다.

  1. 규칙 찾기 문제 : 규칙을 입력받는 모델이 BFS 후보 규칙으로 판정했는데(base 조건)
                      (a) 정답 규칙이 상위 k 후보에 없음, 또는
                      (b) 정답 규칙은 후보에 있지만 다른 후보 규칙 때문에 판정이 바뀜
                          (최댓값을 낸 후보가 정답 규칙이 아니고, 같은 모델에 정답 규칙만 넣으면 맞음)
                      → Llama Guard·Qwen3Guard(규칙을 안 받음)와 '규칙 문장만' 조건(정답 규칙을 직접 넣음)은 해당 없음
  2. 라벨 확인 필요 : 모든 판정 모델이 같은 답인데 사람 라벨만 다름
  3. 규칙 문장 문제 : 정답 규칙 문장을 넣은 모델('규칙 문장만' 조건: ShieldGemma·팀 판정기)이 모두 틀림
  4. 판정 모델 문제 : 위 어디에도 해당하지 않음
"""
import pandas as pd

CAUSES = ["1. 규칙 찾기 문제", "2. 라벨 확인 필요", "3. 규칙 문장 문제", "4. 판정 모델 문제"]
RULE_TAKING = ("ShieldGemma", "Team")


def takes_rules(judge):
    return judge.startswith(RULE_TAKING)


def uses_gold_rule(judge):
    return judge.endswith("rule_only")


def is_base(judge):
    return takes_rules(judge) and judge.endswith("/ base")


def diagnose(m, judges, thresholds, gold_in_topk, base_argmax=None, gold_rule=None):
    """
    m            : text_id, human_harmful, 판정 모델별 위반 확률 열
    judges       : 판정 모델 열 이름 (예: 'ShieldGemma-2B / base')
    thresholds   : {판정 모델: 임계값}
    gold_in_topk : {text_id: 정답 규칙이 상위 k 후보 안에 있으면 True}
    base_argmax  : {text_id: base 조건에서 위반 확률 최댓값을 낸 후보 규칙}
    gold_rule    : {text_id: 정답 규칙 노드}
    """
    base_argmax = base_argmax or {}
    gold_rule = gold_rule or {}
    y = m["human_harmful"].values
    preds = pd.DataFrame({j: (m[j] >= thresholds.get(j, 0.5)).astype(int) for j in judges}, index=m.index)
    all_agree = (preds.nunique(axis=1) == 1).values
    gold_judges = [j for j in judges if uses_gold_rule(j)]
    gold_all_wrong = (preds[gold_judges].ne(m["human_harmful"], axis=0).all(axis=1).values
                      if gold_judges else [False] * len(m))

    rows = []
    for jc in judges:
        same_model_gold = jc.replace("/ base", "/ rule_only")
        for i in (preds[jc].values != y).nonzero()[0]:
            tid = m["text_id"].iloc[i]
            outside = not gold_in_topk.get(tid, True)
            cause = None
            if is_base(jc):
                other_rule = (base_argmax.get(tid) not in (None, gold_rule.get(tid))
                              and same_model_gold in preds and preds[same_model_gold].iloc[i] == y[i])
                if outside or other_rule:
                    cause = CAUSES[0]
            if cause is None:
                cause = CAUSES[1] if all_agree[i] else CAUSES[2] if gold_all_wrong[i] else CAUSES[3]
            rows.append({"text_id": tid, "judge": jc, "p_violation": round(float(m[jc].iloc[i]), 5),
                         "pred": int(preds[jc].iloc[i]), "human_harmful": int(y[i]),
                         "error_type": "놓친 위반" if y[i] == 1 else "잘못 잡은 위반",
                         "cause": cause, "gold_outside_topk": int(outside)})
    return pd.DataFrame(rows, columns=["text_id", "judge", "p_violation", "pred", "human_harmful",
                                       "error_type", "cause", "gold_outside_topk"])
