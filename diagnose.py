"""
P3 원인 진단 함수 (analyze.py 와 app.py 가 같이 씀)

잘못된 판정 하나 = (텍스트, 판정 모델) 한 쌍. 모델마다 따로 진단한다.
위에서부터 처음 해당하는 원인 하나만 붙인다.

  1. 규칙 찾기 문제 : 규칙을 입력받는 모델이 'BFS 로 찾은 후보 규칙'으로 판정했는데(base 조건),
                      정답 규칙이 상위 k 후보에 없음
                      → Llama Guard·Qwen3Guard 는 규칙을 입력받지 않으므로 해당 없음
                      → '규칙 문장만' 조건은 정답 규칙을 직접 넣으므로 해당 없음
  2. 라벨 확인 필요 : 모든 판정 모델이 같은 답인데 사람 라벨만 다름
  3. 규칙 문장 문제 : 정답 규칙 문장을 넣은 모델(ShieldGemma '규칙 문장만')이 모두 틀림
  4. 판정 모델 문제 : 위 어디에도 해당하지 않음 (정답 규칙을 넣으면 맞는데 이 모델은 틀림)

※ 데모 한계: 규칙을 입력받는 모델의 base 판정(BFS 후보로 판정)을 돌리지 않아서 1번은 나오지 않음.
             정답 규칙이 상위 k 후보 밖인지는 'gold_outside_topk' 열에 참고로 표시함.
"""
import pandas as pd

CAUSES = ["1. 규칙 찾기 문제", "2. 라벨 확인 필요", "3. 규칙 문장 문제", "4. 판정 모델 문제"]
RULE_TAKING = ("ShieldGemma", "Team")      # 규칙 문장을 입력받는 모델 이름 앞부분


def takes_rules(judge):
    return judge.startswith(RULE_TAKING)


def uses_gold_rule(judge):
    return "rule_only" in judge


def diagnose(m, judges, threshold, gold_in_topk):
    """
    m           : text_id, human_harmful, 그리고 판정 모델별 위반 확률 열이 있는 표
    judges      : 판정 모델 열 이름 목록 (예: 'ShieldGemma-2B / rule_only')
    threshold   : 이 값 이상이면 '위반'
    gold_in_topk: {text_id: 정답 규칙이 상위 k 후보 안에 있으면 True}
    반환        : 잘못된 판정 한 줄당 한 행 (text_id, judge, p_violation, pred, human_harmful, cause, ...)
    """
    y = m["human_harmful"].values
    preds = (m[judges] >= threshold).astype(int)
    all_agree = (preds.nunique(axis=1) == 1).values
    gold_judges = [j for j in judges if uses_gold_rule(j)]
    gold_all_wrong = (preds[gold_judges].ne(m["human_harmful"], axis=0).all(axis=1).values
                      if gold_judges else [False] * len(m))

    rows = []
    for jc in judges:
        wrong = preds[jc].values != y
        for i in wrong.nonzero()[0]:
            tid = m["text_id"].iloc[i]
            outside = not gold_in_topk.get(tid, True)
            if takes_rules(jc) and not uses_gold_rule(jc) and outside:
                cause = CAUSES[0]
            elif all_agree[i]:
                cause = CAUSES[1]
            elif gold_all_wrong[i]:
                cause = CAUSES[2]
            else:
                cause = CAUSES[3]
            rows.append({"text_id": tid, "judge": jc, "p_violation": round(float(m[jc].iloc[i]), 5),
                         "pred": int(preds[jc].iloc[i]), "human_harmful": int(y[i]),
                         "error_type": "놓친 위반" if y[i] == 1 else "잘못 잡은 위반",
                         "cause": cause, "gold_outside_topk": int(outside)})
    return pd.DataFrame(rows, columns=["text_id", "judge", "p_violation", "pred", "human_harmful",
                                       "error_type", "cause", "gold_outside_topk"])
