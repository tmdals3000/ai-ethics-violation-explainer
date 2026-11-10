"""
1단계: Do-Not-Answer 전처리
- 원본(data_en.csv)은 질문 1개당 한 행이고, 응답 LLM 6개의 답변이 옆으로 붙어 있음
- 이걸 "응답 1개 = 한 행"으로 펼친다 (939행 x 6개 = 5,634행)

실행: python preprocess.py
입력: data_en.csv (이 파일과 같은 폴더)
출력: dna_long.csv
"""
import pandas as pd

# 1) 원본 읽기
df = pd.read_csv("data_en.csv")
print("원본 크기:", df.shape)  # (939, 23)

# 2) 응답 LLM 6개 이름 (컬럼 이름 앞부분과 같음. 예: GPT4_response)
llms = ["GPT4", "ChatGPT", "Claude", "ChatGLM2", "llama2-7b-chat", "vicuna-7b"]

# 3) LLM마다 필요한 컬럼만 뽑아서 같은 이름으로 맞춘 뒤, 세로로 이어 붙이기
pieces = []
for llm in llms:
    part = df[["id", "risk_area", "types_of_harm", "specific_harms", "question"]].copy()
    part["response_llm"] = llm                          # 어느 LLM의 응답인지
    part["response"] = df[f"{llm}_response"]           # 응답 원문
    part["human_harmful"] = df[f"{llm}_harmful"]       # 사람 라벨 (1=유해/위반, 0=정상)
    part["action"] = df[f"{llm}_action"]               # 응답 유형 (0~5, 거절 방식 등)
    pieces.append(part)

long = pd.concat(pieces, ignore_index=True)

# 4) 규칙 트리 3단계 이름 정리
#    specific_harms 는 대부분 "이름: 설명" 형태라서 이름만 떼어 둔다
#    (콜론 없이 설명만 있는 소분류도 8개 있음 → 그때는 이름과 설명을 같은 문장으로 둠)
long = long.rename(columns={"id": "question_id",
                            "risk_area": "rule_l1",        # 대분류 (5개)
                            "types_of_harm": "rule_l2"})   # 중분류 (12개)
long["rule_l3"] = long["specific_harms"].str.split(":").str[0].str.strip()  # 소분류 (61개)
long["rule_l3_desc"] = long["specific_harms"].str.split(":", n=1).str[1].str.strip()
long["rule_l3_desc"] = long["rule_l3_desc"].fillna(long["rule_l3"])
long = long.drop(columns="specific_harms")

# 5) 텍스트마다 고유 ID 붙이기 (예: dna-0-GPT4)
#    승민 님 judgments.csv 와 같은 형식이라 그 파일과 바로 연결된다
long["text_id"] = "dna-" + long["question_id"].astype(str) + "-" + long["response_llm"]
long["source"] = "do_not_answer"

# 6) 컬럼 순서 정리 후 저장
cols = ["text_id", "source", "question_id", "response_llm",
        "rule_l1", "rule_l2", "rule_l3", "rule_l3_desc",
        "question", "response", "human_harmful", "action"]
long = long[cols]
long.to_csv("dna_long.csv", index=False)

# 7) 간단 확인
print("펼친 크기:", long.shape)  # (5634, 12)
print("결측치 개수:", int(long.isna().sum().sum()))
print("text_id 중복 없음:", long["text_id"].is_unique)
print("\n유해(위반) 비율:", round(long["human_harmful"].mean() * 100, 1), "%")
print("\n대분류별 건수:")
print(long["rule_l1"].value_counts())
print("\n저장 완료: dna_long.csv")
