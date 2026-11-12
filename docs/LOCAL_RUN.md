# GitHub에서 받아 로컬로 실행하기

Python 3.12에서 검증한다. 저장된 **실제 dev 3,539건의 분석 결과**를 사용하므로 화면 실행에는 GPU·HF 토큰·Kaggle 계정·원본 데이터 다운로드가 필요 없다. 가상의 판정이나 예시 데이터를 생성하지 않는다.

## Windows

Python 3.12와 Git을 설치한 뒤 PowerShell에서 실행한다.

```powershell
git clone --branch SeungMin https://github.com/tmdals3000/ai-ethics-violation-explainer.git
cd ai-ethics-violation-explainer
.\setup_dashboard.cmd
.\run_dashboard.cmd
```

GitHub에서 `SeungMin` 브랜치를 선택해 **Code → Download ZIP**으로 받아 압축을 풀어도 된다. 처음 한 번 `setup_dashboard.cmd`를 더블클릭하고, 설치가 끝나면 `run_dashboard.cmd`를 더블클릭한다. 브라우저가 열리지 않으면 `http://localhost:8501`에 접속한다. 실행 창에서 `Ctrl+C`를 누르면 종료한다.

설치는 프로젝트 `.venv`에 `requirements-dashboard.txt`의 화면용 의존성만 추가한다. 기존 `.venv`가 있으면 재사용한다. 실행 파일은 프로젝트 `.venv`를 우선 사용하고, 이 개발 PC에서는 기존 `C:\ml\aiethics\.venv`도 사용할 수 있다. 실제 데이터 경로는 저장소 안으로 잡으므로 다른 PC에 `C:\ml` 폴더를 만들 필요가 없다.

## macOS / Linux

```bash
git clone --branch SeungMin https://github.com/tmdals3000/ai-ethics-violation-explainer.git
cd ai-ethics-violation-explainer
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dashboard.txt
AIETHICS_DASHBOARD_SNAPSHOT=1 python -m streamlit run app/app.py
```

## 포함된 데이터와 가능한 작업

`assets/dev_dashboard.zip`에는 기존 실험의 dev 메타데이터·규칙 구조·유사도·PCA 좌표가 들어 있다. 앱이 첫 실행에 `data/dashboard_snapshot/derived/`로 풀어 사용한다. 판정과 지표는 이미 Git에 있는 `runs/baseline_v2/`를 읽는다.

| 포함 | 제외 |
|---|---|
| 실제 dev text_id·사람 라벨·정답 규칙·응답 LLM, 규칙 문장, 유사도, 좌표, 저장된 판정·지표 | 텍스트 원문·대화 맥락·위반/비위반 원문 예시, train/test 텍스트, 토큰, 모델 가중치, SQLite 판정 캐시 |

- P1~P4 필터·계층 탐색·모델 비교·지도·진단·임계값 변경·변화 흐름을 사용할 수 있다.
- 후보 확률이 이미 저장된 τ·집계 변경도 비교할 수 있다. 저장되지 않은 후보가 필요하면 안내를 표시한다.
- 원문 보기와 새 규칙의 GPU 재판정은 비활성화한다. 원문을 빈 문자열로 포함한 스냅샷으로 새 추론을 실행하지 않는다.
- `test` 평가는 포함하지 않는다. 화면에는 dev 실험의 한계와 임시 자료 상태가 표시된다.

## 원문 확인·모델 재판정까지 실행하려면

기존 README의 전체 환경 설정 절차대로 원본 데이터와 모델 라이선스·토큰을 준비하고 `requirements.txt`를 설치한다. `configs/baseline.yaml`의 `paths`를 본인 PC에 맞추고 파이프라인으로 가공 데이터와 판정을 준비한다. 이 작업에는 GPU와 큰 모델 다운로드가 필요하다.

이후 `AIETHICS_DASHBOARD_SNAPSHOT` 환경 변수를 해제하고 해당 전체 환경의 Python으로 실행한다. 예를 들어 이 개발 PC의 PowerShell에서는 다음과 같다.

```powershell
Remove-Item Env:AIETHICS_DASHBOARD_SNAPSHOT -ErrorAction SilentlyContinue
& C:/ml/aiethics/.venv/Scripts/python.exe -m streamlit run app/app.py
```

앱은 설정된 가공 데이터가 없으면 저장 결과로 자동 전환한다. CLI 파이프라인은 자동 전환하지 않으므로 스냅샷을 학습·새 추론 입력으로 사용하지 않는다.

## 스냅샷 갱신과 검증

실제 원본으로 전체 실험을 마친 개발 환경에서만 `python scripts/export_dashboard_snapshot.py`를 실행해 스냅샷을 갱신한다. 이 스크립트는 dev 표본만 고르고 텍스트·맥락·원문 예시를 제외한다. `manifest.json`에 행 수와 파일 해시를 기록한다.

```powershell
.venv\Scripts\python.exe -m pip install pytest==9.2.0
.venv\Scripts\python.exe -m pytest -q
$env:AIETHICS_DASHBOARD_SNAPSHOT = "1"
.venv\Scripts\python.exe scripts/verify_app.py
```

스냅샷의 dev 범위·해시·원문 제외, 새 추론 차단과 화면 필터·선택·되돌리기를 검증한다.

2026-11-12에는 Git에 포함될 파일만 추출한 복사본과 별도 Python 3.12 가상환경으로 확인했다. 화면용 의존성+pytest만 설치한 환경에서 테스트 **33개**, AppTest **14개 확인 지점**을 통과했다. `pip check`도 통과했고 PyTorch·Transformers·SentenceTransformers가 설치되거나 로드되지 않았음을 확인했다. AppTest 종료 시 이 PC의 임시 폴더 정리 권한 경고는 발생했으며 검증 종료 코드는 0이었다.
