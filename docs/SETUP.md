# 환경 설정 (Windows)

RTX 3060(12GB)에서 실행을 확인한 환경입니다. 시스템에 한 번 설치할 것과 프로젝트 가상환경에 설치할 것을 구분합니다.

| 항목 | 버전 | 설치 위치 |
| --- | --- | --- |
| Python | 3.12 | 시스템 (여러 버전과 함께 설치 가능) |
| Java | 17 (Temurin) | 시스템. Spark 4.x에 필요 |
| PySpark | 4.x | 가상환경 (`pip`로 설치하면 Spark 본체가 함께 들어 있음) |
| torch | CUDA 빌드 | 가상환경 |
| 나머지 패키지 | `requirements.txt` | 가상환경 |

## 1. 시스템 설치 (PowerShell)

```powershell
winget install Python.Python.3.12 --source winget
winget install EclipseAdoptium.Temurin.17.JDK --source winget
```

설치 후 VS Code를 완전히 껐다 켜고 확인합니다.

```powershell
py -0            # 3.12가 보이면 됨
java -version    # 17.x
nvidia-smi       # NVIDIA GPU 확인
```

- winget이 Microsoft Store 약관 동의를 물으면 `--source winget`을 붙이면 묻지 않습니다.
- 프로젝트 경로에 **한글과 공백이 없어야** 합니다 (예: `C:\dev\ai-ethics-violation-explainer`).

## 2. 가상환경 (VS Code 터미널)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python --version          # 3.12.x
```

- "스크립트를 실행할 수 없으므로" 오류: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 후 다시 활성화
- `Ctrl + Shift + P` → **Python: Select Interpreter** → `.venv` 선택
- VS Code가 "global environment" 경고를 띄워도 프롬프트에 `(.venv)`가 있으면 가상환경에 설치되고 있는 것입니다.

## 3. 패키지

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # True
```

- 그냥 `pip install torch`는 Windows에서 **CPU 전용**으로 설치됩니다. `cu130`이 없다는 오류가 나면 `cu128`로 바꿉니다.
- GPU가 없으면 `python -m pip install torch`로 설치할 수 있지만 판정이 매우 느립니다.

## 4. Hugging Face

```powershell
hf auth login        # "Log in with your browser" → https://hf.co/oauth/device 에서 코드 입력
hf auth whoami
```

| 모델 | 로그인 | 라이선스 동의 |
| --- | --- | --- |
| google/shieldgemma-2b | 필요 | 모델 페이지에서 Acknowledge license (즉시) |
| meta-llama/Llama-Guard-3-1B | 필요 | 모델 페이지에서 신청 (승인 대기) |
| Qwen/Qwen3Guard-Gen-0.6B | 불필요 | 없음 |
| sentence-transformers/all-roberta-large-v1 | 불필요 | 없음 |

모델은 처음 실행할 때 `C:\Users\<사용자>\.cache\huggingface\hub`에 받아집니다 (4개 합쳐 약 11GB). C 드라이브가 부족하면
`setx HF_HOME "D:\hf_cache"`로 위치를 바꿀 수 있습니다.

## 5. 동작 확인

```powershell
pyspark --version
python scripts\spark_team_judge.py --check
```

마지막 명령이 3건의 변형별 위반 확률을 출력하면 Spark, GPU, 모델, 로그인까지 정상입니다.

## 무시해도 되는 메시지

| 메시지 | 이유 |
| --- | --- |
| `Using incubator modules: jdk.incubator.vector` | Java 시험 기능 알림 |
| `Did not find winutils.exe`, `HADOOP_HOME ... unset` | Spark가 직접 파일을 쓸 때만 필요. 우리는 pandas로 저장 |
| `Unable to load native-hadoop library` | Windows용 하둡 라이브러리 없음 |
| `성공: PID ...인 프로세스가 종료되었습니다` | Spark 종료 시 Windows가 프로세스를 정리하는 메시지 |
| `huggingface_hub ... symlinks` | Windows는 심볼릭 링크 대신 파일을 복사해 캐시 (용량만 조금 더 씀) |
| `torch_dtype is deprecated` | transformers 인자 이름 변경 알림 |

## 자주 나는 오류

| 증상 | 해결 |
| --- | --- |
| `Python worker exited unexpectedly` | 스크립트가 `PYSPARK_PYTHON`을 현재 파이썬으로 지정하는지 확인, `local[1]`로 실행 (모든 스크립트에 이미 들어 있음) |
| `PySpark does not yet fully support pandas >= 3.0.0` | `python -m pip install "pandas<3"` |
| `torch.cuda.is_available()`가 False | CPU용 torch가 설치됨 → `--index-url`로 다시 설치 |
| 판정 중 멈춘 것처럼 보임 | 진행 막대가 작업 단위로만 움직여서 그렇습니다. 새 터미널에서 `nvidia-smi`로 GPU 사용률 확인 |
| 아주 느림, 오류는 없음 | 작업 관리자 → GPU → "공유 GPU 메모리"가 쓰이면 메모리 부족. `--batch-size` 또는 응답 길이(`max_chars`)를 줄임 |
| `401`, `403`, `gated repo` | `hf auth login`과 모델 페이지 라이선스 동의 확인 |
| `FileNotFoundError: ...template.html` | `app/` 폴더에 `*_template.html`이 있는지 확인 |
