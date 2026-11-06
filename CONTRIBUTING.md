# 협업 규칙

이 레포는 GitHub Flow를 사용합니다.

## 브랜치 전략

| 브랜치 | 역할 | 만드는 곳 | 병합 대상 |
|---|---|---|---|
| `main` | 항상 실행 가능한 최신 버전 | - | - |
| `feature/*` | 기능 개발 | `main` | `main` |
| `fix/*` | 버그 수정 | `main` | `main` |
| `docs/*` | 문서 작업 | `main` | `main` |

`main`에는 직접 push할 수 없습니다. PR을 만들고 다른 팀원 1명 이상의 승인을 받아야 병합됩니다. 작업 브랜치에는 자유롭게 push할 수 있습니다.

## 브랜치 이름

- 기능: `feature/이슈번호-기능요약` (예: `feature/12-rule-similarity`)
- 버그 수정: `fix/이슈번호-내용요약` (예: `fix/15-threshold-error`)
- 문서: `docs/내용요약` (예: `docs/readme`)

## 작업 순서

1. 이슈 템플릿으로 이슈를 만듭니다.
2. 최신 `main`에서 브랜치를 만듭니다.
3. 작업 후 내 브랜치에 push합니다.
4. `main`으로 PR을 만들고 PR 템플릿을 작성합니다.
5. 다른 팀원의 승인을 받아 병합하고, 작업 브랜치를 삭제합니다.

```bash
git switch main
git pull origin main
git switch -c feature/12-rule-similarity
git add .
git commit -m "feat: 규칙 찾기 단계별 유사도 계산"
git push -u origin feature/12-rule-similarity
```

작업이 길어지면 중간에 `main`의 변경을 받아 충돌을 줄입니다.

```bash
git switch main
git pull origin main
git switch feature/12-rule-similarity
git merge main
```

## 커밋 메시지

`타입: 작업 내용` 형식으로 작성합니다.

- `feat`: 기능 추가
- `fix`: 버그 수정
- `docs`: 문서 수정
- `refactor`: 동작 변경 없는 코드 정리
- `test`: 테스트 추가·수정
- `chore`: 설정 등 기타 작업

## 주의 사항

- `main`에 바로 반영되므로 PR 전에 로컬에서 실행해 확인합니다.
- 레포가 공개 상태이므로 API 키, 토큰, 비공개 데이터는 올리지 않습니다.
