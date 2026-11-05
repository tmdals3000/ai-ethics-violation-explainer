# 협업 규칙

이 레포는 Git Flow를 사용합니다.

## 브랜치 전략

| 브랜치 | 역할 | 만드는 곳 | 병합 대상 |
|---|---|---|---|
| `main` | 제출·배포할 수 있는 버전 | - | - |
| `develop` | 다음 버전을 개발하는 브랜치 | `main` | `main` (release를 거쳐서) |
| `feature/*` | 기능 개발 | `develop` | `develop` |
| `release-*` | 제출 전 점검과 버그 수정 | `develop` | `main`, `develop` |
| `hotfix-*` | `main`의 긴급 수정 | `main` | `main`, `develop` |

`main`과 `develop`에는 직접 push할 수 없습니다. PR을 만들고 다른 팀원 1명 이상의 승인을 받아야 병합됩니다.

## 브랜치 이름

- 기능: `feature/이슈번호-기능요약` (예: `feature/12-rule-similarity`)
- 릴리스: `release-버전` (예: `release-1.0`)
- 긴급 수정: `hotfix-버전` (예: `hotfix-1.0.1`)

## 작업 순서

1. 이슈 템플릿으로 이슈를 만듭니다.
2. 최신 `develop`에서 브랜치를 만듭니다.
3. 작업 후 내 브랜치에 push합니다.
4. `develop`으로 PR을 만들고 PR 템플릿을 작성합니다.
5. 다른 팀원의 승인을 받아 병합하고, 작업 브랜치를 삭제합니다.

```bash
git switch develop
git pull origin develop
git switch -c feature/12-rule-similarity
git add .
git commit -m "feat: 규칙 찾기 단계별 유사도 계산"
git push -u origin feature/12-rule-similarity
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

레포가 공개 상태이므로 API 키, 토큰, 비공개 데이터는 올리지 않습니다.
