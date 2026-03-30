# In-House Clustering PPT 작업 TO-DO LIST

## 현재 상태 (2026-03-30)

### 완료된 작업
- [x] `in-house-clustering` 프로젝트 구조 파악
- [x] PPT MCP 서버 설치: `office-powerpoint-mcp-server` (uvx, v2.0.7)
- [x] `~/.claude.json`에 MCP 서버 설정 추가

### 대기 중 (Claude Code 재시작 필요)
- [ ] Claude Code 재시작 후 `/mcp` 명령으로 `ppt` 서버 연결 확인
- [ ] PPT MCP 도구로 `output/clustering_methods_presentation.pptx` 수정

---

## 프로젝트 정보

| 항목 | 내용 |
|------|------|
| 프로젝트 경로 | `/Users/jongchan/Desktop/claude/in-house-clustering/` |
| PPT 파일 경로 | `/Users/jongchan/Desktop/claude/in-house-clustering/output/clustering_methods_presentation.pptx` |
| 슬라이드 수 | 11장 (타이틀 1 + 클러스터링 방법 10) |
| PPT 생성 스크립트 | `make_slides_1to5.py`, `make_slides_6to10.py`, `make_ppt.py` |

---

## MCP 서버 설정 정보

- **서버 이름:** `ppt`
- **패키지:** `office-powerpoint-mcp-server`
- **실행 방식:** `uvx --from office-powerpoint-mcp-server ppt_mcp_server`
- **설정 위치:** `~/.claude.json` → `mcpServers.ppt`
- **사용 가능 도구:** 34가지 (슬라이드 추가/수정/삭제, 텍스트/이미지/표/차트, 테마 등)

---

## 재시작 후 진행 순서

1. `/mcp` 명령 실행 → `ppt` 서버 상태 `connected` 확인
2. 사용자에게 PPT에서 **어떤 수정을 원하는지** 확인
3. PPT MCP 도구로 `clustering_methods_presentation.pptx` 수정 진행

---

## PPT 구성 (현재)

| 슬라이드 | 내용 |
|---------|------|
| 01 | 타이틀: "In-House Layout Feature Clustering" |
| 02 | Decision Tree Clustering |
| 03 | K-means MiniBatch |
| 04 | Autoencoder + K-means |
| 05 | Gaussian Mixture Model (GMM) |
| 06 | Bisecting K-means |
| 07 | Agglomerative Hierarchical (Ward) |
| 08 | HDBSCAN |
| 09 | Spectral Clustering |
| 10 | Isolation Forest + K-means |
| 11 | VAE + K-means |
