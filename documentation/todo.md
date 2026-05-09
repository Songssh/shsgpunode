# TODO - SHS GPU Worker Node

## 1. 업로드 파일 자동 정리 기능 추가

### 배경

현재 Worker Node는 multipart 작업 제출 시 업로드된 입력 파일을 `data/uploads/`에 저장한다.

예:

```text
data/uploads/{uuid}_{filename}

Whisper 작업의 경우 업로드된 오디오/영상 파일을 읽어서 처리하고, 결과 파일은 다음 위치에 저장된다.

data/tasks/{task_id}/output/

하지만 현재 구조에서는 작업이 끝난 뒤에도 data/uploads/에 저장된 원본 업로드 파일이 자동으로 삭제되지 않는다.

따라서 Whisper 같은 파일 기반 작업을 많이 실행하면 data/uploads/와 data/tasks/ 디렉토리가 계속 커질 수 있다.

2. 목표

Worker Node에 오래된 업로드 파일과 작업 디렉토리를 정리하는 cleanup 기능을 추가한다.

목표는 다음과 같다.

오래된 업로드 원본 파일 자동 삭제
오래된 완료/실패/취소 작업 디렉토리 자동 삭제
실행 중인 작업 파일은 절대 삭제하지 않음
Central Server artifact 다운로드 흐름을 깨뜨리지 않음
보관 기간은 .env에서 조정 가능하게 함
3. 추가할 환경변수

.env.example과 app/config.py에 아래 설정을 추가한다.

# ------------------------------------------------------------
# Cleanup / Retention
# ------------------------------------------------------------
CLEANUP_ENABLED=true

# data/uploads/ 안의 오래된 업로드 원본 파일 보관 시간
UPLOAD_RETENTION_HOURS=24

# data/tasks/ 안의 오래된 task workdir 보관 시간
TASK_RETENTION_HOURS=168

# cleanup loop 실행 주기
CLEANUP_INTERVAL_SECONDS=3600

권장 기본값:

UPLOAD_RETENTION_HOURS=24
TASK_RETENTION_HOURS=168
CLEANUP_INTERVAL_SECONDS=3600

의미:

업로드 원본 파일은 24시간 후 삭제
완료된 작업 디렉토리는 7일 후 삭제
cleanup은 1시간마다 실행
4. 구현 파일

새 파일 추가:

app/core/cleanup.py

역할:

data/uploads/ 오래된 파일 삭제
data/tasks/ 오래된 task 디렉토리 삭제
삭제 대상 검증
cleanup loop 제공
로그 출력
5. 구현해야 할 함수 예시
cleanup_old_uploads()
cleanup_old_task_dirs()
cleanup_once()
cleanup_loop()

각 함수 역할:

cleanup_old_uploads()
- data/uploads/ 내부 파일 중 오래된 파일 삭제
- 디렉토리는 삭제하지 않음
- 현재 시간 기준 UPLOAD_RETENTION_HOURS보다 오래된 파일만 삭제

cleanup_old_task_dirs()
- data/tasks/{task_id}/ 디렉토리 중 오래된 항목 삭제
- TASK_RETENTION_HOURS보다 오래된 task 디렉토리만 삭제
- running/queued/pending task는 삭제하지 않도록 주의

cleanup_once()
- uploads cleanup과 task cleanup을 한 번 실행

cleanup_loop()
- CLEANUP_INTERVAL_SECONDS마다 cleanup_once() 반복 실행
6. app/main.py 수정

서버 시작 시 CLEANUP_ENABLED=true이면 cleanup background task를 시작한다.

예상 흐름:

FastAPI lifespan 시작
↓
queue worker 시작
↓
central heartbeat loop 시작
↓
cleanup loop 시작
↓
서버 종료 시 cleanup loop cancel
7. 삭제 안전 규칙

cleanup 기능은 반드시 아래 규칙을 지켜야 한다.

1. data/uploads/ 외부 파일 삭제 금지
2. data/tasks/ 외부 디렉토리 삭제 금지
3. 절대 경로 직접 삭제 금지
4. path resolve 후 기준 디렉토리 내부인지 확인
5. 실행 중인 task의 work_dir 삭제 금지
6. queued/pending/running 상태 task 삭제 금지
7. completed/failed/cancelled 상태 task만 삭제 가능
8. task 상태를 모르면 삭제하지 않음
8. 주의할 점

현재 Worker Task 정보는 메모리 기반이다.

Worker가 재시작되면 기존 task 상태를 모를 수 있다.

따라서 data/tasks/{task_id} 디렉토리만 보고 삭제할 경우, 너무 짧은 보관 시간은 위험할 수 있다.

MVP에서는 다음 정책을 사용한다.

data/uploads/
- 파일 수정 시간이 UPLOAD_RETENTION_HOURS보다 오래되면 삭제

data/tasks/
- 디렉토리 수정 시간이 TASK_RETENTION_HOURS보다 오래되면 삭제
- 단, 현재 메모리에 존재하는 pending/queued/running task의 work_dir는 삭제하지 않음
9. Artifact 다운로드와의 관계

Central Server는 Worker의 artifact 파일을 proxy한다.

외부 클라이언트 호출:

GET /api/tasks/{central_task_id}/artifacts?path=output/result.txt

Central Server 내부 호출:

GET /api/worker/tasks/{worker_task_id}/artifacts?path=output/result.txt

cleanup이 data/tasks/{task_id}/output/을 삭제하면 이후 artifact 다운로드는 불가능해진다.

따라서 task result 보관 기간은 너무 짧게 잡지 않는다.

권장:

TASK_RETENTION_HOURS=168

즉, 완료된 작업 결과는 기본 7일 보관한다.

10. 테스트 항목
uploads cleanup 테스트
1. data/uploads/에 오래된 테스트 파일 생성
2. cleanup 실행
3. 오래된 파일이 삭제되는지 확인
4. 최근 파일은 삭제되지 않는지 확인
tasks cleanup 테스트
1. data/tasks/{old_task_id}/output/result.txt 생성
2. 오래된 task 디렉토리로 mtime 조정
3. cleanup 실행
4. 오래된 task 디렉토리가 삭제되는지 확인
5. 최근 task 디렉토리는 삭제되지 않는지 확인
안전 테스트
1. data/uploads/ 외부 파일이 삭제되지 않는지 확인
2. data/tasks/ 외부 디렉토리가 삭제되지 않는지 확인
3. running task work_dir가 삭제되지 않는지 확인
4. queued task work_dir가 삭제되지 않는지 확인
11. 완료 기준

이 TODO는 아래 조건을 만족하면 완료로 본다.

- app/core/cleanup.py 추가됨
- .env.example에 cleanup 설정 추가됨
- app/config.py에 cleanup 설정 추가됨
- app/main.py lifespan에서 cleanup loop 시작/종료 처리됨
- 오래된 data/uploads 파일 자동 삭제 가능
- 오래된 data/tasks 작업 디렉토리 자동 삭제 가능
- pending/queued/running task는 삭제하지 않음
- cleanup 로그 확인 가능
- 수동 cleanup_once() 테스트 가능
12. 향후 개선

MVP 이후에는 다음 기능을 추가할 수 있다.

- task metadata를 파일 또는 DB에 영구 저장
- completed_at 기준으로 task cleanup
- 관리자 API로 cleanup 수동 실행
- 관리자 API로 디스크 사용량 조회
- cleanup dry-run 모드
- task별 보관 기간 설정
- 중요한 작업 결과 pin 기능
- Central Server에서 결과 보관 정책 관리