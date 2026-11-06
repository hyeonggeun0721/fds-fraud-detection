-- =============================================================================
-- F001: FDS 전용 스키마 `fds` (5-1)
--
-- 실행:  psql -d security_db_fds -v ON_ERROR_STOP=1 -f db/F001_fds_schema.sql
-- 되돌리기: psql -d security_db_fds -v ON_ERROR_STOP=1 -f db/F000_drop_fds.sql
--
-- 원칙 (2026-11-06 결정, 방식 A)
--   - 보안 DB 과제(public 스키마, V001~V037)는 한 글자도 건드리지 않는다
--   - 2026-11-06부터 대상 DB는 과제 원본(security_db)이 아니라 복제본 security_db_fds (원본 백업에 FDS 데이터가 섞이지 않게)
--   - FDS 객체는 전부 `fds` 스키마 안에만 만든다 → 스키마째 지우면 흔적 없이 원상복구
--   - 보안 이벤트·알림과의 연결(V038)은 과제 제출 뒤에 따로 한다
--   - 전체를 한 트랜잭션으로 실행 → 중간에 하나라도 실패하면 아무것도 안 만들어짐
-- =============================================================================
BEGIN;

CREATE SCHEMA fds;
COMMENT ON SCHEMA fds IS 'FDS(이상거래탐지) 포트폴리오 프로젝트 — ~/fds 저장소에서 관리. 보안 DB 과제(public)와 분리';

-- -----------------------------------------------------------------------------
-- 1) 거래: 모델 입력만 담는다. 정답(사기 여부)은 여기 없다 → 아래 fraud_labels
-- -----------------------------------------------------------------------------
CREATE TABLE fds.transactions (
    txn_id      integer PRIMARY KEY,               -- fds_common.load_data(중복 제거) 후 행 번호 → 파이썬 결과와 1:1 대응
    time_sec    integer NOT NULL CHECK (time_sec BETWEEN 0 AND 172800),   -- 첫 거래로부터 지난 초 (이틀 = 172,800초)
    time_block  smallint NOT NULL CHECK (time_block BETWEEN 0 AND 4),     -- 4-1의 시간 구간 번호
    amount      numeric(10, 2) NOT NULL CHECK (amount >= 0),
    v1  double precision NOT NULL, v2  double precision NOT NULL, v3  double precision NOT NULL, v4  double precision NOT NULL,
    v5  double precision NOT NULL, v6  double precision NOT NULL, v7  double precision NOT NULL, v8  double precision NOT NULL,
    v9  double precision NOT NULL, v10 double precision NOT NULL, v11 double precision NOT NULL, v12 double precision NOT NULL,
    v13 double precision NOT NULL, v14 double precision NOT NULL, v15 double precision NOT NULL, v16 double precision NOT NULL,
    v17 double precision NOT NULL, v18 double precision NOT NULL, v19 double precision NOT NULL, v20 double precision NOT NULL,
    v21 double precision NOT NULL, v22 double precision NOT NULL, v23 double precision NOT NULL, v24 double precision NOT NULL,
    v25 double precision NOT NULL, v26 double precision NOT NULL, v27 double precision NOT NULL, v28 double precision NOT NULL,
    loaded_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX transactions_block_time_idx ON fds.transactions (time_block, time_sec);   -- "구간 4를 시간 순서대로" 조회용
COMMENT ON TABLE fds.transactions IS 'Kaggle creditcard(중복 제거 283,726건). 모델 입력만 저장, 사기 라벨은 fds.fraud_labels';

-- -----------------------------------------------------------------------------
-- 2) 사기 라벨: 현실에서는 거래 후 며칠~몇 주 뒤에 신고·조사로 확정된다 (3-3a)
--    → 거래와 분리하고, 채점 계정은 이 테이블을 아예 읽지 못하게 한다(아래 권한)
-- -----------------------------------------------------------------------------
CREATE TABLE fds.fraud_labels (
    txn_id      integer PRIMARY KEY REFERENCES fds.transactions (txn_id) ON DELETE CASCADE,
    is_fraud    boolean NOT NULL,
    source      text NOT NULL DEFAULT 'kaggle' CHECK (source IN ('kaggle', 'manual')),
    labeled_at  timestamptz NOT NULL DEFAULT now()
);
COMMENT ON TABLE fds.fraud_labels IS '사후 확정 라벨. 채점 계정(svc_fds_scorer)은 접근 불가 = 라벨 누수 구조적 차단';

-- -----------------------------------------------------------------------------
-- 3) 모델 실행 기록: "이 점수는 어떤 모델·설정·기준점으로 나왔나"를 남긴다
-- -----------------------------------------------------------------------------
CREATE TABLE fds.model_runs (
    run_id          serial PRIMARY KEY,
    model_name      text NOT NULL,                          -- 예: 'lightgbm'
    params          jsonb NOT NULL,                         -- LGBM_PARAMS 그대로
    train_blocks    smallint[] NOT NULL,                    -- 학습에 쓴 시간 구간 (예: {0,1,2,3})
    score_blocks    smallint[] NOT NULL,                    -- 채점할 시간 구간 (예: {4})
    threshold       double precision NOT NULL CHECK (threshold > 0 AND threshold < 1),
    threshold_rule  text NOT NULL,                          -- 기준점을 어떻게 정했나 (예: '학습 구간 내부 교차검증, 재현율 80%')
    code_version    text NOT NULL,                          -- FDS 저장소 git 커밋 → 재현 가능
    started_at      timestamptz NOT NULL DEFAULT now(),
    finished_at     timestamptz,
    CHECK (NOT (train_blocks && score_blocks))              -- 학습 구간과 채점 구간이 겹치면 거부 (누수 방지)
);
COMMENT ON TABLE fds.model_runs IS '채점 1회 = 1행. 점수의 출처(모델·설정·기준점·코드 버전) 기록';

-- -----------------------------------------------------------------------------
-- 4) 점수: 거래별 점수 + 경보 여부 + 판단 근거(SHAP 상위 3개, 4-2)
-- -----------------------------------------------------------------------------
CREATE TABLE fds.scores (
    run_id      integer NOT NULL REFERENCES fds.model_runs (run_id) ON DELETE CASCADE,
    txn_id      integer NOT NULL REFERENCES fds.transactions (txn_id) ON DELETE CASCADE,
    score       double precision NOT NULL CHECK (score >= 0 AND score <= 1),
    is_alert    boolean NOT NULL,                           -- score >= 그 실행의 기준점
    reasons     jsonb NOT NULL CHECK (jsonb_typeof(reasons) = 'array'),   -- [{"feature":"V14","value":-8.2,"shap":8.5}, ...]
    scored_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, txn_id)                            -- 같은 실행에서 같은 거래를 두 번 채점할 수 없음
);
CREATE INDEX scores_alert_idx ON fds.scores (run_id, score DESC) WHERE is_alert;   -- "경보 목록을 점수 높은 순" 조회용
COMMENT ON TABLE fds.scores IS '채점 결과. 한 번 쓰면 수정·삭제 불가(채점 계정 권한)';

-- -----------------------------------------------------------------------------
-- 5) 권한: 최소권한 채점 계정 (보안 DB 과제의 svc_event_ingest와 같은 원칙)
--    비밀번호는 여기 쓰지 않는다 → 실행 후 \password svc_fds_scorer 로 따로 설정
-- -----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'svc_fds_scorer') THEN
        CREATE ROLE svc_fds_scorer LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT CONNECTION LIMIT 3;
    END IF;
END $$;

REVOKE ALL ON SCHEMA fds FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA fds FROM PUBLIC;
GRANT USAGE ON SCHEMA fds TO svc_fds_scorer;
GRANT SELECT ON fds.transactions TO svc_fds_scorer;                     -- 입력 읽기
GRANT SELECT, INSERT ON fds.model_runs TO svc_fds_scorer;               -- 실행 기록 남기기
GRANT UPDATE (finished_at) ON fds.model_runs TO svc_fds_scorer;         -- 끝난 시각만 고칠 수 있음
GRANT USAGE ON SEQUENCE fds.model_runs_run_id_seq TO svc_fds_scorer;
GRANT INSERT ON fds.scores TO svc_fds_scorer;                           -- 점수 쓰기만 (읽기·수정·삭제 없음)
-- fds.fraud_labels 에는 아무 권한도 주지 않는다

-- 대시보드용 읽기 전용 role(보안 DB 과제에 이미 있는 security_manager)이 있으면 읽기 권한
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'security_manager') THEN
        GRANT USAGE ON SCHEMA fds TO security_manager;
        GRANT SELECT ON ALL TABLES IN SCHEMA fds TO security_manager;
    END IF;
END $$;

-- -----------------------------------------------------------------------------
-- 6) 이 스키마의 변경 이력 (보안 DB 과제의 schema_migrations와 섞지 않음)
-- -----------------------------------------------------------------------------
CREATE TABLE fds.schema_history (
    version     text PRIMARY KEY,
    description text NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now()
);
INSERT INTO fds.schema_history (version, description)
VALUES ('F001', 'fds 스키마: transactions, fraud_labels, model_runs, scores, svc_fds_scorer 권한');

COMMIT;
