-- =============================================================================
-- F002: 운영 뷰 + 기준점 모니터링 (5-2b)
--
-- 실행:  psql -d security_db_fds -v ON_ERROR_STOP=1 -f db/F002_fds_monitoring.sql
--
-- 만드는 것
--   1) fds.vw_alert_queue        분석가용 경보 대기열 (라벨 없음 — 운영 화면)
--   2) fds.vw_run_performance    실행별·시간대별 성능 (라벨 사용 — 확정된 라벨로 사후 평가)
--   3) fds.threshold_check(목표)  "목표 재현율을 맞추려면 기준점이 얼마였어야 했나"
--
-- 권한 원칙 (F001과 같음)
--   - 라벨을 쓰는 2), 3)은 security_invoker / SECURITY INVOKER → 조회하는 사람의 권한으로 라벨을 읽음
--     (뷰 소유자 권한으로 읽으면, 뷰만 열어 줘도 채점 계정이 라벨을 우회해서 보게 됨)
--   - 채점 계정(svc_fds_scorer)에는 아무것도 열어 주지 않음. 읽기 전용 security_manager에만 SELECT/EXECUTE
-- =============================================================================
BEGIN;

-- 1) 경보 대기열 -------------------------------------------------------------
--    점수 높은 순 순번 + 판단 근거를 사람이 읽기 쉬운 한 줄로 ("V14=-8.20(+8.51), V4=...")
CREATE VIEW fds.vw_alert_queue AS
SELECT
    s.run_id,
    row_number() OVER (PARTITION BY s.run_id ORDER BY s.score DESC) AS queue_rank,   -- 실행 안에서 몇 번째로 의심스러운가
    s.txn_id,
    round(s.score::numeric, 4)                                       AS score,
    t.time_sec,
    t.time_sec / 3600                                                AS hour_since_start,
    t.amount,
    (SELECT string_agg(format('%s=%s(%s)', r->>'feature',
                              round((r->>'value')::numeric, 2),
                              to_char((r->>'shap')::numeric, 'FMS990.00')), ', ' ORDER BY ord)
       FROM jsonb_array_elements(s.reasons) WITH ORDINALITY AS e(r, ord)) AS reasons_text,
    s.scored_at
FROM fds.scores s
JOIN fds.transactions t USING (txn_id)
WHERE s.is_alert;
COMMENT ON VIEW fds.vw_alert_queue IS '분석가 경보 대기열: 실행별 점수 순, 판단 근거(SHAP 상위 3개) 한 줄 요약. 라벨 없음';

-- 2) 실행별·시간대별 성능 ------------------------------------------------------
--    GROUP BY ROLLUP: 시간대별 행 + 맨 끝에 실행 전체 합계 행(hour_since_start가 NULL)
CREATE VIEW fds.vw_run_performance WITH (security_invoker = true) AS
SELECT
    s.run_id,
    t.time_sec / 3600                                                        AS hour_since_start,   -- NULL = 실행 전체
    count(*)                                                                 AS scored,
    count(*) FILTER (WHERE l.is_fraud)                                       AS frauds,
    count(*) FILTER (WHERE s.is_alert)                                       AS alerts,
    count(*) FILTER (WHERE s.is_alert AND l.is_fraud)                        AS true_alerts,
    count(*) FILTER (WHERE s.is_alert AND NOT l.is_fraud)                    AS false_alerts,
    round(count(*) FILTER (WHERE s.is_alert AND l.is_fraud)::numeric
          / NULLIF(count(*) FILTER (WHERE l.is_fraud), 0), 3)                AS recall,       -- 사기 중 잡은 비율
    round(count(*) FILTER (WHERE s.is_alert AND l.is_fraud)::numeric
          / NULLIF(count(*) FILTER (WHERE s.is_alert), 0), 3)                AS precision    -- 경보 중 진짜 사기 비율
FROM fds.scores s
JOIN fds.transactions t USING (txn_id)
JOIN fds.fraud_labels l USING (txn_id)            -- 라벨이 확정된 거래만 평가 (아직 라벨 없는 거래는 빠짐)
GROUP BY s.run_id, ROLLUP (t.time_sec / 3600);
COMMENT ON VIEW fds.vw_run_performance IS '실행별·시간대별 재현율/정밀도. hour_since_start NULL = 실행 전체. 라벨 권한 있는 사람만 결과가 나옴';

-- 3) 기준점 점검 함수 ----------------------------------------------------------
--    "확정된 라벨로 보니, 목표 재현율을 맞추려면 기준점이 얼마였어야 했고, 그때 경보·오경보는 몇 건인가"
CREATE FUNCTION fds.threshold_check(target_recall numeric DEFAULT 0.80)
RETURNS TABLE (
    run_id               integer,
    target               numeric,
    current_threshold    numeric,
    current_recall       numeric,
    current_alerts       bigint,
    current_false_alerts bigint,
    needed_threshold     numeric,
    needed_alerts        bigint,
    needed_false_alerts  bigint
)
LANGUAGE sql STABLE
SECURITY INVOKER                       -- 호출한 사람 권한으로 실행 (라벨 권한 없으면 실패)
SET search_path = fds, pg_temp         -- 검색 경로 고정 (보안 DB 과제 V021과 같은 원칙)
AS $$
    WITH s AS (
        SELECT sc.run_id, sc.score, sc.is_alert, l.is_fraud
        FROM fds.scores sc JOIN fds.fraud_labels l USING (txn_id)
    ),
    f AS (                              -- 사기 거래만 점수 높은 순으로 줄 세우기
        SELECT s.run_id, s.score,
               row_number() OVER (PARTITION BY s.run_id ORDER BY s.score DESC) AS rk,
               count(*)     OVER (PARTITION BY s.run_id)                       AS n_fraud
        FROM s WHERE s.is_fraud
    ),
    need AS (                           -- 목표를 맞추려면 위에서 ceil(목표 × 사기 수)번째 사기까지 잡아야 함
        SELECT f.run_id, f.score AS thr
        FROM f WHERE f.rk = ceil(target_recall * f.n_fraud)
    )
    SELECT
        r.run_id,
        target_recall,
        round(r.threshold::numeric, 4),
        round((count(*) FILTER (WHERE s.is_alert AND s.is_fraud))::numeric
              / NULLIF(count(*) FILTER (WHERE s.is_fraud), 0), 3),
        count(*) FILTER (WHERE s.is_alert),
        count(*) FILTER (WHERE s.is_alert AND NOT s.is_fraud),
        round(n.thr::numeric, 4),
        count(*) FILTER (WHERE s.score >= n.thr),
        count(*) FILTER (WHERE s.score >= n.thr AND NOT s.is_fraud)
    FROM fds.model_runs r
    JOIN s    ON s.run_id = r.run_id
    JOIN need n ON n.run_id = r.run_id
    GROUP BY r.run_id, r.threshold, n.thr
    ORDER BY r.run_id;
$$;
COMMENT ON FUNCTION fds.threshold_check(numeric) IS '확정 라벨 기준: 현재 기준점의 재현율·오경보 vs 목표 재현율에 필요한 기준점·경보·오경보';

-- 권한 ------------------------------------------------------------------------
REVOKE ALL ON fds.vw_alert_queue, fds.vw_run_performance FROM PUBLIC;
REVOKE ALL ON FUNCTION fds.threshold_check(numeric) FROM PUBLIC;     -- 함수는 기본이 '누구나 실행'이라 회수
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'security_manager') THEN
        GRANT SELECT ON fds.vw_alert_queue, fds.vw_run_performance TO security_manager;
        GRANT EXECUTE ON FUNCTION fds.threshold_check(numeric) TO security_manager;
    END IF;
END $$;

INSERT INTO fds.schema_history (version, description)
VALUES ('F002', '운영 뷰: vw_alert_queue, vw_run_performance(security_invoker), threshold_check()');

COMMIT;
