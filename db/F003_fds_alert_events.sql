-- =============================================================================
-- F003: FDS 경보 → 보안 이벤트 자동 생성 트리거 (5-3)
--
-- 실행:  psql -d security_db_fds -v ON_ERROR_STOP=1 -f db/F003_fds_alert_events.sql
-- 선행: 과제 저장소 fds-integration 브랜치의 V038 (FRAUD_SUSPECT 유형, 결제 자산, svc_fds_scorer 권한·RLS)
--
-- 동작
--   fds.scores 에 경보(is_alert = true) 행이 들어오는 순간, 같은 트랜잭션 안에서
--     ① public.security_events 에 FRAUD_SUSPECT 이벤트 1건 생성
--        - 점수 ≥ 0.99 → CRITICAL (기존 alert_poller 가 즉시 알림), 나머지 → HIGH
--        - 설명: '[FDS] run 2 · txn 242795 · 점수 0.9995 · V14=-14.27(+7.51), ...'
--     ② fds.alert_events 에 (실행, 거래) ↔ 이벤트 번호 연결 기록
--   → 점수만 저장되고 이벤트가 빠지는 일이 없음. 둘 중 하나라도 실패하면 점수 저장도 취소
--
-- 권한 원칙
--   - 트리거 함수는 SECURITY INVOKER: 채점 계정(svc_fds_scorer) 권한 그대로 실행
--     → V038 의 RLS(FRAUD_SUSPECT + [FDS] + 결제 자산)가 그대로 적용됨
--   - 채점 계정은 assets 를 읽을 수 없으므로, 결제 자산 번호는 적용 시점에 fds.settings 에 저장해 둔다
-- =============================================================================
BEGIN;

-- 0) 선행 조건 확인: V038 이 적용돼 있어야 함
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM public.event_types WHERE event_type_name = 'FRAUD_SUSPECT')
       OR NOT EXISTS (SELECT 1 FROM public.assets WHERE asset_name = '카드결제승인시스템') THEN
        RAISE EXCEPTION 'V038(fds-integration 브랜치)이 먼저 적용돼야 합니다';
    END IF;
END $$;

-- 1) 설정값 테이블: 결제 자산 번호, CRITICAL 기준 점수
CREATE TABLE fds.settings (
    key         text PRIMARY KEY,
    value       text NOT NULL,
    description text
);
INSERT INTO fds.settings (key, value, description)
SELECT 'payment_asset_id', asset_id::text, '보안 이벤트를 연결할 자산(public.assets) — V038 의 카드결제승인시스템'
FROM public.assets WHERE asset_name = '카드결제승인시스템';
INSERT INTO fds.settings (key, value, description)
VALUES ('critical_score', '0.99', '이 점수 이상 경보는 CRITICAL(즉시 알림), 미만은 HIGH');

-- 2) 연결 테이블: 어떤 실행의 어떤 거래가 몇 번 보안 이벤트가 됐나
CREATE TABLE fds.alert_events (
    run_id      integer NOT NULL,
    txn_id      integer NOT NULL,
    event_id    integer NOT NULL UNIQUE REFERENCES public.security_events (event_id),
    severity    varchar(10) NOT NULL CHECK (severity IN ('HIGH', 'CRITICAL')),
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, txn_id),                                        -- 경보 1건 = 이벤트 1건 (중복 생성 방지)
    FOREIGN KEY (run_id, txn_id) REFERENCES fds.scores (run_id, txn_id) ON DELETE CASCADE
);
COMMENT ON TABLE fds.alert_events IS 'FDS 경보 ↔ public.security_events 연결. 트리거 fds.fn_publish_alert 가 기록';

-- 3) 트리거 함수
CREATE FUNCTION fds.fn_publish_alert() RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = fds, pg_temp
AS $$
DECLARE
    v_asset    integer := (SELECT value::integer FROM fds.settings WHERE key = 'payment_asset_id');
    v_critical double precision := (SELECT value::double precision FROM fds.settings WHERE key = 'critical_score');
    v_event_id integer := nextval('public.security_events_event_id_seq');   -- RETURNING 대신 번호를 먼저 받음(채점 계정은 SELECT 권한 없음)
    v_severity varchar(10);
    v_reasons  text;
BEGIN
    v_severity := CASE WHEN NEW.score >= v_critical THEN 'CRITICAL' ELSE 'HIGH' END;

    SELECT string_agg(format('%s=%s(%s)', r->>'feature',
                             round((r->>'value')::numeric, 2),
                             to_char((r->>'shap')::numeric, 'FMS990.00')), ', ' ORDER BY ord)
      INTO v_reasons
      FROM jsonb_array_elements(NEW.reasons) WITH ORDINALITY AS e(r, ord);

    INSERT INTO public.security_events (event_id, asset_id, event_type, severity, status, detected_at, description)
    VALUES (v_event_id, v_asset, 'FRAUD_SUSPECT', v_severity, 'OPEN', NEW.scored_at,
            format('[FDS] run %s · txn %s · 점수 %s · %s',
                   NEW.run_id, NEW.txn_id, to_char(NEW.score, 'FM0.0000'), v_reasons));

    INSERT INTO fds.alert_events (run_id, txn_id, event_id, severity)
    VALUES (NEW.run_id, NEW.txn_id, v_event_id, v_severity);

    RETURN NULL;                                                         -- AFTER 트리거: 반환값 사용 안 함
END $$;
COMMENT ON FUNCTION fds.fn_publish_alert() IS '경보 점수 → security_events(FRAUD_SUSPECT) + fds.alert_events. SECURITY INVOKER';

-- 4) 트리거: 경보 행에만 동작 (WHEN 조건 → 경보 아닌 5만여 건은 함수 호출조차 안 함)
CREATE TRIGGER trg_scores_publish_alert
    AFTER INSERT ON fds.scores
    FOR EACH ROW
    WHEN (NEW.is_alert)
    EXECUTE FUNCTION fds.fn_publish_alert();

-- 5) 권한
REVOKE ALL ON fds.settings, fds.alert_events FROM PUBLIC;
REVOKE ALL ON FUNCTION fds.fn_publish_alert() FROM PUBLIC;
GRANT SELECT ON fds.settings TO svc_fds_scorer;                       -- 자산 번호·기준 점수 읽기
GRANT INSERT ON fds.alert_events TO svc_fds_scorer;                   -- 연결 기록 쓰기 (읽기·수정 없음)
GRANT EXECUTE ON FUNCTION fds.fn_publish_alert() TO svc_fds_scorer;
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'security_manager') THEN
        GRANT SELECT ON fds.settings, fds.alert_events TO security_manager;
    END IF;
END $$;

INSERT INTO fds.schema_history (version, description)
VALUES ('F003', '경보 → security_events(FRAUD_SUSPECT) 트리거, fds.alert_events, fds.settings');

COMMIT;
