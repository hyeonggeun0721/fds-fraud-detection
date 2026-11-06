-- =============================================================================
-- F000: FDS 스키마 완전 제거 (되돌리기용)
--
-- 실행:  psql -d security_db_fds -v ON_ERROR_STOP=1 -f db/F000_drop_fds.sql
--
-- fds 스키마 안의 모든 테이블·데이터와 채점 계정을 지운다.
-- 보안 DB 과제(public 스키마)는 건드리지 않는다.
-- =============================================================================
BEGIN;

DROP SCHEMA IF EXISTS fds CASCADE;

-- 채점 계정(role)은 DB 하나가 아니라 PostgreSQL 전체에 하나뿐이다.
-- 다른 DB(예: 실습용 fds_lab과 security_db 둘 다)에 fds 스키마가 남아 있으면 role은 지우지 않고 남겨 둔다.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'svc_fds_scorer') THEN
        EXECUTE 'DROP OWNED BY svc_fds_scorer';      -- 이 DB 안의 권한 정리
        BEGIN
            DROP ROLE svc_fds_scorer;
        EXCEPTION WHEN dependent_objects_still_exist THEN
            RAISE NOTICE '다른 DB에 svc_fds_scorer 권한이 남아 있어 role은 유지 (그 DB에서도 F000을 실행하면 삭제됨)';
        END;
    END IF;
END $$;

COMMIT;
