"""5-1b: 거래·라벨을 PostgreSQL(fds 스키마)에 적재한다. + 실습 1회차: 적재 방법별 속도 비교

사용법:
  # 전체 적재 (COPY, 한 트랜잭션)
  python scripts/13_load_db.py data_raw/creditcard.csv --db fds_lab
  # 실습: 적재 방법 3가지 속도 비교 (임시 테이블에만 넣으므로 fds 테이블은 그대로)
  python scripts/13_load_db.py data_raw/creditcard.csv --db fds_lab --speed-test 20000

- 접속: 맥 로컬 PostgreSQL에 현재 사용자(summer)로 접속. 채점 계정은 transactions에 쓰기 권한이 없으므로 적재는 소유자가 한다
- txn_id = fds_common.load_data(중복 제거) 후 행 번호, time_block = 4-1과 같은 시간 구간 → 파이썬 실험 결과와 1:1 대응
- 이미 데이터가 있으면 적재하지 않고 멈춘다(두 번 실행해도 중복 없음)
- 적재 후 검산: 건수·사기 수·금액 합계를 파이썬 쪽 값과 비교
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import psycopg

from fds_common import load_data

time_blocks = importlib.import_module("11_time_split").time_blocks   # 4-1과 똑같은 구간 나누기

V_COLS = [f"v{i}" for i in range(1, 29)]
TXN_COLS = ["txn_id", "time_sec", "time_block", "amount"] + V_COLS


def build_rows(csv_path):
    """DB에 넣을 (거래 행 목록, 라벨 행 목록, 검산용 값)을 만든다."""
    X, y = load_data(csv_path)
    block = time_blocks(X["Time"].to_numpy())
    V = X[[f"V{i}" for i in range(1, 29)]].to_numpy()
    txn_rows = [
        (i, int(t), int(b), round(float(a), 2), *map(float, v))
        for i, (t, b, a, v) in enumerate(zip(X["Time"], block, X["Amount"], V))
    ]
    label_rows = [(i, bool(lbl)) for i, lbl in enumerate(y)]
    expected = {"n": len(y), "fraud": int(y.sum()), "amount_sum": round(float(np.round(X["Amount"], 2).sum()), 2)}
    return txn_rows, label_rows, expected


def copy_rows(cur, table, cols, rows):
    """COPY: 행을 한 줄씩 보내지만 SQL 문장은 1개 → 서버가 한 번에 받아 저장"""
    with cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN") as cp:
        for r in rows:
            cp.write_row(r)


def full_load(conn, txn_rows, label_rows, expected):
    with conn.transaction():                         # 전부 성공하거나, 하나라도 실패하면 전부 취소
        cur = conn.cursor()
        n_now = cur.execute("SELECT count(*) FROM fds.transactions").fetchone()[0]
        if n_now:
            print(f"이미 {n_now:,}건이 있어 적재하지 않습니다. 다시 넣으려면 F000 → F001로 초기화하세요.")
            return None
        t0 = time.perf_counter()
        copy_rows(cur, "fds.transactions", TXN_COLS, txn_rows)
        copy_rows(cur, "fds.fraud_labels", ["txn_id", "is_fraud"], label_rows)
        sec = time.perf_counter() - t0

        got = cur.execute("""
            SELECT count(*), (SELECT count(*) FROM fds.fraud_labels WHERE is_fraud), sum(amount)
            FROM fds.transactions""").fetchone()
        check = {"n": got[0], "fraud": got[1], "amount_sum": float(got[2])}
        ok = check["n"] == expected["n"] and check["fraud"] == expected["fraud"] \
            and abs(check["amount_sum"] - expected["amount_sum"]) < 0.01
        print(f"적재 {check['n']:,}건 / 사기 {check['fraud']}건 / 금액 합계 {check['amount_sum']:,.2f} ({sec:.1f}초)")
        print(f"검산(파이썬 값과 비교): {'일치' if ok else '불일치 → 취소'}  기대값 {expected}")
        if not ok:
            raise RuntimeError("검산 불일치 — 트랜잭션을 취소합니다")
    with conn.cursor() as cur:
        cur.execute("ANALYZE fds.transactions")      # 통계 갱신 → 실행 계획(EXPLAIN)이 정확해짐 (실습 3회차에서 사용)
        cur.execute("ANALYZE fds.fraud_labels")
    return {"rows": check["n"], "seconds": round(sec, 2)}


def speed_test(conn, txn_rows, n):
    """같은 n건을 세 가지 방법으로 임시 테이블에 넣고 시간을 잰다."""
    rows = txn_rows[:n]
    placeholders = ", ".join(["%s"] * len(TXN_COLS))
    insert_sql = f"INSERT INTO t ({', '.join(TXN_COLS)}) VALUES ({placeholders})"
    results = []

    def fresh_table(cur):
        cur.execute("DROP TABLE IF EXISTS t")
        cur.execute("CREATE TEMP TABLE t (LIKE fds.transactions INCLUDING ALL)")

    # ① 한 건씩 INSERT + 매번 COMMIT (autocommit: 문장 하나가 곧 트랜잭션 하나)
    conn.autocommit = True
    with conn.cursor() as cur:
        fresh_table(cur)
        t0 = time.perf_counter()
        for r in rows:
            cur.execute(insert_sql, r)
        results.append(("① INSERT 한 건씩 + 매번 COMMIT", time.perf_counter() - t0))
    conn.autocommit = False

    # ② 한 건씩 INSERT, COMMIT은 마지막에 한 번
    with conn.cursor() as cur:
        fresh_table(cur)
        conn.commit()
        t0 = time.perf_counter()
        for r in rows:
            cur.execute(insert_sql, r)
        conn.commit()
        results.append(("② INSERT 한 건씩 + COMMIT 한 번", time.perf_counter() - t0))

    # ③ COPY 한 번
    with conn.cursor() as cur:
        fresh_table(cur)
        conn.commit()
        t0 = time.perf_counter()
        copy_rows(cur, "t", TXN_COLS, rows)
        conn.commit()
        results.append(("③ COPY", time.perf_counter() - t0))
        cur.execute("DROP TABLE t")
        conn.commit()

    total = len(txn_rows)
    print(f"\n{n:,}건 적재 시간 비교 (임시 테이블, fds 테이블은 변경 없음)")
    out = []
    for name, sec in results:
        rate = n / sec
        out.append({"method": name, "rows": n, "seconds": round(sec, 3), "rows_per_sec": round(rate),
                    "estimated_full_seconds": round(total / rate, 1)})
        print(f"  {name:<28} {sec:7.2f}초 | 초당 {rate:>9,.0f}건 | 전체 {total:,}건 환산 {total / rate:7.1f}초")
    base = results[-1][1]
    for name, sec in results[:-1]:
        print(f"  {name} 는 COPY보다 {sec / base:.0f}배 느림")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--db", required=True, help="대상 DB 이름 (fds_lab 또는 security_db_fds)")
    ap.add_argument("--speed-test", type=int, default=0, metavar="N", help="실습: N건으로 적재 방법 3가지 속도 비교")
    args = ap.parse_args()

    txn_rows, label_rows, expected = build_rows(args.csv_path)
    print(f"준비: 거래 {expected['n']:,}건 / 사기 {expected['fraud']}건 → DB '{args.db}'")
    with psycopg.connect(dbname=args.db) as conn:
        if args.speed_test:
            res = {"db": args.db, "speed_test": speed_test(conn, txn_rows, args.speed_test)}
            name = "db_load_speed_results.json"
        else:
            r = full_load(conn, txn_rows, label_rows, expected)
            if r is None:
                return
            res = {"db": args.db, "full_load": r, "expected": expected}
            name = f"db_load_{args.db}.json"
    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / name).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"saved reports/{name}")


if __name__ == "__main__":
    sys.exit(main())
