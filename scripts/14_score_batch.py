"""5-2: 배치 채점 — 과거(구간 0~3)로 학습한 모델이 '새로 들어온 거래'(구간 4)를 시간 순서대로 채점해 DB에 저장한다.

사용법: python scripts/14_score_batch.py --db security_db
        (실습 DB에서 시험해 보려면 --db fds_lab)

역할을 접속 계정으로 나눈다 (최소권한)
  - 학습·평가 담당 = 소유자 계정(summer): 거래 + 사기 라벨을 읽어 모델을 학습하고, 채점이 끝난 뒤 성능을 평가
  - 채점 담당     = svc_fds_scorer: 라벨은 읽을 수 없음. 거래를 읽고 실행 기록·점수만 쓸 수 있음
흐름
  1) [학습] 구간 0~3 거래 + 라벨로 LightGBM 학습 (2-2 최종 설정 LGBM_PARAMS)
  2) [기준점] 구간 0~3 안에서 4겹 교차검증 점수로 '재현율 80%' 기준점을 고름 (2-4와 같은 방법, 채점 구간은 안 봄)
  3) [채점] svc_fds_scorer로 접속 → model_runs에 실행 기록 → 구간 4를 시간 순서대로 5,000건씩 읽어
           점수 + 경보 여부 + 판단 근거(SHAP 상위 3개, 사기 쪽으로 민 특성)를 scores에 저장 (묶음마다 COMMIT)
  4) [평가] 소유자 계정으로 점수와 라벨을 조인해 성능 확인 → 4-1 시간순 결과(구간 4, PR-AUC 0.8065)와 같아야 함
LightGBM만 쓴다 (PyTorch와 함께 불러오면 libomp 충돌)
"""
import argparse
import importlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from lightgbm import LGBMClassifier
from psycopg.types.json import Jsonb

from fds_common import LGBM_PARAMS, evaluate

thr_mod = importlib.import_module("06_threshold")      # 2-4의 기준점 함수 재사용

TRAIN_BLOCKS = [0, 1, 2, 3]
SCORE_BLOCKS = [4]
RECALL_TARGET = 0.80
BATCH = 5000          # 한 번에 채점하는 거래 수 (실시간 대신 짧은 주기의 묶음 처리를 흉내)
N_REASONS = 3

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]     # 학습 때와 같은 열 순서
SELECT_FEATURES = ", ".join(['time_sec AS "Time"'] + [f'v{i} AS "V{i}"' for i in range(1, 29)] + ['amount::float8 AS "Amount"'])


def code_version():
    """지금 코드의 git 커밋. 커밋 안 된 변경이 있으면 -dirty를 붙인다."""
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True,
                                        stderr=subprocess.DEVNULL).strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def load_train(conn):
    """[학습 담당] 학습 구간의 거래 + 라벨"""
    rows = conn.execute(f"""
        SELECT t.txn_id, {SELECT_FEATURES}, l.is_fraud
        FROM fds.transactions t JOIN fds.fraud_labels l USING (txn_id)
        WHERE t.time_block = ANY(%s)
        ORDER BY t.txn_id""", (TRAIN_BLOCKS,)).fetchall()
    df = pd.DataFrame(rows, columns=["txn_id"] + FEATURES + ["is_fraud"])
    return df[FEATURES], df["is_fraud"].astype(int).to_numpy()


def reasons_for(contrib_row, x_row):
    """SHAP 값이 가장 큰(사기 쪽으로 가장 세게 민) 특성 3개"""
    top = np.argsort(-contrib_row[:-1])[:N_REASONS]           # 마지막 열은 기준값
    return [{"feature": FEATURES[j], "value": round(float(x_row[j]), 4), "shap": round(float(contrib_row[j]), 4)}
            for j in top]


def score_stream(conn, model, run_id, threshold):
    """[채점 담당] 구간 4를 시간 순서대로 BATCH건씩 읽어 채점하고 저장"""
    n_total = n_alert = 0
    with conn.cursor(name="stream", withhold=True) as cur:    # 서버 쪽 커서: BATCH건씩만 받아 옴, withhold = COMMIT 뒤에도 유지
        cur.itersize = BATCH
        cur.execute(f"""
            SELECT txn_id, {SELECT_FEATURES}
            FROM fds.transactions WHERE time_block = ANY(%s)
            ORDER BY time_sec, txn_id""", (SCORE_BLOCKS,))
        batch_no = 0
        while True:
            rows = cur.fetchmany(BATCH)
            if not rows:
                break
            ids = [r[0] for r in rows]
            X = pd.DataFrame([r[1:] for r in rows], columns=FEATURES)
            score = model.predict_proba(X)[:, 1]
            contrib = model.predict(X, pred_contrib=True)
            alert = score >= threshold
            with conn.cursor() as w:                          # 이름 없는 커서로 쓰기 (같은 트랜잭션)
                with w.copy("COPY fds.scores (run_id, txn_id, score, is_alert, reasons) FROM STDIN") as cp:
                    for i in range(len(ids)):
                        cp.write_row((run_id, ids[i], float(score[i]), bool(alert[i]),
                                      Jsonb(reasons_for(contrib[i], X.iloc[i].to_numpy()))))
            conn.commit()                                     # 묶음마다 확정 → 중간에 멈춰도 앞 묶음은 남음
            n_total += len(ids)
            n_alert += int(alert.sum())
            batch_no += 1
            print(f"  묶음 {batch_no:>2}: {len(ids):,}건 채점 | 경보 {int(alert.sum())}건 (누적 {n_total:,} / 경보 {n_alert})", flush=True)
    return n_total, n_alert


def evaluate_run(conn, run_id, threshold):
    """[평가 담당] 점수와 라벨을 조인해 성능 확인"""
    rows = conn.execute("""
        SELECT s.score, l.is_fraud
        FROM fds.scores s JOIN fds.fraud_labels l USING (txn_id)
        WHERE s.run_id = %s""", (run_id,)).fetchall()
    score = np.array([r[0] for r in rows])
    y = np.array([int(r[1]) for r in rows])
    alert = score >= threshold
    tp, fp = int((alert & (y == 1)).sum()), int((alert & (y == 0)).sum())
    return {"n": len(y), "fraud": int(y.sum()), "alerts": int(alert.sum()), "true_alerts": tp, "false_alerts": fp,
            "recall": round(tp / max(1, y.sum()), 4), "precision": round(tp / max(1, alert.sum()), 4),
            **evaluate(y, score)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--scorer-user", default="svc_fds_scorer")
    args = ap.parse_args()

    # 1) 학습 + 2) 기준점 — 소유자 계정
    with psycopg.connect(dbname=args.db) as owner:
        X_tr, y_tr = load_train(owner)
    print(f"[학습] 구간 {TRAIN_BLOCKS}: 거래 {len(y_tr):,}건 / 사기 {y_tr.sum()}건")
    t0 = time.perf_counter()
    oof = thr_mod.inner_oof_scores(X_tr, y_tr)
    threshold = thr_mod.threshold_for_recall(y_tr, oof, RECALL_TARGET)
    model = LGBMClassifier(**LGBM_PARAMS).fit(X_tr, y_tr)
    print(f"[기준점] 학습 구간 안쪽 4겹 교차검증, 재현율 {RECALL_TARGET:.0%} → {threshold:.4f} ({time.perf_counter() - t0:.0f}초)")

    # 3) 채점 — 최소권한 채점 계정
    version = code_version()
    with psycopg.connect(dbname=args.db, user=args.scorer_user) as scorer:
        run_id = scorer.execute("""
            INSERT INTO fds.model_runs (model_name, params, train_blocks, score_blocks, threshold, threshold_rule, code_version)
            VALUES ('lightgbm', %s, %s, %s, %s, %s, %s) RETURNING run_id""",
            (Jsonb(LGBM_PARAMS), TRAIN_BLOCKS, SCORE_BLOCKS, threshold,
             f"학습 구간 안쪽 4겹 교차검증(StratifiedKFold rs=0) 점수로 재현율 {RECALL_TARGET:.0%} 달성하는 가장 높은 기준점",
             version)).fetchone()[0]
        scorer.commit()
        print(f"[채점] {args.scorer_user}로 접속, 실행 번호 {run_id} (코드 {version})")
        t0 = time.perf_counter()
        n_total, n_alert = score_stream(scorer, model, run_id, threshold)
        scorer.execute("UPDATE fds.model_runs SET finished_at = now() WHERE run_id = %s", (run_id,))
        scorer.commit()
        sec = time.perf_counter() - t0
    print(f"[채점] 완료: {n_total:,}건 / 경보 {n_alert}건 ({sec:.1f}초)")

    # 4) 평가 — 소유자 계정
    with psycopg.connect(dbname=args.db) as owner:
        ev = evaluate_run(owner, run_id, threshold)
    print(f"[평가] 사기 {ev['fraud']}건 중 {ev['true_alerts']}건 경보(재현율 {ev['recall']:.1%}) | "
          f"경보 {ev['alerts']}건 중 진짜 사기 {ev['true_alerts']}건(정밀도 {ev['precision']:.1%}) | 오경보 {ev['false_alerts']}건 | "
          f"PR-AUC {ev['pr_auc']:.4f} (4-1 시간순 구간 4: 0.8065)")

    out = Path("reports")
    result = {"db": args.db, "run_id": run_id, "code_version": version, "train_blocks": TRAIN_BLOCKS,
              "score_blocks": SCORE_BLOCKS, "threshold": threshold, "batch": BATCH,
              "scoring_seconds": round(sec, 1), "eval": ev}
    (out / f"score_batch_{args.db}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved reports/score_batch_{args.db}.json")


if __name__ == "__main__":
    main()
