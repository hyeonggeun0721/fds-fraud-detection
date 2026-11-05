"""2-4 보완: 기준점을 고를 때 '여유'를 두면, 새 데이터에서 목표가 더 잘 지켜지는가? 대가는 얼마인가?

사용법: python scripts/06b_threshold_margin.py data_raw/creditcard.csv reports

실제 목표는 고정한다
  - 재현율 목표: 테스트에서 사기의 80% 이상 잡기
  - 오경보 예산: 테스트에서 거래 1만 건당 오경보 0.5건 이하
기준점을 고를 때만 더 엄격하게 설정해 본다
  - 재현율: 학습 조각에서 80% / 85% / 90%를 맞추는 기준점
  - 오경보: 학습 조각에서 0.5 / 0.4 / 0.3건을 맞추는 기준점

조각 5개로는 '지킨 비율'이 거칠어서(1조각 = 20%p), 5겹 나누기를 방법만 바꿔 3번 반복 → 테스트 조각 15개.
기준점 고르는 함수는 06_threshold.py의 것을 그대로 가져다 쓴다.
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

from fds_common import LGBM_PARAMS, load_data

th = importlib.import_module("06_threshold")  # 이름이 숫자로 시작해서 import 문 대신 importlib로 불러옴

RECALL_GOAL = 0.80
FP_GOAL = 0.5
RECALL_SETTINGS = [0.80, 0.85, 0.90]
FP_SETTINGS = [0.5, 0.4, 0.3]
N_REPEATS = 3


def main(csv_path: str, out_dir: str) -> None:
    X, y = load_data(csv_path)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=42)
    folds = list(rskf.split(np.zeros(len(y)), y))
    print(f"거래 {len(y):,}건 / 사기 {y.sum()}건 / 테스트 조각 {len(folds)}개 (5겹 × {N_REPEATS}회)", flush=True)

    rows = []
    for k, (train_idx, test_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_te, y_te = X.iloc[test_idx], y[test_idx]
        inner = th.inner_oof_scores(X_tr, y_tr)                       # 학습 조각만으로 만든 점수
        test_score = LGBMClassifier(**LGBM_PARAMS).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]

        for setting in RECALL_SETTINGS:
            thr = th.threshold_for_recall(y_tr, inner, setting)
            r = th.apply_threshold(y_te, test_score, thr)
            rows.append({"fold": k, "policy": "재현율 목표", "setting": setting, **r, "kept": r["recall"] >= RECALL_GOAL})
        for budget in FP_SETTINGS:
            thr = th.threshold_for_fp_budget(y_tr, inner, budget)
            r = th.apply_threshold(y_te, test_score, thr)
            rows.append({"fold": k, "policy": "오경보 예산", "setting": budget, **r, "kept": r["fp_per_10k"] <= FP_GOAL})

        done = [x for x in rows if x["fold"] == k]
        print(f"조각 {k:>2} 완료 | " + " | ".join(
            f"{x['policy'][:3]} {x['setting']}: {'O' if x['kept'] else 'X'}" for x in done), flush=True)

    table = pd.DataFrame(rows)
    summary = (
        table.groupby(["policy", "setting"], sort=False)
        .agg(kept=("kept", "sum"), recall_mean=("recall", "mean"), fp_per_10k_mean=("fp_per_10k", "mean"),
             caught_total=("caught", "sum"), missed_total=("missed", "sum"), false_alarms_total=("false_alarms", "sum"),
             threshold_mean=("threshold", "mean"))
        .round(4)
        .reset_index()
    )
    summary["kept_rate"] = (summary["kept"] / len(folds)).round(3)

    result = {
        "goals": {"recall": RECALL_GOAL, "fp_per_10k": FP_GOAL},
        "n_test_folds": len(folds),
        "summary": summary.to_dict(orient="records"),
        "folds": [{**r, "kept": bool(r["kept"])} for r in rows],
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "threshold_margin_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
