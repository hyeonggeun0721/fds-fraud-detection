"""3-1단계: 라벨 없이 '이상한 거래'만 찾으면 사기가 얼마나 잡히나? (Isolation Forest, 5겹 교차검증)

사용법: python scripts/07_isolation_forest.py data_raw/creditcard.csv reports

비교하는 3가지 (같은 5조각, 같은 지표)
  1) Isolation Forest (비지도)   : 학습 조각의 거래를 라벨 없이 전부 사용
  2) Isolation Forest (정상만)   : 학습 조각에서 정상 거래만 사용 → '정상의 모습'만 배움
  3) LightGBM (지도, 기준)       : 2-2 최종 모델. 사기 라벨을 보고 배움

※ 라벨은 학습에 쓰지 않지만(1), 평가는 2단계와 똑같이 실제 라벨로 PR-AUC를 잰다.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest

from fds_common import LGBM_PARAMS, cv_folds, evaluate, load_data

IF_PARAMS = dict(
    n_estimators=200,      # 나무 200그루
    max_samples=256,       # 나무마다 거래 256건만 무작위로 뽑아 만든다 (원 논문 기본값)
    random_state=0,
    n_jobs=-1,             # CPU 코어를 모두 사용
)
METHODS = ["Isolation Forest (비지도)", "Isolation Forest (정상만)", "LightGBM (지도)"]


def top_k_hits(y_true, score):
    """의심 점수 상위 k건(k = 실제 사기 수)을 경보로 울렸을 때 그중 진짜 사기 수."""
    k = int(y_true.sum())
    top = np.argsort(score)[::-1][:k]
    return int(y_true[top].sum())


def fit_and_score(method, X_tr, y_tr, X_te):
    """방법별로 학습하고 테스트 거래의 '의심 점수'(클수록 수상함)를 돌려준다."""
    if method == "Isolation Forest (비지도)":
        model = IsolationForest(**IF_PARAMS).fit(X_tr)            # 라벨(y_tr)을 넘기지 않음
        return -model.score_samples(X_te)                         # score_samples는 '정상다움' → 부호를 뒤집음
    if method == "Isolation Forest (정상만)":
        model = IsolationForest(**IF_PARAMS).fit(X_tr[y_tr == 0])  # 정상 거래만
        return -model.score_samples(X_te)
    if method == "LightGBM (지도)":
        model = LGBMClassifier(**LGBM_PARAMS).fit(X_tr, y_tr)
        return model.predict_proba(X_te)[:, 1]
    raise ValueError(method)


def main(csv_path: str, out_dir: str) -> None:
    X, y = load_data(csv_path)
    folds = cv_folds(y)
    print(f"거래 {len(y):,}건 / 사기 {y.sum()}건 / {len(folds)}겹", flush=True)

    oof = {m: np.zeros(len(y)) for m in METHODS}
    rows = []
    for k, (train_idx, test_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_te, y_te = X.iloc[test_idx], y[test_idx]
        for method in METHODS:
            start = time.perf_counter()
            score = fit_and_score(method, X_tr, y_tr, X_te)
            seconds = time.perf_counter() - start
            oof[method][test_idx] = score
            hits = top_k_hits(y_te, score)
            rows.append({"fold": k, "method": method, **evaluate(y_te, score),
                         "top_k_hits": hits, "seconds": round(seconds, 1)})
            print(f"조각 {k} | {method:<22} | PR-AUC {rows[-1]['pr_auc']:.3f} | "
                  f"상위 {int(y_te.sum())}건 중 사기 {hits}건 | {seconds:.1f}초", flush=True)

    table = pd.DataFrame(rows)
    summary = (table.groupby("method", sort=False)[["pr_auc", "precision_at_recall80", "top_k_hits", "seconds"]]
               .agg(["mean", "std"]).round(4))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]

    # LightGBM과 같은 조각끼리 짝지은 차이
    wide = table.pivot(index="fold", columns="method", values="pr_auc")
    paired = {}
    for method in METHODS[:2]:
        diff = wide[method] - wide["LightGBM (지도)"]
        paired[method] = {"mean_diff": round(float(diff.mean()), 4), "std_diff": round(float(diff.std()), 4),
                          "wins_vs_lightgbm": int((diff > 0).sum())}

    result = {
        "data": {"rows": int(len(y)), "fraud": int(y.sum())},
        "if_params": {k: v for k, v in IF_PARAMS.items() if k != "n_jobs"},
        "summary": summary.reset_index().to_dict(orient="records"),
        "paired_pr_auc_vs_lightgbm": paired,
        "overall_out_of_fold": {m: {**evaluate(y, s), "top_k_hits": top_k_hits(y, s)} for m, s in oof.items()},
        "folds": rows,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "isolation_forest_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    np.savez_compressed(out / "oof_scores_3-1.npz", y=y, names=np.array(METHODS),
                        **{f"score_{i}": oof[m] for i, m in enumerate(METHODS)})
    print(json.dumps({"summary": result["summary"], "paired": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
