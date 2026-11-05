"""2-3단계: 불균형 처리 방법 비교 (LightGBM, 5겹 교차검증)

사용법: python scripts/05_imbalance.py data_raw/creditcard.csv reports

비교하는 4가지 (모델 설정은 2-2와 같고, 학습 데이터를 다루는 방법만 다름)
  1) 처리 없음      : 2-2 그대로 (기준)
  2) 클래스 가중치  : 사기 1건을 틀렸을 때 벌점을 (정상 수 ÷ 사기 수)배로 키움
  3) 언더샘플링 1:10 : 정상 거래를 무작위로 버려서 사기:정상 = 1:10으로 맞춤
  4) SMOTE 1:10     : 사기 거래 사이에 가짜 사기를 만들어 넣어 사기:정상 = 1:10으로 맞춤

※ 샘플링은 학습 조각에만 한다. 테스트 조각은 실제 비율(0.17%) 그대로 둬야 실제 성능을 잴 수 있다.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier

from fds_common import LGBM_PARAMS, cv_folds, evaluate, load_data

# (2026-11-04) imbalanced-learn 0.12.4 + scikit-learn 1.6 조합에서 나던 '곧 바뀔 기능' 경고를 숨기던 설정.
# Python 3.12 환경(imbalanced-learn 0.14.2 + scikit-learn 1.9.1)에서는 경고가 나지 않지만, 옛 환경 재현을 위해 남겨 둠.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\.")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"sklearn\.")

RATIO = 0.1  # 샘플링 후 사기 수 = 정상 수 × 0.1  (사기:정상 = 1:10)


def prepare_train(method, X_tr, y_tr):
    """방법에 따라 학습 데이터와 모델 설정을 돌려준다."""
    params = dict(LGBM_PARAMS)
    if method == "처리 없음":
        return X_tr, y_tr, params
    if method == "클래스 가중치":
        params["scale_pos_weight"] = (y_tr == 0).sum() / (y_tr == 1).sum()  # 약 600
        return X_tr, y_tr, params
    if method == "언더샘플링 1:10":
        X_res, y_res = RandomUnderSampler(sampling_strategy=RATIO, random_state=0).fit_resample(X_tr, y_tr)
        return X_res, y_res, params
    if method == "SMOTE 1:10":
        X_res, y_res = SMOTE(sampling_strategy=RATIO, k_neighbors=5, random_state=0).fit_resample(X_tr, y_tr)
        return X_res, y_res, params
    raise ValueError(method)


def at_threshold(y_true, score, threshold=0.5):
    """기준점 0.5로 잘랐을 때 실제로 울리는 경보 수 (불균형 처리가 '점수 크기'를 어떻게 바꾸는지 보려고)."""
    pred = score >= threshold
    tp = int((pred & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    return {"caught_at_0.5": tp, "false_alarms_at_0.5": fp, "recall_at_0.5": round(tp / int(y_true.sum()), 4)}


METHODS = ["처리 없음", "클래스 가중치", "언더샘플링 1:10", "SMOTE 1:10"]


def main(csv_path: str, out_dir: str) -> None:
    X, y = load_data(csv_path)
    folds = cv_folds(y)
    print(f"거래 {len(y):,}건 / 사기 {y.sum()}건 / {len(folds)}겹 / 방법 {len(METHODS)}개", flush=True)

    oof = {m: np.zeros(len(y)) for m in METHODS}
    rows = []
    for k, (train_idx, test_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_te, y_te = X.iloc[test_idx], y[test_idx]
        for method in METHODS:
            start = time.perf_counter()
            X_fit, y_fit, params = prepare_train(method, X_tr, y_tr)
            model = LGBMClassifier(**params).fit(X_fit, y_fit)
            seconds = time.perf_counter() - start
            score = model.predict_proba(X_te)[:, 1]
            oof[method][test_idx] = score
            rows.append({
                "fold": k, "method": method, **evaluate(y_te, score), **at_threshold(y_te, score),
                "train_rows": int(len(y_fit)), "train_fraud": int(np.sum(y_fit)), "seconds": round(seconds, 1),
            })
            print(f"조각 {k} | {method:<10} | 학습 {len(y_fit):>7,}건 | PR-AUC {rows[-1]['pr_auc']:.3f} | {seconds:.1f}초", flush=True)

    table = pd.DataFrame(rows)

    # (1) 방법별 요약
    cols = ["pr_auc", "precision_at_recall80", "recall_at_0.5", "false_alarms_at_0.5", "seconds"]
    summary = table.groupby("method")[cols].agg(["mean", "std"]).round(4)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reindex(METHODS)

    # (2) '처리 없음'과 같은 조각끼리 짝지은 차이
    paired = {}
    for metric in ["pr_auc", "precision_at_recall80"]:
        wide = table.pivot(index="fold", columns="method", values=metric)
        paired[metric] = {}
        for method in METHODS[1:]:
            diff = wide[method] - wide["처리 없음"]
            paired[metric][method] = {
                "mean_diff": round(float(diff.mean()), 4),
                "std_diff": round(float(diff.std()), 4),
                "wins_vs_none": int((diff > 0).sum()),
                "per_fold": [round(float(v), 4) for v in diff],
            }

    result = {
        "data": {"rows": int(len(y)), "fraud": int(y.sum())},
        "ratio_after_sampling": RATIO,
        "summary": summary.reset_index().to_dict(orient="records"),
        "paired_vs_none": paired,
        "overall_out_of_fold": {m: {**evaluate(y, s), **at_threshold(y, s)} for m, s in oof.items()},
        "folds": rows,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "imbalance_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    np.savez_compressed(out / "oof_scores_2-3.npz", y=y, names=np.array(METHODS),
                        **{f"score_{i}": oof[m] for i, m in enumerate(METHODS)})
    print(json.dumps({"summary": result["summary"], "paired_vs_none": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
