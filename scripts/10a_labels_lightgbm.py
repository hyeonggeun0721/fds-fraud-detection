"""3-3a (LightGBM 쪽): 확인된 사기 라벨이 적을 때 LightGBM은 얼마나 버티나? (5겹 교차검증)

사용법: python scripts/10a_labels_lightgbm.py data_raw/creditcard.csv reports

- 학습 조각에서 사기 라벨을 100 / 50 / 25 / 10 / 5 / 2%만 남기고, 나머지 사기는 '정상'으로 섞어 둔다
- 테스트는 항상 실제 라벨로 채점한다
- 어떤 사기를 숨길지는 fds_common.hide_labels(seed=조각 번호)로 정해서, 오토인코더 쪽(10b)과 똑같이 숨긴다
- PyTorch와 libomp가 충돌하므로 이 스크립트는 LightGBM만 쓴다
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from fds_common import LABEL_FRACS, LGBM_PARAMS, cv_folds, evaluate, hide_labels, load_data


def top_k_hits(y_true, score):
    k = int(y_true.sum())
    return int(y_true[np.argsort(score)[::-1][:k]].sum())


def main(csv_path: str, out_dir: str) -> None:
    X, y = load_data(csv_path)
    folds = cv_folds(y)
    print(f"거래 {len(y):,}건 / 사기 {y.sum()}건 / {len(folds)}겹 / 라벨 비율 {LABEL_FRACS}", flush=True)

    rows = []
    for k, (train_idx, test_idx) in enumerate(folds):
        y_tr = y[train_idx]
        for frac in LABEL_FRACS:
            y_obs = hide_labels(y_tr, frac, seed=k)            # 확인된 사기만 1, 나머지 사기는 0으로 섞임
            model = LGBMClassifier(**LGBM_PARAMS).fit(X.iloc[train_idx], y_obs)
            score = model.predict_proba(X.iloc[test_idx])[:, 1]
            rows.append({"fold": k, "label_frac": frac, "labeled_fraud": int(y_obs.sum()),
                         "hidden_fraud": int(y_tr.sum() - y_obs.sum()),
                         **evaluate(y[test_idx], score), "top_k_hits": top_k_hits(y[test_idx], score)})
            r = rows[-1]
            print(f"조각 {k} | 라벨 {frac:>4.0%} (확인된 사기 {r['labeled_fraud']:>3}건) | PR-AUC {r['pr_auc']:.3f}", flush=True)

    table = pd.DataFrame(rows)
    summary = (table.groupby("label_frac", sort=False)[["labeled_fraud", "pr_auc", "precision_at_recall80", "top_k_hits"]]
               .agg(["mean", "std"]).round(4))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    result = {"method": "LightGBM (지도)", "label_fracs": LABEL_FRACS,
              "summary": summary.reset_index().to_dict(orient="records"), "folds": rows}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels_lightgbm_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(summary.to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
