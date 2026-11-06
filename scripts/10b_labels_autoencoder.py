"""3-3a (오토인코더 쪽): 확인된 사기 라벨이 적을 때 '정상만' 오토인코더는 얼마나 버티나? (5겹 교차검증)

사용법: python scripts/10b_labels_autoencoder.py data_raw/creditcard.csv reports [cpu|mps]

- 오토인코더는 '관측된 라벨이 정상'인 거래로 학습한다 → 확인 안 된 사기가 학습 데이터에 섞여 들어간다
- 학습 길이는 3-3b 결론대로 50에폭 (확인용 오차 최소 시점 모델 사용)
- 숨기는 사기는 LightGBM 쪽(10a)과 똑같다 (fds_common.hide_labels, seed=조각 번호)
- LightGBM과 libomp가 충돌하므로 이 스크립트는 PyTorch만 쓴다
"""
import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch

from fds_common import LABEL_FRACS, cv_folds, evaluate, hide_labels, load_data

ae = importlib.import_module("08_autoencoder")
ae.AE_CONFIG["epochs"] = 50     # 3-3b 결정: 08의 설정(20에폭)을 이 스크립트 안에서만 50으로 바꿔 쓴다


def main(csv_path: str, out_dir: str, device_name: str = "cpu") -> None:
    device = torch.device(device_name)
    X_df, y = load_data(csv_path)
    X_ae = ae.preprocess(X_df)
    folds = cv_folds(y)
    print(f"거래 {len(y):,}건 / 사기 {y.sum()}건 / {len(folds)}겹 / 에폭 {ae.AE_CONFIG['epochs']} / 라벨 비율 {LABEL_FRACS}",
          flush=True)

    rows = []
    for k, (train_idx, test_idx) in enumerate(folds):
        y_tr = y[train_idx]
        for frac in LABEL_FRACS:
            y_obs = hide_labels(y_tr, frac, seed=k)
            fit_idx = train_idx[y_obs == 0]                     # '정상으로 보이는' 거래 = 진짜 정상 + 확인 안 된 사기
            model, scaler, _ = ae.train_autoencoder(X_ae[fit_idx], device)
            score = ae.reconstruction_error(model, scaler, X_ae[test_idx], device)
            rows.append({"fold": k, "label_frac": frac, "labeled_fraud": int(y_obs.sum()),
                         "fraud_mixed_into_training": int(y[fit_idx].sum()),
                         **evaluate(y[test_idx], score), "top_k_hits": ae.top_k_hits(y[test_idx], score)})
            r = rows[-1]
            print(f"조각 {k} | 라벨 {frac:>4.0%} (학습에 섞인 사기 {r['fraud_mixed_into_training']:>3}건) | "
                  f"PR-AUC {r['pr_auc']:.3f}", flush=True)

    table = pd.DataFrame(rows)
    summary = (table.groupby("label_frac", sort=False)[["fraud_mixed_into_training", "pr_auc", "precision_at_recall80", "top_k_hits"]]
               .agg(["mean", "std"]).round(4))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    result = {"method": "오토인코더 (관측상 정상만)", "epochs": ae.AE_CONFIG["epochs"], "device": str(device),
              "label_fracs": LABEL_FRACS, "summary": summary.reset_index().to_dict(orient="records"), "folds": rows}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels_autoencoder_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(summary.to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4])
