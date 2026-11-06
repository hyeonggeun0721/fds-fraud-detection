"""3-3b: 오토인코더(정상만)를 더 오래 학습하면 나아지는가? (5겹 교차검증)

사용법: python scripts/09_ae_epochs.py data_raw/creditcard.csv reports [cpu|mps]

3-2에서는 20에폭까지 확인용 오차가 계속 줄었다. 그래서 조각마다 100에폭까지 한 번 학습하면서,
20 / 50 / 100에폭 시점의 '그때까지 확인용 오차가 가장 낮았던 모델'로 각각 테스트한다.
(3-2와 같은 규칙: 확인용 오차 최소 시점의 모델을 쓴다 → 20에폭 결과는 3-2와 같아야 함)
"""
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from fds_common import cv_folds, evaluate, load_data

ae = importlib.import_module("08_autoencoder")   # 모델 구조·전처리·복원 오차 계산을 그대로 사용 (PyTorch만 불러옴)

CHECKPOINTS = [20, 50, 100]


def train_with_checkpoints(X_fit, device, seed=0):
    """3-2와 같은 방식으로 학습하되, CHECKPOINTS 시점마다 '그때까지 최선'인 모델을 저장해 둔다."""
    cfg = ae.AE_CONFIG
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scaler = StandardScaler().fit(X_fit)
    Z = scaler.transform(X_fit).astype(np.float32)
    idx = rng.permutation(len(Z))
    n_val = int(len(Z) * cfg["val_ratio"])
    val, tr = torch.from_numpy(Z[idx[:n_val]]).to(device), torch.from_numpy(Z[idx[n_val:]]).to(device)

    model = ae.AutoEncoder(Z.shape[1], cfg["hidden"], cfg["bottleneck"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()
    history, best_val, best_state, best_epoch, snapshots = [], float("inf"), None, 0, {}
    for epoch in range(1, max(CHECKPOINTS) + 1):
        model.train()
        order = torch.randperm(len(tr), device=device)
        for start in range(0, len(tr), cfg["batch_size"]):
            batch = tr[order[start:start + cfg["batch_size"]]]
            opt.zero_grad()
            loss_fn(model(batch), batch).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val), val).item()
        history.append(round(val_loss, 5))
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch in CHECKPOINTS:
            snapshots[epoch] = ({k: v.clone() for k, v in best_state.items()}, best_epoch)
    return model, scaler, history, snapshots


def main(csv_path: str, out_dir: str, device_name: str = "cpu") -> None:
    device = torch.device(device_name)
    X_df, y = load_data(csv_path)
    X_ae = ae.preprocess(X_df)
    folds = cv_folds(y)
    print(f"거래 {len(y):,}건 / 사기 {y.sum()}건 / {len(folds)}겹 / 장치 {device} / 확인 시점 {CHECKPOINTS}", flush=True)

    oof = {e: np.zeros(len(y)) for e in CHECKPOINTS}
    rows, histories = [], []
    for k, (train_idx, test_idx) in enumerate(folds):
        start = time.perf_counter()
        fit_idx = train_idx[y[train_idx] == 0]                         # 정상만
        model, scaler, hist, snaps = train_with_checkpoints(X_ae[fit_idx], device)
        histories.append({"fold": k, "val_loss": hist})
        for e in CHECKPOINTS:
            state, used_epoch = snaps[e]
            model.load_state_dict(state)
            score = ae.reconstruction_error(model, scaler, X_ae[test_idx], device)
            oof[e][test_idx] = score
            rows.append({"fold": k, "epochs": e, "best_epoch_used": used_epoch, **evaluate(y[test_idx], score),
                         "top_k_hits": ae.top_k_hits(y[test_idx], score)})
        line = " | ".join(f"{r['epochs']}에폭 PR-AUC {r['pr_auc']:.3f}" for r in rows[-len(CHECKPOINTS):])
        print(f"조각 {k} | {line} | {time.perf_counter() - start:.1f}초", flush=True)

    table = pd.DataFrame(rows)
    summary = table.groupby("epochs")[["pr_auc", "precision_at_recall80", "top_k_hits", "best_epoch_used"]].agg(["mean", "std"]).round(4)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    wide = table.pivot(index="fold", columns="epochs", values="pr_auc")
    paired = {f"{e}에폭 − 20에폭": {"mean_diff": round(float((wide[e] - wide[20]).mean()), 4),
                                    "std_diff": round(float((wide[e] - wide[20]).std()), 4),
                                    "wins": int(((wide[e] - wide[20]) > 0).sum())} for e in CHECKPOINTS[1:]}
    result = {
        "checkpoints": CHECKPOINTS, "ae_config": {**ae.AE_CONFIG, "epochs": max(CHECKPOINTS)}, "device": str(device),
        "summary": summary.reset_index().to_dict(orient="records"),
        "paired_pr_auc_vs_20": paired,
        "overall_out_of_fold": {str(e): evaluate(y, s) for e, s in oof.items()},
        "training_history": histories,
        "folds": rows,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ae_epochs_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"summary": result["summary"], "paired": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:4])
