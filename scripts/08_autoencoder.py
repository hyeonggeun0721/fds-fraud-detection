"""3-2단계: 정상 거래만 배운 오토인코더가 사기를 '낯설다'고 알아보는가? (PyTorch, 5겹 교차검증)

사용법: python scripts/08_autoencoder.py data_raw/creditcard.csv reports [cpu|mps]
        (세 번째 값을 생략하면 cpu. 작은 모델이라 cpu가 충분히 빠르고 결과가 매번 같음)

오토인코더: 거래(변수 30개)를 8개 숫자로 압축했다가 다시 30개로 복원하도록 학습.
           정상 거래로만 배우면 정상은 잘 복원하고, 정상과 다른 거래는 복원 오차가 커진다 → 오차 = 의심 점수

비교하는 4가지 (같은 5조각, 같은 지표)
  1) 오토인코더 (정상만)  : 학습 조각의 정상 거래만으로 학습
  2) 오토인코더 (비지도)  : 학습 조각 전부를 라벨 없이 학습
  3) Isolation Forest (비지도) : 3-1 결과 점수 파일(reports/oof_scores_3-1.npz)에서 불러옴
  4) LightGBM (지도)      : 3-1 결과 점수 파일에서 불러옴

v2(2026-11-05): 1차 실행에서 PyTorch 학습 뒤 LightGBM이 병렬 처리를 시작할 때 맥에서 프로그램이 강제 종료됨
(Segmentation fault). 원인은 PyTorch·LightGBM(Homebrew)·scikit-learn이 각자 가져온 libomp 3개가 한 프로그램에 동시에
올라간 충돌. 그래서 이 스크립트는 PyTorch만 학습하고, IF·LightGBM은 3-1에서 같은 5조각으로 계산해 둔 점수를 쓴다.
"""
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

# LightGBM이 들어 있는 07 스크립트를 불러오면 그쪽 libomp도 함께 올라오므로, 작은 함수 하나는 여기 다시 적는다
REFERENCE_SCORES = "oof_scores_3-1.npz"   # 3-1 결과: IF (비지도), IF (정상만), LightGBM (지도)


def top_k_hits(y_true, score):
    """의심 점수 상위 k건(k = 실제 사기 수)을 경보로 울렸을 때 그중 진짜 사기 수. (07_isolation_forest.py와 같음)"""
    k = int(y_true.sum())
    return int(y_true[np.argsort(score)[::-1][:k]].sum())

AE_CONFIG = dict(
    hidden=20,          # 30 → 20 → 8 → 20 → 30
    bottleneck=8,       # 가장 좁은 층: 거래를 8개 숫자로 압축
    epochs=20,          # 학습 데이터를 20바퀴 반복
    batch_size=512,     # 한 번에 512건씩 보고 가중치 수정
    lr=1e-3,            # 학습률 (Adam)
    val_ratio=0.1,      # 학습에 쓰는 데이터 중 10%는 '복원 오차 확인용'으로 떼어 둠
)
METHODS = ["오토인코더 (정상만)", "오토인코더 (비지도)", "Isolation Forest (비지도)", "LightGBM (지도)"]


class AutoEncoder(nn.Module):
    def __init__(self, n_features, hidden, bottleneck):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, bottleneck), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, n_features))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def preprocess(X):
    """금액은 한쪽으로 길게 치우쳐 있어서 log(1+금액)으로 바꾼다. 나머지는 그대로."""
    X = X.copy()
    X["Amount"] = np.log1p(X["Amount"])
    return X.to_numpy(dtype=np.float32)


def train_autoencoder(X_fit, device, seed=0):
    """X_fit으로 오토인코더를 학습하고 (모델, 스케일러, 에폭별 확인용 오차)를 돌려준다."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scaler = StandardScaler().fit(X_fit)                       # 평균 0, 표준편차 1 (학습 데이터 기준)
    Z = scaler.transform(X_fit).astype(np.float32)

    idx = rng.permutation(len(Z))
    n_val = int(len(Z) * AE_CONFIG["val_ratio"])
    val, tr = torch.from_numpy(Z[idx[:n_val]]).to(device), torch.from_numpy(Z[idx[n_val:]]).to(device)

    model = AutoEncoder(Z.shape[1], AE_CONFIG["hidden"], AE_CONFIG["bottleneck"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=AE_CONFIG["lr"])
    loss_fn = nn.MSELoss()
    history, best_state, best_val = [], None, float("inf")
    for epoch in range(AE_CONFIG["epochs"]):
        model.train()
        order = torch.randperm(len(tr), device=device)
        for start in range(0, len(tr), AE_CONFIG["batch_size"]):
            batch = tr[order[start:start + AE_CONFIG["batch_size"]]]
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)               # 입력 = 정답 (자기 자신을 복원)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val), val).item()
        history.append(round(val_loss, 5))
        if val_loss < best_val:                                # 확인용 오차가 가장 낮았던 시점의 모델을 남김
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, scaler, history


def reconstruction_error(model, scaler, X, device):
    """거래마다 복원 오차(변수 30개의 제곱 오차 평균) = 의심 점수."""
    Z = torch.from_numpy(scaler.transform(X).astype(np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        err = ((model(Z) - Z) ** 2).mean(dim=1)
    return err.cpu().numpy()


def main(csv_path: str, out_dir: str, device_name: str = "cpu") -> None:
    device = torch.device(device_name)
    X_df, y = load_data(csv_path)
    X_ae = preprocess(X_df)                                    # 오토인코더용
    folds = cv_folds(y)
    print(f"거래 {len(y):,}건 / 사기 {y.sum()}건 / {len(folds)}겹 / 장치 {device}", flush=True)

    ref = np.load(Path(out_dir) / REFERENCE_SCORES)
    assert len(ref["y"]) == len(y) and (ref["y"] == y).all(), "3-1 점수 파일과 거래 순서가 다름 → 07을 먼저 다시 실행"
    ref_scores = {str(n): ref[f"score_{i}"] for i, n in enumerate(ref["names"])}

    oof = {m: np.zeros(len(y)) for m in METHODS}
    rows, histories = [], []
    for k, (train_idx, test_idx) in enumerate(folds):
        y_tr, y_te = y[train_idx], y[test_idx]
        for method in METHODS:
            start = time.perf_counter()
            if method.startswith("오토인코더"):
                fit_idx = train_idx[y_tr == 0] if "정상만" in method else train_idx
                model, scaler, hist = train_autoencoder(X_ae[fit_idx], device)
                score = reconstruction_error(model, scaler, X_ae[test_idx], device)
                histories.append({"fold": k, "method": method, "val_loss": hist})
            else:                                              # IF, LightGBM: 3-1에서 같은 조각으로 계산해 둔 점수
                score = ref_scores[method][test_idx]
            seconds = time.perf_counter() - start
            oof[method][test_idx] = score
            hits = top_k_hits(y_te, score)
            rows.append({"fold": k, "method": method, **evaluate(y_te, score), "top_k_hits": hits,
                         "seconds": round(seconds, 1)})
            print(f"조각 {k} | {method:<20} | PR-AUC {rows[-1]['pr_auc']:.3f} | "
                  f"상위 {int(y_te.sum())}건 중 사기 {hits}건 | {seconds:.1f}초", flush=True)

    table = pd.DataFrame(rows)
    summary = (table.groupby("method", sort=False)[["pr_auc", "precision_at_recall80", "top_k_hits", "seconds"]]
               .agg(["mean", "std"]).round(4))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]

    wide = table.pivot(index="fold", columns="method", values="pr_auc")
    paired = {}
    for a, b in [("오토인코더 (정상만)", "LightGBM (지도)"), ("오토인코더 (정상만)", "Isolation Forest (비지도)"),
                 ("오토인코더 (정상만)", "오토인코더 (비지도)")]:
        diff = wide[a] - wide[b]
        paired[f"{a} − {b}"] = {"mean_diff": round(float(diff.mean()), 4), "std_diff": round(float(diff.std()), 4),
                                 "wins": int((diff > 0).sum())}

    # 서로 다른 사기를 잡는가? (5겹 전체 점수 기준, 각자 상위 k건에 든 사기)
    k_all = int(y.sum())
    caught = {m: set(np.where(y == 1)[0]) & set(np.argsort(s)[::-1][:k_all]) for m, s in oof.items()}
    overlap = {m: {"caught_in_top_k": len(caught[m]),
                   "not_caught_by_lightgbm": len(caught[m] - caught["LightGBM (지도)"])} for m in METHODS}

    result = {
        "data": {"rows": int(len(y)), "fraud": int(y.sum())},
        "ae_config": AE_CONFIG, "device": str(device),
        "reference_scores": f"IF·LightGBM은 {REFERENCE_SCORES}(3-1)에서 불러옴, 표의 seconds는 0",
        "summary": summary.reset_index().to_dict(orient="records"),
        "paired_pr_auc": paired,
        "overall_out_of_fold": {m: {**evaluate(y, s), "top_k_hits": top_k_hits(y, s)} for m, s in oof.items()},
        "overlap_with_lightgbm": overlap,
        "training_history": histories,
        "folds": rows,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "autoencoder_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    np.savez_compressed(out / "oof_scores_3-2.npz", y=y, names=np.array(METHODS),
                        **{f"score_{i}": oof[m] for i, m in enumerate(METHODS)})
    print(json.dumps({"summary": result["summary"], "paired": paired, "overlap": overlap}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:4])
