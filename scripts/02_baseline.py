"""1단계-B: 기본 모델(로지스틱 회귀)과 '불균형 데이터에서 정확도의 함정' 확인.

사용법: python -I scripts/02_baseline.py data_raw/creditcard.csv reports
- 분할: 층화 무작위 분할 80/20 (시간 순서 분할은 4단계에서 비교)
- 비교: (a) 전부 정상으로 예측 (b) 로지스틱 회귀 (c) 로지스틱 회귀 + 클래스 가중치
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 한글 글꼴: 설치된 것 중 첫 번째를 사용 (맥 AppleGothic → 윈도우 맑은 고딕 → 리눅스 Noto)
from matplotlib import font_manager

_installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next(
    (f for f in ["AppleGothic", "Malgun Gothic", "Noto Sans CJK KR"] if f in _installed), "sans-serif"
)
plt.rcParams["axes.unicode_minus"] = False

SEED = 42


def evaluate(name, y_true, score, threshold=0.5):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    res = {
        "model": name,
        "accuracy_pct": round(accuracy_score(y_true, pred) * 100, 3),
        "precision": round(precision_score(y_true, pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, pred), 3),
        "pr_auc": round(average_precision_score(y_true, score), 3),
        "roc_auc": round(roc_auc_score(y_true, score), 3) if len(np.unique(score)) > 1 else 0.5,
        "caught_fraud": int(tp),
        "missed_fraud": int(fn),
        "false_alarms": int(fp),
    }
    # 재현율 80%를 달성할 때 정밀도(경보 중 진짜 사기 비율)
    p, r, _ = precision_recall_curve(y_true, score)
    ok = r >= 0.80
    res["precision_at_recall80"] = round(float(p[ok].max()), 3) if ok.any() else 0.0
    return res


def main(csv_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    X = df.drop(columns=["Class"])
    y = df["Class"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    results, curves = [], {}

    # (a) 전부 정상으로 예측
    zero = np.zeros(len(y_te))
    results.append(evaluate("전부 '정상' 예측", y_te, zero))

    # (b) 로지스틱 회귀, (c) 클래스 가중치
    for name, cw in [("로지스틱 회귀", None), ("로지스틱 회귀 + 클래스 가중치", "balanced")]:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=cw, random_state=SEED))
        model.fit(X_tr, y_tr)
        score = model.predict_proba(X_te)[:, 1]
        results.append(evaluate(name, y_te, score))
        curves[name] = precision_recall_curve(y_te, score)

    payload = {
        "split": "stratified random 80/20, seed 42",
        "test_size": int(len(y_te)),
        "test_fraud": int(y_te.sum()),
        "results": results,
    }
    (out / "baseline_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    # 그림 3: PR 곡선
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"로지스틱 회귀": "#2e86ab", "로지스틱 회귀 + 클래스 가중치": "#e07a5f"}
    for name, (p, r, _) in curves.items():
        ap = next(x["pr_auc"] for x in results if x["model"] == name)
        ax.plot(r, p, color=colors[name], label=f"{name} (PR-AUC {ap})")
    ax.axhline(y_te.mean(), color="gray", ls="--", label=f"무작위 기준선 ({y_te.mean():.4f})")
    ax.set_xlabel("재현율 (잡아낸 사기 비율)")
    ax.set_ylabel("정밀도 (경보 중 진짜 사기 비율)")
    ax.set_title("정밀도-재현율 곡선 (테스트셋)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "03_pr_curve_baseline.png", dpi=130)
    plt.close(fig)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
