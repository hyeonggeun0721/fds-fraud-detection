"""1단계-A: 카드 사기 데이터 탐색(EDA).

사용법: python -I scripts/01_eda.py data_raw/creditcard.csv reports
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 한글 글꼴: 설치된 것 중 첫 번째를 사용 (맥 AppleGothic → 윈도우 맑은 고딕 → 리눅스 Noto)
from matplotlib import font_manager

_installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next(
    (f for f in ["AppleGothic", "Malgun Gothic", "Noto Sans CJK KR"] if f in _installed), "sans-serif"
)
plt.rcParams["axes.unicode_minus"] = False

EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492


def main(csv_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # 무결성 검증: 원본(Kaggle mlg-ulb/creditcardfraud)과 행 수·사기 건수 일치 확인
    n_rows, n_fraud = len(df), int(df["Class"].sum())
    assert list(df.columns[:1]) == ["Time"] and df.columns[-1] == "Class", "컬럼 구조 불일치"
    assert n_rows == EXPECTED_ROWS, f"행 수 불일치: {n_rows}"
    assert n_fraud == EXPECTED_FRAUD, f"사기 건수 불일치: {n_fraud}"

    fraud, normal = df[df["Class"] == 1], df[df["Class"] == 0]

    # 정상 vs 사기에서 평균 차이가 큰 변수 (표준화한 평균 차이)
    feats = [c for c in df.columns if c.startswith("V")]
    std = df[feats].std()
    effect = ((fraud[feats].mean() - normal[feats].mean()) / std).abs().sort_values(ascending=False)

    summary = {
        "rows": n_rows,
        "fraud": n_fraud,
        "fraud_rate_pct": round(n_fraud / n_rows * 100, 4),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "time_span_hours": round(df["Time"].max() / 3600, 1),
        "amount": {
            "normal_median": round(float(normal["Amount"].median()), 2),
            "fraud_median": round(float(fraud["Amount"].median()), 2),
            "normal_mean": round(float(normal["Amount"].mean()), 2),
            "fraud_mean": round(float(fraud["Amount"].mean()), 2),
            "fraud_max": round(float(fraud["Amount"].max()), 2),
            "fraud_amount_zero": int((fraud["Amount"] == 0).sum()),
        },
        "top_separating_features": {k: round(float(v), 2) for k, v in effect.head(8).items()},
        "all_normal_baseline_accuracy_pct": round((1 - n_fraud / n_rows) * 100, 3),
    }
    (out / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    # 그림 1: 시간대별 거래량과 사기 비율
    hours = (df["Time"] // 3600).astype(int)
    by_hour = df.groupby(hours)["Class"].agg(["count", "sum"])
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(by_hour.index, by_hour["count"], color="#9db4c0", label="전체 거래 수")
    ax1.set_xlabel("데이터 시작 후 경과 시간 (시)")
    ax1.set_ylabel("거래 수")
    ax2 = ax1.twinx()
    ax2.plot(by_hour.index, by_hour["sum"] / by_hour["count"] * 100, color="#c0392b", marker="o", ms=3, label="사기 비율(%)")
    ax2.set_ylabel("사기 비율 (%)")
    ax1.set_title("시간대별 거래 수와 사기 비율")
    fig.tight_layout()
    fig.savefig(fig_dir / "01_hourly_fraud_rate.png", dpi=130)
    plt.close(fig)

    # 그림 2: 정상/사기 분리력이 큰 변수 상위 4개 분포
    top4 = list(effect.head(4).index)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, f in zip(axes, top4):
        lo, hi = np.percentile(df[f], [0.5, 99.5])
        bins = np.linspace(min(lo, fraud[f].min()), hi, 60)
        ax.hist(normal[f], bins=bins, density=True, alpha=0.6, color="#5d8aa8", label="정상")
        ax.hist(fraud[f], bins=bins, density=True, alpha=0.6, color="#c0392b", label="사기")
        ax.set_title(f)
    axes[0].legend()
    fig.suptitle("정상 vs 사기 분포 차이가 가장 큰 변수")
    fig.tight_layout()
    fig.savefig(fig_dir / "02_top_features.png", dpi=130)
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
