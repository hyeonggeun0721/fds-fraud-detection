"""3-1단계 결과 그래프.

사용법: python scripts/plot_isolation_forest.py reports data_raw/creditcard.csv
  - 12_isolation_forest.png
      왼쪽  = PR 곡선 (IF 비지도 / IF 정상만 / LightGBM)
      오른쪽 = 금액 분포: 전체 정상 vs 'IF가 가장 수상하다고 본 정상' vs 사기  → "특이함 ≠ 사기"
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from sklearn.metrics import average_precision_score, precision_recall_curve

from fds_common import load_data

_installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next(
    (f for f in ["AppleGothic", "Malgun Gothic", "Noto Sans CJK KR"] if f in _installed), "sans-serif"
)
plt.rcParams["axes.unicode_minus"] = False

COLORS = ["#8e7cc3", "#c27ba0", "#e07a5f"]


def main(report_dir: str, csv_path: str) -> None:
    rep = Path(report_dir)
    z = np.load(rep / "oof_scores_3-1.npz")
    y = z["y"]
    names = [str(n) for n in z["names"]]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # ── 왼쪽: PR 곡선 ──
    ax = axes[0]
    for i, name in enumerate(names):
        s = z[f"score_{i}"]
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=COLORS[i], lw=2, label=f"{name} ({average_precision_score(y, s):.3f})")
    ax.axhline(y.mean(), color="#999", ls="--", lw=1, label=f"무작위 기준선 ({y.mean():.4f})")
    ax.set_xlabel("재현율 (잡아낸 사기 비율)")
    ax.set_ylabel("정밀도 (경보 중 진짜 사기 비율)")
    ax.set_title("정밀도-재현율 곡선 (괄호 = PR-AUC)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="upper right")

    # ── 오른쪽: 금액 분포 ──
    X, y2 = load_data(csv_path)
    assert len(y2) == len(y) and (y2 == y).all(), "데이터와 점수 파일의 거래 순서가 다름"
    amount = X["Amount"].to_numpy()
    s_if = z["score_0"]                                   # IF (비지도)
    k = int(y.sum())
    top = np.argsort(s_if)[::-1][:k]                      # IF가 가장 수상하다고 본 상위 k건
    groups = [
        ("전체 정상", amount[y == 0]),
        (f"IF 상위 {k}건 중\n정상 ({int((y[top] == 0).sum())}건)", amount[top][y[top] == 0]),
        (f"사기 ({k}건)", amount[y == 1]),
    ]
    ax = axes[1]
    data = [np.log10(g + 1) for _, g in groups]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], ["#cfd8dc", "#8e7cc3", "#e07a5f"]):
        patch.set_facecolor(c)
    for i, (_, g) in enumerate(groups, start=1):
        ax.text(i + 0.28, np.log10(np.median(g) + 1), f"중앙값 {np.median(g):,.0f}", va="center", ha="left", fontsize=8.5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    tick_values = [0, 10, 100, 1000, 10000]
    ax.set_yticks([np.log10(v + 1) for v in tick_values])   # 금액+1에 로그를 씌웠으므로 같은 방식으로 눈금 위치 계산
    ax.set_yticklabels([f"{v:,}" for v in tick_values])
    ax.set_xlim(0.5, 3.9)
    ax.set_ylabel("거래 금액 (로그 눈금)")
    ax.set_title("IF가 수상하다고 본 정상 거래는 '고액'이 많다")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("라벨 없는 이상탐지(Isolation Forest): 특이한 거래 ≠ 사기 거래", fontsize=11)
    fig.tight_layout()
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(rep / "figures" / "12_isolation_forest.png", dpi=130)
    plt.close(fig)
    print("saved 12_isolation_forest.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
