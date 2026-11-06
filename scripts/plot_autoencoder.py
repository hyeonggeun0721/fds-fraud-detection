"""3-2단계 결과 그래프.

사용법: python scripts/plot_autoencoder.py reports
  - 13_autoencoder.png
      왼쪽   = PR 곡선 (오토인코더 정상만 / 비지도, IF, LightGBM)
      가운데 = 오토인코더(정상만)의 복원 오차 분포: 정상 vs 사기
      오른쪽 = 오토인코더(비지도)의 복원 오차 분포: 사기까지 배우면 사기 오차가 줄어든다
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from sklearn.metrics import average_precision_score, precision_recall_curve

_installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next(
    (f for f in ["AppleGothic", "Malgun Gothic", "Noto Sans CJK KR"] if f in _installed), "sans-serif"
)
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"오토인코더 (정상만)": "#3a7d44", "오토인코더 (비지도)": "#9bc59d",
          "Isolation Forest (비지도)": "#8e7cc3", "LightGBM (지도)": "#e07a5f"}


def main(report_dir: str) -> None:
    rep = Path(report_dir)
    z = np.load(rep / "oof_scores_3-2.npz")
    y = z["y"]
    names = [str(n) for n in z["names"]]
    scores = {n: z[f"score_{i}"] for i, n in enumerate(names)}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.2, 1, 1]})

    ax = axes[0]
    for n in names:
        p, r, _ = precision_recall_curve(y, scores[n])
        ax.plot(r, p, color=COLORS[n], lw=2, label=f"{n} ({average_precision_score(y, scores[n]):.3f})")
    ax.set_xlabel("재현율 (잡아낸 사기 비율)")
    ax.set_ylabel("정밀도 (경보 중 진짜 사기 비율)")
    ax.set_title("정밀도-재현율 곡선 (괄호 = PR-AUC)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="upper right")

    bins = np.logspace(-2, 3, 70)
    for ax, n in zip(axes[1:], ["오토인코더 (정상만)", "오토인코더 (비지도)"]):
        s = scores[n]
        ax.hist(s[y == 0], bins=bins, density=True, alpha=0.55, color="#9db4c0", label="정상")
        ax.hist(s[y == 1], bins=bins, density=True, alpha=0.65, color="#c0392b", label="사기")
        for grp, c in [(0, "#5d7b88"), (1, "#c0392b")]:
            med = np.median(s[y == grp])
            ax.axvline(med, color=c, ls="--", lw=1.2)
            ax.text(med * 1.15, ax.get_ylim()[1] * (0.9 if grp == 0 else 0.75), f"중앙값 {med:.2f}", color=c, fontsize=8.5)
        ax.set_xscale("log")
        ax.set_xlabel("복원 오차 (로그 눈금)")
        ratio = np.median(s[y == 1]) / np.median(s[y == 0])
        ax.set_title(f"{n}: 사기 오차가 정상의 {ratio:.1f}배")
        ax.legend(fontsize=8.5, loc="upper left")
    axes[1].set_ylabel("밀도")

    fig.suptitle("오토인코더: 정상만 배워야 사기가 '복원이 안 되는 거래'로 드러난다", fontsize=11)
    fig.tight_layout()
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(rep / "figures" / "13_autoencoder.png", dpi=130)
    plt.close(fig)
    print("saved 13_autoencoder.png")


if __name__ == "__main__":
    main(sys.argv[1])
