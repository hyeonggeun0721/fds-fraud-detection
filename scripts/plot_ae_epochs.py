"""3-3b 결과 그래프: 오토인코더 학습 길이.

사용법: python scripts/plot_ae_epochs.py reports
  - 14_ae_epochs.png
      왼쪽  = 조각별 PR-AUC (20 / 50 / 100에폭)
      오른쪽 = 조각별 확인용 오차(정상 거래 복원 오차) 학습 곡선
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

_installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next(
    (f for f in ["AppleGothic", "Malgun Gothic", "Noto Sans CJK KR"] if f in _installed), "sans-serif"
)
plt.rcParams["axes.unicode_minus"] = False

FOLD_COLORS = ["#2e86ab", "#e07a5f", "#c0392b", "#3a7d44", "#8e7cc3"]


def main(report_dir: str) -> None:
    rep = Path(report_dir)
    res = json.loads((rep / "ae_epochs_results.json").read_text())
    cps = res["checkpoints"]
    folds = sorted({f["fold"] for f in res["folds"]})

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    for k in folds:
        vals = [next(f["pr_auc"] for f in res["folds"] if f["fold"] == k and f["epochs"] == e) for e in cps]
        ax.plot(cps, vals, marker="o", color=FOLD_COLORS[k], label=f"조각 {k}")
    means = [next(r["pr_auc_mean"] for r in res["summary"] if r["epochs"] == e) for e in cps]
    ax.plot(cps, means, color="#222", lw=3, marker="s", label="평균")
    for e, m in zip(cps, means):
        ax.annotate(f"{m:.3f}", (e, m), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(cps)
    ax.set_xlabel("학습 에폭")
    ax.set_ylabel("테스트 PR-AUC")
    ax.set_title("사기 탐지 성능: 50에폭까지는 모든 조각이 오르고, 100에폭은 엇갈림")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    for h in res["training_history"]:
        v = h["val_loss"]
        ax.plot(np.arange(1, len(v) + 1), v, color=FOLD_COLORS[h["fold"]], lw=1.4, label=f"조각 {h['fold']}")
    for e in cps:
        ax.axvline(e, color="#bbb", ls=":", lw=1)
    ax.set_xlabel("학습 에폭")
    ax.set_ylabel("확인용 오차 (정상 거래 복원 오차)")
    ax.set_title("정상 거래 복원은 100에폭까지 계속 좋아짐")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("정상을 더 잘 복원한다 ≠ 사기를 더 잘 찾는다 (오토인코더 정상만, 5겹 교차검증)", fontsize=11)
    fig.tight_layout()
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(rep / "figures" / "14_ae_epochs.png", dpi=130)
    plt.close(fig)
    print("saved 14_ae_epochs.png")


if __name__ == "__main__":
    main(sys.argv[1])
