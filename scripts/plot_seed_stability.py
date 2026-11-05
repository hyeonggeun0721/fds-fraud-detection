"""2-1 보완 결과 그래프: 시드 10회 반복 시 점수 흔들림.

사용법: python scripts/plot_seed_stability.py reports
(03b_seed_stability.py가 만든 reports/seed_stability.json → reports/figures/05_seed_stability.png)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

_installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next(
    (f for f in ["AppleGothic", "Malgun Gothic", "Noto Sans CJK KR"] if f in _installed), "sans-serif"
)
plt.rcParams["axes.unicode_minus"] = False


def main(report_dir: str) -> None:
    rep = Path(report_dir)
    runs = pd.DataFrame(json.loads((rep / "seed_stability.json").read_text())["runs"])
    combos = [
        ("중복 유지", "로지스틱 회귀"),
        ("중복 유지", "로지스틱 회귀 + 클래스 가중치"),
        ("중복 제거", "로지스틱 회귀"),
        ("중복 제거", "로지스틱 회귀 + 클래스 가중치"),
    ]
    labels = ["중복 유지\n로지스틱", "중복 유지\n+가중치", "중복 제거\n로지스틱", "중복 제거\n+가중치"]
    colors = ["#9db4c0", "#9db4c0", "#2e86ab", "#2e86ab"]
    metrics = [("pr_auc", "PR-AUC"), ("precision_at_recall80", "재현율 80%일 때 정밀도")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (key, title) in zip(axes, metrics):
        for i, (d, m) in enumerate(combos):
            vals = runs[(runs["data"] == d) & (runs["model"] == m)].sort_values("seed")[key]
            jitter = [(s - 4.5) * 0.025 for s in range(len(vals))]  # 점이 겹치지 않게 살짝 옆으로
            ax.scatter([i + j for j in jitter], vals, color=colors[i], edgecolor="#333", s=28, zorder=3)
            ax.hlines(vals.mean(), i - 0.25, i + 0.25, color="#c0392b", lw=2, zorder=4)
            ax.text(i + 0.28, vals.mean(), f"{vals.mean():.3f}", va="center", fontsize=8.5, color="#c0392b")
            ax.vlines(i, vals.min(), vals.max(), color="#999", lw=1, zorder=2)
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlim(-0.6, len(combos) - 0.2)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("점수 (점 하나 = 시드 하나, 빨간 선 = 평균)")
    fig.suptitle("나누는 방법(시드 0~9)만 바꿨을 때 점수가 흔들리는 범위", fontsize=11)
    fig.tight_layout()
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(rep / "figures" / "05_seed_stability.png", dpi=130)
    plt.close(fig)
    print("saved", rep / "figures" / "05_seed_stability.png")


if __name__ == "__main__":
    main(sys.argv[1])
