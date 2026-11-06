"""4-1 결과 그래프: 시간순 전진 검증.

사용법: python scripts/plot_time_split.py reports
  - 16_time_split.png
      왼쪽  = 테스트 구간별 PR-AUC (시간순 / 같은 시험지 무작위 / 다른 시험지 무작위)
      오른쪽 = 테스트 구간별 사기 비율 (전체 평균과 비교)
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


def main(report_dir: str) -> None:
    rep = Path(report_dir)
    res = json.loads((rep / "time_split_results.json").read_text())
    folds = res["folds"]
    tests = sorted({f["test_block"] for f in folds})
    xs = np.arange(len(tests))

    def pick(split, variant, b):
        return [f["pr_auc"] for f in folds if f["split"] == split and f["variant"] == variant and f["test_block"] == b]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    series = [("시간순 (과거만 학습), Time 그대로", "시간순", "A", "#e07a5f", "o", "-"),
              ("시간순 (과거만 학습), Time 제거", "시간순", "B", "#c0392b", "o", ":"),
              ("같은 시험지, 과거+미래 무작위 학습 (3회 평균)", "같은 시험지 무작위", "A", "#2e86ab", "s", "-"),
              ("다른 시험지 무작위 (4-1 대조군)", "무작위(같은 크기)", "A", "#999999", "^", "--")]
    for label, split, v, color, mk, ls in series:
        vals = [pick(split, v, b) for b in tests]
        means = np.array([np.mean(x) for x in vals])
        ax.plot(xs, means, color=color, marker=mk, ls=ls, lw=2, label=label)
        if split == "같은 시험지 무작위":
            lo = means - np.array([min(x) for x in vals])
            hi = np.array([max(x) for x in vals]) - means
            ax.errorbar(xs, means, yerr=[lo, hi], fmt="none", ecolor=color, capsize=4, lw=1.2)
    ax.axvspan(len(tests) - 1.5, len(tests) - 0.5, color="#eee", zorder=0)
    ax.text(len(tests) - 1, 0.705, "비교 의미 없음\n(다른 구간이 모두 과거)", ha="center", fontsize=8, color="#777")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"테스트 구간 {b}" for b in tests])
    ax.set_ylim(0.7, 0.95)
    ax.set_ylabel("PR-AUC")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("시험지를 맞추면 '과거만 학습'의 손해는 거의 사라짐")

    ax = axes[1]
    blocks = res["blocks"]
    rates = [b["fraud_rate"] * 100 for b in blocks]
    overall = sum(b["fraud"] for b in blocks) / sum(b["n"] for b in blocks) * 100
    colors = ["#bbb"] + ["#e07a5f"] * (len(blocks) - 1)
    ax.bar([b["block"] for b in blocks], rates, color=colors)
    ax.axhline(overall, color="#2e86ab", ls="--", lw=1.5, label=f"전체 평균 {overall:.3f}%")
    for b, r in zip(blocks, rates):
        ax.text(b["block"], r + 0.004, f"{r:.3f}%", ha="center", fontsize=8.5)
    ax.set_xticks([b["block"] for b in blocks])
    ax.set_xticklabels([f"구간 {b['block']}\n{b['time_from_h']:.0f}~{b['time_to_h']:.0f}h" for b in blocks], fontsize=8.5)
    ax.set_ylabel("사기 비율 (%)")
    ax.legend(fontsize=8.5)
    ax.set_title("테스트 구간마다 사기 비율이 다름 (회색 = 학습 전용)")

    fig.suptitle("4-1 시간순 전진 검증: 차이의 대부분은 '시간 순서'가 아니라 '시험지(테스트 구간)' 차이", fontsize=11)
    fig.tight_layout()
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(rep / "figures" / "16_time_split.png", dpi=130)
    plt.close(fig)
    print("saved 16_time_split.png")


if __name__ == "__main__":
    main(sys.argv[1])
