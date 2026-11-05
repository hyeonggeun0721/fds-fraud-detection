"""2-4단계 결과 그래프.

사용법: python scripts/plot_threshold.py reports
  - 10_threshold_by_fold.png : 조각별로 '학습 조각에서 고른 기준점'을 테스트 조각에 적용했을 때 실제 결과
      왼쪽 = 재현율(목표 80%), 오른쪽 = 1만 건당 오경보(예산 0.5건)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next(
    (f for f in ["AppleGothic", "Malgun Gothic", "Noto Sans CJK KR"] if f in _installed), "sans-serif"
)
plt.rcParams["axes.unicode_minus"] = False

STYLE = {
    "고정 0.5": dict(color="#9a9a9a", marker="o"),
    "재현율 80% 목표": dict(color="#e07a5f", marker="s"),
    "오경보 예산": dict(color="#2e86ab", marker="D"),
}


def main(report_dir: str) -> None:
    rep = Path(report_dir)
    res = json.loads((rep / "threshold_results.json").read_text())
    tgt = res["targets"]
    folds = sorted({f["fold"] for f in res["folds"]})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    panels = [
        (axes[0], "test_recall", tgt["recall"], "테스트 재현율 (잡은 사기 비율)", f"목표 {tgt['recall']:.0%}", "재현율 80% 목표"),
        (axes[1], "test_fp_per_10k", tgt["fp_per_10k"], "테스트 오경보 (거래 1만 건당)", f"예산 {tgt['fp_per_10k']}건", "오경보 예산"),
    ]
    offsets = {"고정 0.5": -0.18, "재현율 80% 목표": 0.0, "오경보 예산": 0.18}
    for ax, key, line, ylabel, line_label, owner in panels:
        for policy, st in STYLE.items():
            rows = [f for f in res["folds"] if f["policy"] == policy]
            xs = [f["fold"] + offsets[policy] for f in rows]
            ys = [f[key] for f in rows]
            ax.scatter(xs, ys, s=55 if policy == owner else 35, edgecolor="#333", label=policy, zorder=3, **st)
            if policy == owner:  # 이 패널의 목표를 가진 정책은 기준점 값을 표시
                for x, yv, f in zip(xs, ys, rows):
                    ax.annotate(f"기준점 {f['threshold']:.2f}", (x, yv), xytext=(0, -14), textcoords="offset points",
                                ha="center", fontsize=7, color=st["color"])
        ax.axhline(line, color="#c0392b", ls="--", lw=1.2)
        ax.text(folds[0] - 0.45, line, line_label, color="#c0392b", fontsize=8.5, va="bottom", ha="left")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.04)  # 아래쪽 기준점 글자가 잘리지 않게 여백
        ax.set_xlim(folds[0] - 0.5, folds[-1] + 0.5)
        ax.set_xticks(folds)
        ax.set_xticklabels([f"조각 {k}" for k in folds])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
    kept = res["target_kept_on_test_folds"]
    axes[0].set_title(f"재현율 목표가 지켜진 조각: {kept['재현율 80% 목표']}/{len(folds)}")
    axes[1].set_title(f"오경보 예산이 지켜진 조각: {kept['오경보 예산']}/{len(folds)}")
    axes[0].legend(fontsize=8.5, loc="lower left", bbox_to_anchor=(0.0, 0.08))
    fig.suptitle("학습 조각에서 고른 기준점을 테스트 조각에 그대로 적용한 결과", fontsize=11)
    fig.tight_layout()
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(rep / "figures" / "10_threshold_by_fold.png", dpi=130)
    plt.close(fig)
    print("saved 10_threshold_by_fold.png")


if __name__ == "__main__":
    main(sys.argv[1])
