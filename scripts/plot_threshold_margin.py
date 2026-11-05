"""2-4 보완 결과 그래프: 기준점에 여유를 둘 때 '목표 지킨 비율'과 '대가'.

사용법: python scripts/plot_threshold_margin.py reports
  - 11_threshold_margin.png
      왼쪽  = 재현율 정책: 학습 조각 설정 80/85/90% → 실제 80% 지킨 조각 수(막대) vs 오경보 합계(선, 로그 눈금)
      오른쪽 = 오경보 정책: 학습 조각 설정 0.5/0.4/0.3건 → 실제 0.5건 지킨 조각 수(막대) vs 놓친 사기 합계(선)
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


def main(report_dir: str) -> None:
    rep = Path(report_dir)
    res = json.loads((rep / "threshold_margin_results.json").read_text())
    n = res["n_test_folds"]
    summ = res["summary"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    panels = [
        (axes[0], "재현율 목표", lambda s: f"{s:.0%}", "false_alarms_total", "오경보 합계 (로그 눈금)", True, "#e07a5f",
         f"재현율 정책: 실제 목표 {res['goals']['recall']:.0%}"),
        (axes[1], "오경보 예산", lambda s: f"{s}건", "missed_total", "놓친 사기 합계", False, "#2e86ab",
         f"오경보 정책: 실제 목표 1만 건당 {res['goals']['fp_per_10k']}건"),
    ]
    for ax, policy, fmt, cost_key, cost_label, log, color, title in panels:
        rows = [r for r in summ if r["policy"] == policy]
        xs = range(len(rows))
        bars = ax.bar(xs, [r["kept"] for r in rows], color="#d9e2e7", edgecolor="#555", width=0.55)
        ax.set_ylim(0, n + 2.5)
        ax.set_ylabel(f"실제 목표를 지킨 조각 수 (/{n})")
        ax.set_xticks(list(xs))
        # 지킨 조각 수는 선과 겹치지 않도록 x축 이름에 함께 표시
        ax.set_xticklabels([f"학습 조각에서 {fmt(r['setting'])} 맞춤\n→ {r['kept']}/{n} 지킴" for r in rows], fontsize=9)
        ax.set_title(title)

        ax2 = ax.twinx()
        costs = [r[cost_key] for r in rows]
        ax2.plot(list(xs), costs, color=color, marker="o", lw=2)
        for x, c in zip(xs, costs):
            ax2.annotate(f"{c:,}", (x, c), xytext=(8, 4), textcoords="offset points", color=color, fontsize=9)
        if log:
            ax2.set_yscale("log")
            ax2.set_ylim(min(costs) / 2, max(costs) * 4)
        else:
            ax2.set_ylim(0, max(costs) * 1.3)
        ax2.set_ylabel(cost_label, color=color)
        ax2.tick_params(axis="y", colors=color)

    fig.suptitle("기준점에 여유를 두면: 목표는 더 잘 지켜지지만 대가가 따른다 (테스트 조각 15개)", fontsize=11)
    fig.tight_layout()
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(rep / "figures" / "11_threshold_margin.png", dpi=130)
    plt.close(fig)
    print("saved 11_threshold_margin.png")


if __name__ == "__main__":
    main(sys.argv[1])
