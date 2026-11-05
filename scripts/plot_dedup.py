"""2-1단계 결과 그래프: 중복 유지 vs 제거 후 성능 비교.

사용법: python scripts/plot_dedup.py reports
(03_duplicates.py가 만든 reports/dedup_results.json을 읽어 reports/figures/04_dedup_comparison.png 생성)
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
    res = json.loads((rep / "dedup_results.json").read_text())
    rows = res["comparison"]
    models = ["로지스틱 회귀", "로지스틱 회귀 + 클래스 가중치"]
    datas = ["중복 유지(1단계 방식)", "중복 제거 후 분할"]
    colors = {datas[0]: "#9db4c0", datas[1]: "#2e86ab"}
    metrics = [("pr_auc", "PR-AUC"), ("precision_at_recall80", "재현율 80%일 때 정밀도")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.36
    for ax, (key, title) in zip(axes, metrics):
        for i, d in enumerate(datas):
            vals = [next(r[key] for r in rows if r["data"] == d and r["model"] == m) for m in models]
            xs = [x + (i - 0.5) * width for x in range(len(models))]
            bars = ax.bar(xs, vals, width, color=colors[d], label=d)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(["로지스틱 회귀", "+ 클래스 가중치"])
        ax.set_ylim(0, 1)
        ax.set_title(title)
    axes[0].legend(fontsize=9, loc="lower left")
    leak = res["leakage"]
    fig.suptitle(
        f"중복 제거 전후 성능 비교  (기존 분할의 테스트 거래 중 {leak['test_rows_with_copy_in_train']}건이 "
        f"학습 데이터에 똑같은 복사본 존재, 그중 사기 {leak['of_which_fraud']}건)",
        fontsize=10.5,
    )
    fig.tight_layout()
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(rep / "figures" / "04_dedup_comparison.png", dpi=130)
    plt.close(fig)
    print("saved", rep / "figures" / "04_dedup_comparison.png")


if __name__ == "__main__":
    main(sys.argv[1])
