"""3-3a 결과 그래프: 확인된 사기 라벨 비율에 따른 PR-AUC.

사용법: python scripts/plot_labels.py reports
  - 15_label_scarcity.png
    LightGBM(10a), 오토인코더(10b), Isolation Forest(3-1, 라벨 미사용) 기준선
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
    lg = json.loads((rep / "labels_lightgbm_results.json").read_text())
    au = json.loads((rep / "labels_autoencoder_results.json").read_text())
    iso = json.loads((rep / "isolation_forest_results.json").read_text())
    if_mean = next(r["pr_auc_mean"] for r in iso["summary"] if r["method"] == "Isolation Forest (비지도)")

    fracs = lg["label_fracs"]
    xs = np.arange(len(fracs))
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for res, name, color, mk in [(lg, "LightGBM (지도)", "#e07a5f", "o"),
                                 (au, f"오토인코더 (관측상 정상만, {au['epochs']}에폭)", "#3a7d44", "s")]:
        means = np.array([next(r["pr_auc_mean"] for r in res["summary"] if r["label_frac"] == f) for f in fracs])
        stds = np.array([next(r["pr_auc_std"] for r in res["summary"] if r["label_frac"] == f) for f in fracs])
        ax.plot(xs, means, color=color, marker=mk, lw=2.2, label=name)
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.15)
        for x, m in zip(xs, means):
            ax.annotate(f"{m:.3f}", (x, m), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=color)
    ax.axhline(if_mean, color="#8e7cc3", ls="--", lw=1.5, label=f"Isolation Forest (라벨 미사용) {if_mean:.3f}")

    labeled = [next(r["labeled_fraud_mean"] for r in lg["summary"] if r["label_frac"] == f) for f in fracs]
    mixed = [next(r["fraud_mixed_into_training_mean"] for r in au["summary"] if r["label_frac"] == f) for f in fracs]
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{f:.0%}\n확인된 사기 {l:.0f}건\n(정상에 섞인 사기 {m:.0f}건)" for f, l, m in zip(fracs, labeled, mixed)],
                       fontsize=8.5)
    ax.set_xlabel("학습 조각에서 '확인된' 사기 라벨 비율 (나머지 사기는 정상으로 섞여 있음)")
    ax.set_ylabel("PR-AUC (5겹 평균, 띠 = ±표준편차)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("라벨이 적어도 LightGBM이 앞선다 — 오토인코더는 사기가 섞이는 순간 무너짐")
    fig.tight_layout()
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(rep / "figures" / "15_label_scarcity.png", dpi=130)
    plt.close(fig)
    print("saved 15_label_scarcity.png")


if __name__ == "__main__":
    main(sys.argv[1])
