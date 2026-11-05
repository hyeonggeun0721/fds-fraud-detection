"""2-2단계 결과 그래프.

사용법: python scripts/plot_tree_model.py reports
  - 06_pr_curve_tree_model.png : 5겹 전체 점수(OOF)로 그린 PR 곡선 (로지스틱 / LightGBM / 고치기 전 LightGBM)
  - 07_tree_model_by_fold.png  : 조각별 PR-AUC를 같은 조각끼리 짝지어 비교
"""
import json
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

COLORS = {"로지스틱 회귀": "#2e86ab", "LightGBM": "#e07a5f"}


def main(report_dir: str) -> None:
    rep = Path(report_dir)
    fig_dir = rep / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── 그림 1: PR 곡선 ─────────────────────────────────
    z = np.load(rep / "oof_scores_2-2.npz")
    y = z["y"]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for i, name in enumerate(z["names"]):
        s = z[f"score_{i}"]
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=COLORS[str(name)], lw=2, label=f"{name} (PR-AUC {average_precision_score(y, s):.3f})")
    old = rep / "oof_scores_2-2_v1_exploded.npz"
    if old.exists():  # 고치기 전(min_child_weight 기본값) LightGBM
        zo = np.load(old)
        s = zo["score_1"]
        p, r, _ = precision_recall_curve(zo["y"], s)
        ax.plot(r, p, color="#999", lw=1.2, ls="--",
                label=f"LightGBM 고치기 전 (PR-AUC {average_precision_score(zo['y'], s):.3f})")
    ax.axvline(0.8, color="#bbb", lw=0.8, ls=":")
    ax.text(0.805, 0.03, "재현율 80%", fontsize=8, color="#777")
    ax.set_xlabel("재현율 (잡아낸 사기 비율)")
    ax.set_ylabel("정밀도 (경보 중 진짜 사기 비율)")
    ax.set_title("정밀도-재현율 곡선 (5겹 교차검증, 중복 제거 데이터)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "06_pr_curve_tree_model.png", dpi=130)
    plt.close(fig)

    # ── 그림 2: 조각별 짝지은 비교 ──────────────────────
    res = json.loads((rep / "tree_model_results.json").read_text())
    folds = sorted({f["fold"] for f in res["folds"]})
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    right_ends = []
    for k in folds:
        vals = {f["model"]: f["pr_auc"] for f in res["folds"] if f["fold"] == k}
        ax.plot([0, 1], [vals["로지스틱 회귀"], vals["LightGBM"]], color="#bbb", lw=1, zorder=1)
        for x, m in enumerate(["로지스틱 회귀", "LightGBM"]):
            ax.scatter(x, vals[m], color=COLORS[m], s=45, zorder=2, edgecolor="#333")
        right_ends.append((vals["LightGBM"], k))
    # 조각 이름표가 겹치지 않도록 최소 간격을 두고 배치
    gap = 0.006
    placed = []
    for y_val, k in sorted(right_ends):
        y_lab = max(y_val, placed[-1] + gap) if placed else y_val
        placed.append(y_lab)
        ax.annotate(f"조각 {k}", xy=(1, y_val), xytext=(1.06, y_lab), va="center", fontsize=8, color="#555",
                    arrowprops=dict(arrowstyle="-", color="#ccc", lw=0.6))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["로지스틱 회귀", "LightGBM"])
    ax.set_xlim(-0.3, 1.35)
    ax.set_ylabel("PR-AUC")
    pr = res["paired_lightgbm_minus_logistic"]["pr_auc"]
    ax.set_title(f"같은 조각끼리 비교: LightGBM이 {pr['lightgbm_wins']}/{len(folds)}번 우세, 평균 +{pr['mean_diff']:.3f}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "07_tree_model_by_fold.png", dpi=130)
    plt.close(fig)
    print("saved 06_pr_curve_tree_model.png, 07_tree_model_by_fold.png")


if __name__ == "__main__":
    main(sys.argv[1])
