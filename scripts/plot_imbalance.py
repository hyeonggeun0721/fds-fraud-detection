"""2-3단계 결과 그래프.

사용법: python scripts/plot_imbalance.py reports
  - 08_imbalance_pr_auc.png      : 방법별 조각 PR-AUC (점 하나 = 조각 하나)
  - 09_imbalance_false_alarms.png: 같은 사기 수준을 잡을 때 오경보 수 비교
        (기준점 0.5로 자를 때 vs 각 방법에 맞게 기준점을 옮겨 재현율 80%를 맞출 때)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from sklearn.metrics import precision_recall_curve

_installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next(
    (f for f in ["AppleGothic", "Malgun Gothic", "Noto Sans CJK KR"] if f in _installed), "sans-serif"
)
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"처리 없음": "#7a7a7a", "클래스 가중치": "#c0392b", "언더샘플링 1:10": "#e0a458", "SMOTE 1:10": "#2e86ab"}


def alarms_at_recall(y, s, target=0.80):
    """재현율 target을 넘기는 가장 높은 기준점에서 (잡은 사기, 오경보)."""
    p, r, t = precision_recall_curve(y, s)
    ok = np.where(r[:-1] >= target)[0]          # 마지막 점(재현율 0)은 기준점이 없음
    j = ok[np.argmax(p[ok])]
    pred = s >= t[j]
    return int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())


def main(report_dir: str) -> None:
    rep = Path(report_dir)
    res = json.loads((rep / "imbalance_results.json").read_text())
    methods = [r["method"] for r in res["summary"]]
    fig_dir = rep / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── 그림 1: 조각별 PR-AUC ───────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for i, m in enumerate(methods):
        vals = [f["pr_auc"] for f in res["folds"] if f["method"] == m]
        ax.scatter([i + (k - 2) * 0.05 for k in range(len(vals))], vals, color=COLORS[m], edgecolor="#333", s=40, zorder=3)
        ax.hlines(np.mean(vals), i - 0.22, i + 0.22, color="#222", lw=2, zorder=4)
        ax.text(i + 0.25, np.mean(vals), f"{np.mean(vals):.3f}", va="center", fontsize=9)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m + ("\n(점수 폭주)" if m == "클래스 가중치" else "") for m in methods])
    ax.set_ylabel("PR-AUC (점 = 조각, 굵은 선 = 평균)")
    ax.set_title("불균형 처리 방법별 PR-AUC (LightGBM, 5겹 교차검증)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "08_imbalance_pr_auc.png", dpi=130)
    plt.close(fig)

    # ── 그림 2: 오경보 비교 ─────────────────────────────
    z = np.load(rep / "oof_scores_2-3.npz")
    y = z["y"]
    fa_05, fa_r80, caught_05, caught_r80 = [], [], [], []
    for i, m in enumerate(z["names"]):
        s = z[f"score_{i}"]
        caught_05.append(int(((s >= 0.5) & (y == 1)).sum()))
        fa_05.append(int(((s >= 0.5) & (y == 0)).sum()))
        c, f = alarms_at_recall(y, s)
        caught_r80.append(c)
        fa_r80.append(f)
    names = [str(n) for n in z["names"]]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.6))
    b1 = ax.bar(x - w / 2, fa_05, w, color="#cfd8dc", edgecolor="#555", label="기준점 0.5로 자를 때")
    b2 = ax.bar(x + w / 2, fa_r80, w, color=[COLORS[n] for n in names], edgecolor="#555",
                label="재현율 80%에 맞춰 기준점을 옮길 때")
    for bars, caught in [(b1, caught_05), (b2, caught_r80)]:
        for b, c in zip(bars, caught):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.08 + 1, f"{int(b.get_height()):,}\n(사기 {c})",
                    ha="center", fontsize=7.5)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(0, max(fa_05 + fa_r80) * 6)
    ax.set_xticks(x)
    ax.set_xticklabels([n + ("\n(점수 폭주)" if n == "클래스 가중치" else "") for n in names])
    ax.set_ylabel("오경보 수 (로그 눈금)")
    ax.set_title(f"오경보 수: 같은 모델도 기준점에 따라 크게 달라진다 (사기 {int(y.sum())}건 중)")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "09_imbalance_false_alarms.png", dpi=130)
    plt.close(fig)
    print("saved 08_imbalance_pr_auc.png, 09_imbalance_false_alarms.png")


if __name__ == "__main__":
    main(sys.argv[1])
