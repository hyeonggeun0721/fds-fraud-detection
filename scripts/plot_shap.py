"""4-2 결과 그래프: SHAP.

사용법: python scripts/plot_shap.py reports data_raw/creditcard.csv
  (12_shap.py가 만든 reports/shap_oof.npz와 shap_results.json이 필요)
  - 17_shap_global.png
      왼쪽  = 특성별 |SHAP| 평균 (전체 거래 vs 사기 거래)
      오른쪽 = 특성 값과 SHAP 값의 관계 (사기에서 중요한 특성 8개, 점 하나 = 거래 하나)
  - 18_shap_cases.png
      사례 4개의 판단 근거 (기준값에서 출발해 특성별로 점수를 올리고/내린 크기)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).parent))
from fds_common import load_data  # noqa: E402

_installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next(
    (f for f in ["AppleGothic", "Malgun Gothic", "Noto Sans CJK KR"] if f in _installed), "sans-serif"
)
plt.rcParams["axes.unicode_minus"] = False

N_NORMAL_SAMPLE = 4000   # 오른쪽 그림에 찍을 정상 거래 수 (전부 찍으면 28만 개라 표본만)


def plot_global(res, npz, X, out):
    glob = res["global"]
    feats = [g["feature"] for g in glob]
    top_all = feats[:12]
    by_fraud = sorted(glob, key=lambda g: -g["mean_abs_fraud"])
    top = list(dict.fromkeys(top_all + [g["feature"] for g in by_fraud[:8]]))   # 전체 상위 12 + 사기 상위 8
    rank = {g["feature"]: g["rank"] for g in glob}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    gmap = {g["feature"]: g for g in glob}
    ys = np.arange(len(top))[::-1]
    ax.barh(ys + 0.2, [gmap[f]["mean_abs_all"] for f in top], height=0.4, color="#2e86ab", label="전체 거래")
    ax.barh(ys - 0.2, [gmap[f]["mean_abs_fraud"] for f in top], height=0.4, color="#e07a5f", label="사기 거래만")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{f} (전체 {rank[f]}위)" for f in top], fontsize=9)
    for lab in ax.get_yticklabels():
        if lab.get_text().startswith("Time "):
            lab.set_color("#c0392b")    # 4-1에서 빼도 성능이 같았던 특성
    ax.set_xlabel("|SHAP| 평균 (원시 점수 단위)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("전체 순위와 '사기에서의' 순위가 다르다")

    ax = axes[1]
    shap, y = npz["shap"], npz["y"]
    cols = list(npz["features"])
    rng = np.random.default_rng(0)
    rows = np.concatenate([np.where(y == 1)[0], rng.choice(np.where(y == 0)[0], N_NORMAL_SAMPLE, replace=False)])
    show = [g["feature"] for g in by_fraud[:8]]
    for i, f in enumerate(show[::-1]):
        j = cols.index(f)
        v = X[f].to_numpy()[rows]
        pct = (np.argsort(np.argsort(X[f].to_numpy())) / len(X))[rows]   # 값의 백분위 (0 = 가장 작음, 1 = 가장 큼)
        jitter = rng.uniform(-0.28, 0.28, len(rows))
        is_f = y[rows] == 1
        ax.scatter(shap[rows[~is_f], j], i + jitter[~is_f], c=pct[~is_f], cmap="coolwarm", s=4, alpha=0.4, lw=0)
        sc = ax.scatter(shap[rows[is_f], j], i + jitter[is_f], c=pct[is_f], cmap="coolwarm", s=14, alpha=0.9,
                        marker="D", edgecolors="k", linewidths=0.3)
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(show[::-1])
    ax.set_xlabel("SHAP 값 (+ = 사기 쪽으로, − = 정상 쪽으로)")
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("특성 값 (파랑 = 작음, 빨강 = 큼)")
    ax.set_title(f"사기에서 중요한 8개: ◆ = 사기 전부, 점 = 정상 표본 {N_NORMAL_SAMPLE:,}건")
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("4-2 SHAP 전체 수준: LightGBM 최종 모델(5겹, 학습 때 안 본 거래만 설명)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "17_shap_global.png", dpi=130)
    plt.close(fig)


def plot_cases(res, out):
    cases = res["cases"]
    thr = res["recall80_threshold_oof"]
    thr_raw = np.log(thr / (1 - thr))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    for ax, c in zip(axes.ravel(), cases):
        names = [f"{f['feature']} = {f['value']:.2f}" for f in c["top_features"]] + ["나머지 특성 합"]
        vals = [f["shap"] for f in c["top_features"]] + [c["rest_shap"]]
        start = c["base_value"]
        lefts, pos = [], start
        for v in vals:
            lefts.append(pos if v >= 0 else pos + v)
            pos += v
        ys = np.arange(len(vals))[::-1]
        ax.barh(ys, np.abs(vals), left=lefts, color=["#e07a5f" if v >= 0 else "#2e86ab" for v in vals], height=0.6)
        for y_, v, l in zip(ys, vals, lefts):
            ax.text(l + abs(v) + 0.15, y_, f"{v:+.2f}", va="center", fontsize=8)
        ax.axvline(start, color="#555", ls=":", lw=1)
        ax.axvline(thr_raw, color="#c0392b", ls="--", lw=1.2)
        ax.axvline(c["raw_score"], color="k", lw=1.6)
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xlim(-16, 13)
        label = "사기" if c["label"] == 1 else "정상"
        ax.set_title(f"{c['case']} (실제 {label}) — 점수 {c['score']:.4g}", fontsize=10)
        ax.set_xlabel("원시 점수 (log-odds)   점선 = 기준값, 빨간 점선 = 사기 80% 기준점, 검은 선 = 최종 점수", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("4-2 SHAP 사례: 기준값(약 −12.7)에서 출발해 특성마다 점수를 올리거나(빨강) 내린다(파랑)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "18_shap_cases.png", dpi=130)
    plt.close(fig)


def main(report_dir: str, csv_path: str) -> None:
    rep = Path(report_dir)
    res = json.loads((rep / "shap_results.json").read_text())
    npz = np.load(rep / "shap_oof.npz", allow_pickle=True)
    X, y = load_data(csv_path)
    assert (npz["y"] == y).all(), "shap_oof.npz와 데이터 순서가 다름"
    out = rep / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plot_global(res, npz, X, out)
    plot_cases(res, out)
    print("saved 17_shap_global.png, 18_shap_cases.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
