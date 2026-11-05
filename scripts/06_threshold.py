"""2-4단계: 기준점(임계값) 정하기 — '과거 데이터로 정한 기준점이 새 데이터에서도 지켜지는가?'

사용법: python scripts/06_threshold.py data_raw/creditcard.csv reports

방법 (조각마다 반복)
  1) 학습 조각 안에서 다시 4겹으로 나눠, 학습 조각 거래마다 '처음 보는 상태'의 점수를 만든다 (안쪽 교차검증)
  2) 그 점수로 운영 목표에 맞는 기준점을 고른다  ← 테스트 조각은 전혀 보지 않음
  3) 학습 조각 전체로 모델을 다시 학습하고, 테스트 조각에 그 기준점을 그대로 적용한다
  4) 목표가 테스트 조각에서도 지켜졌는지 확인한다

비교하는 기준점 정책
  - 고정 0.5        : 아무 고민 없이 0.5 (2-3에서 문제를 본 방식)
  - 재현율 80% 목표 : 사기의 80% 이상을 잡는 기준점 중 가장 높은 것
  - 오경보 예산     : 거래 1만 건당 오경보 0.5건 이하가 되는 기준점 중 가장 낮은 것
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold

from fds_common import LGBM_PARAMS, cv_folds, load_data

RECALL_TARGET = 0.80          # 정책 2: 사기의 80% 이상 잡기
FP_PER_10K_BUDGET = 0.5       # 정책 3: 거래 1만 건당 오경보 0.5건 이하 (데이터 기준 하루 약 7건)
INNER_SPLITS = 4              # 안쪽 교차검증 조각 수


def threshold_for_recall(y, score, target):
    """재현율이 target 이상인 기준점 중 가장 높은 값 (= 오경보를 가장 적게 내면서 목표 달성)."""
    _, recall, thresholds = precision_recall_curve(y, score)
    ok = recall[:-1] >= target            # 마지막 값(재현율 0)에는 대응하는 기준점이 없음
    return float(thresholds[ok].max())


def threshold_for_fp_budget(y, score, per_10k):
    """오경보가 '거래 1만 건당 per_10k건' 이하가 되는 기준점 중 가장 낮은 값 (= 예산 안에서 사기를 최대한 잡음)."""
    allowed = int(np.floor(per_10k * len(y) / 10_000))      # 허용 오경보 수
    normal_desc = np.sort(score[y == 0])[::-1]              # 정상 거래 점수, 높은 순
    # 정상 거래 중 (allowed+1)번째로 높은 점수보다 '아주 조금 더 큰' 값을 기준점으로 → 그 위에는 정상이 allowed건만 남음
    return float(np.nextafter(normal_desc[allowed], np.inf))


def apply_threshold(y, score, threshold):
    """기준점으로 잘랐을 때 실제 결과."""
    alarm = score >= threshold
    tp = int((alarm & (y == 1)).sum())
    fp = int((alarm & (y == 0)).sum())
    return {
        "threshold": threshold,
        "caught": tp,
        "missed": int(y.sum()) - tp,
        "false_alarms": fp,
        "recall": round(tp / int(y.sum()), 4),
        "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "fp_per_10k": round(fp / len(y) * 10_000, 3),
    }


def inner_oof_scores(X_tr, y_tr):
    """학습 조각 안에서 4겹 교차검증 → 학습 조각 거래마다 '자기를 안 본 모델'의 점수."""
    oof = np.zeros(len(y_tr))
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=0)
    for a, b in inner.split(np.zeros(len(y_tr)), y_tr):
        model = LGBMClassifier(**LGBM_PARAMS).fit(X_tr.iloc[a], y_tr[a])
        oof[b] = model.predict_proba(X_tr.iloc[b])[:, 1]
    return oof


def main(csv_path: str, out_dir: str) -> None:
    X, y = load_data(csv_path)
    folds = cv_folds(y)
    print(f"거래 {len(y):,}건 / 사기 {y.sum()}건 / 바깥 {len(folds)}겹 × 안쪽 {INNER_SPLITS}겹", flush=True)

    rows = []
    for k, (train_idx, test_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_te, y_te = X.iloc[test_idx], y[test_idx]

        # 1~2) 학습 조각만 보고 기준점 고르기
        inner = inner_oof_scores(X_tr, y_tr)
        chosen = {
            "고정 0.5": 0.5,
            "재현율 80% 목표": threshold_for_recall(y_tr, inner, RECALL_TARGET),
            "오경보 예산": threshold_for_fp_budget(y_tr, inner, FP_PER_10K_BUDGET),
        }

        # 3) 학습 조각 전체로 다시 학습 → 테스트 조각 점수
        model = LGBMClassifier(**LGBM_PARAMS).fit(X_tr, y_tr)
        test_score = model.predict_proba(X_te)[:, 1]

        # 4) 고른 기준점을 테스트 조각에 그대로 적용
        for policy, thr in chosen.items():
            on_train = apply_threshold(y_tr, inner, thr)    # 고를 때 본 결과
            on_test = apply_threshold(y_te, test_score, thr)  # 새 데이터에서의 실제 결과
            rows.append({"fold": k, "policy": policy, "threshold": thr,
                         **{f"train_{key}": v for key, v in on_train.items() if key != "threshold"},
                         **{f"test_{key}": v for key, v in on_test.items() if key != "threshold"}})
            print(f"조각 {k} | {policy:<9} | 기준점 {thr:.4f} | 테스트 재현율 {on_test['recall']:.3f} "
                  f"| 오경보 {on_test['false_alarms']:>3}건 ({on_test['fp_per_10k']:.2f}/1만건)", flush=True)

    table = pd.DataFrame(rows)
    cols = ["threshold", "test_recall", "test_precision", "test_false_alarms", "test_fp_per_10k", "test_caught", "test_missed"]
    summary = table.groupby("policy", sort=False)[cols].agg(["mean", "std", "min", "max"]).round(4)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]

    # 목표가 테스트 조각에서 지켜진 횟수
    kept = {
        "재현율 80% 목표": int((table.loc[table.policy == "재현율 80% 목표", "test_recall"] >= RECALL_TARGET).sum()),
        "오경보 예산": int((table.loc[table.policy == "오경보 예산", "test_fp_per_10k"] <= FP_PER_10K_BUDGET).sum()),
    }

    result = {
        "targets": {"recall": RECALL_TARGET, "fp_per_10k": FP_PER_10K_BUDGET, "inner_splits": INNER_SPLITS},
        "summary": summary.reset_index().to_dict(orient="records"),
        "target_kept_on_test_folds": kept,
        "totals_over_folds": table.groupby("policy", sort=False)[["test_caught", "test_missed", "test_false_alarms"]].sum().to_dict(orient="index"),
        "folds": rows,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "threshold_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"target_kept_on_test_folds": kept, "totals_over_folds": result["totals_over_folds"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
