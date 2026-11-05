"""2-2단계: 트리 모델(LightGBM)이 로지스틱 회귀보다 얼마나 나은가? (5겹 교차검증)

사용법: python scripts/04_tree_model.py data_raw/creditcard.csv reports

- 데이터: 중복 제거본 (2-1 결정)
- 평가: 5겹 교차검증, 두 모델은 같은 조각끼리 짝지어 비교 (2-1 보완에서 배운 점)
- 불균형 처리(가중치·샘플링)는 아직 하지 않는다 → 2-3에서 따로 비교
- v2(2026-11-04): LightGBM에 min_child_weight=1 추가. 1차 실행에서 원시 점수가 폭주해(최대 ~25,000) 결과가 무효였음
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fds_common import cv_folds, evaluate, load_data

# 모델을 '만드는 방법'을 저장한다. 조각마다 새 모델을 처음부터 학습해야 하기 때문.
MODELS = {
    "로지스틱 회귀": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=0)
    ),
    "LightGBM": lambda: LGBMClassifier(
        n_estimators=300,       # 나무 300그루를 차례로 만든다
        learning_rate=0.05,     # 나무 하나가 이전 실수를 고치는 정도(작을수록 천천히, 꼼꼼히)
        num_leaves=31,          # 나무 하나의 최대 잎(규칙) 수 = 복잡도
        subsample=0.8,          # 나무마다 거래의 80%만 무작위로 보고 학습 → 과적합 방지
        subsample_freq=1,       # 위 무작위 추출을 나무마다 새로 한다
        colsample_bytree=0.8,   # 나무마다 변수의 80%만 사용 → 과적합 방지
        min_child_weight=1,     # 잎 하나에 '확신 정도(헤시안) 합'이 최소 1은 있어야 잎을 만든다
                                # 기본값 0.001이면 거의 0으로 나누는 잎이 생겨 점수가 폭주함 (2-2 1차 실행에서 확인)
        random_state=0,
        verbose=-1,             # 학습 중 메시지 끄기
    ),
}


def main(csv_path: str, out_dir: str) -> None:
    X, y = load_data(csv_path)
    folds = cv_folds(y)
    print(f"거래 {len(y):,}건 / 사기 {y.sum()}건 / {len(folds)}겹", flush=True)

    oof = {name: np.zeros(len(y)) for name in MODELS}  # 거래마다 '자기가 테스트였을 때' 받은 점수
    rows = []
    for k, (train_idx, test_idx) in enumerate(folds):
        for name, make_model in MODELS.items():
            model = make_model()
            model.fit(X.iloc[train_idx], y[train_idx])
            score = model.predict_proba(X.iloc[test_idx])[:, 1]
            oof[name][test_idx] = score
            rows.append({"fold": k, "model": name, **evaluate(y[test_idx], score)})
            print(f"조각 {k} | {name:<8} | PR-AUC {rows[-1]['pr_auc']:.3f}", flush=True)

    table = pd.DataFrame(rows)

    # (1) 모델별 5조각 요약: 평균 ± 표준편차
    summary = table.groupby("model")[["pr_auc", "precision_at_recall80"]].agg(["mean", "std", "min", "max"]).round(4)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]

    # (2) 같은 조각끼리 짝지은 차이: LightGBM − 로지스틱 회귀
    wide = table.pivot(index="fold", columns="model", values=["pr_auc", "precision_at_recall80"])
    paired = {}
    for metric in ["pr_auc", "precision_at_recall80"]:
        diff = wide[(metric, "LightGBM")] - wide[(metric, "로지스틱 회귀")]
        paired[metric] = {
            "mean_diff": round(float(diff.mean()), 4),
            "std_diff": round(float(diff.std()), 4),
            "lightgbm_wins": int((diff > 0).sum()),
            "per_fold": [round(float(v), 4) for v in diff],
        }

    # (3) 5조각 점수를 모두 모아 전체 데이터 기준으로 한 번 더 계산
    overall = {name: evaluate(y, s) for name, s in oof.items()}

    result = {
        "data": {"rows": int(len(y)), "fraud": int(y.sum())},
        "summary": summary.reset_index().to_dict(orient="records"),
        "paired_lightgbm_minus_logistic": paired,
        "overall_out_of_fold": overall,
        "folds": rows,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tree_model_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    np.savez_compressed(out / "oof_scores_2-2.npz", y=y, **{f"score_{i}": oof[n] for i, n in enumerate(MODELS)},
                        names=np.array(list(MODELS)))  # 그래프·2-4(기준점 정하기)용
    print(json.dumps({"summary": result["summary"], "paired": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
