"""2-1 보완: 분할 방법(시드)을 바꿔 10번 반복해, 점수가 '우연히' 얼마나 흔들리는지 잰다.

사용법: python scripts/03b_seed_stability.py data_raw/creditcard.csv reports

알고 싶은 것
  1) 같은 모델이라도 테스트에 어떤 사기가 들어가느냐에 따라 점수가 얼마나 달라지나? (= 흔들림 범위)
  2) 2-1에서 본 '중복 제거 후 점수 하락'은 이 흔들림보다 큰가? (= 진짜 효과인가, 우연인가)
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = range(10)  # 시드 0~9: 데이터를 10가지 서로 다른 방식으로 나눈다
MODELS = {"로지스틱 회귀": None, "로지스틱 회귀 + 클래스 가중치": "balanced"}


def precision_at_recall(y_true, score, target=0.80):
    precision, recall, _ = precision_recall_curve(y_true, score)
    return float(precision[recall >= target].max())


def run_once(data, seed, class_weight):
    """seed로 나누고, 1단계와 같은 모델을 학습해 두 지표를 돌려준다."""
    train, test = train_test_split(data, test_size=0.2, stratify=data["Class"], random_state=seed)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=0),
    )
    model.fit(train.drop(columns=["Class"]), train["Class"])
    score = model.predict_proba(test.drop(columns=["Class"]))[:, 1]
    return {
        "pr_auc": average_precision_score(test["Class"], score),
        "precision_at_recall80": precision_at_recall(test["Class"], score),
        "test_fraud": int(test["Class"].sum()),
    }


def main(csv_path: str, out_dir: str) -> None:
    df = pd.read_csv(csv_path)
    datasets = {"중복 유지": df, "중복 제거": df.drop_duplicates()}

    runs = []
    for data_name, data in datasets.items():
        for model_name, cw in MODELS.items():
            for seed in SEEDS:
                r = run_once(data, seed, cw)
                runs.append({"data": data_name, "model": model_name, "seed": seed, **r})
                print(f"{data_name} | {model_name} | seed {seed} | PR-AUC {r['pr_auc']:.3f}", flush=True)

    # 조합별로 10번 결과를 요약: 평균, 표준편차(흔들림 크기), 최솟값, 최댓값
    table = pd.DataFrame(runs)
    summary = (
        table.groupby(["data", "model"])[["pr_auc", "precision_at_recall80"]]
        .agg(["mean", "std", "min", "max"])
        .round(3)
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().to_dict(orient="records")

    result = {"seeds": list(SEEDS), "summary": summary, "runs": [{**r, "pr_auc": round(r["pr_auc"], 4),
              "precision_at_recall80": round(r["precision_at_recall80"], 4)} for r in runs]}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "seed_stability.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
