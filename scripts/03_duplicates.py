"""2-1단계: 중복 거래가 평가 결과를 부풀리는지 확인한다.

사용법: python scripts/03_duplicates.py data_raw/creditcard.csv reports

질문 3가지
  1) 완전히 똑같은 거래는 몇 건이고, 그중 사기는 몇 건인가?
  2) 1단계처럼 나누면, 테스트 거래 중 '학습 데이터에 똑같은 복사본이 있는' 거래가 몇 건인가? (= 누수)
  3) 중복을 지우고 나눴을 때 성능(PR-AUC)이 얼마나 달라지는가?
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

SEED = 42  # 1단계와 같은 값 → 같은 방식으로 나뉨


def precision_at_recall(y_true, score, target=0.80):
    """사기의 target(80%)만큼 잡도록 기준점을 낮췄을 때, 경보 중 진짜 사기 비율."""
    precision, recall, _ = precision_recall_curve(y_true, score)
    return float(precision[recall >= target].max())


def fit_and_score(train, test, class_weight):
    """1단계와 같은 모델(스케일 맞추기 + 로지스틱 회귀)을 학습하고 두 지표를 돌려준다."""
    X_tr, y_tr = train.drop(columns=["Class"]), train["Class"]
    X_te, y_te = test.drop(columns=["Class"]), test["Class"]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=SEED),
    )
    model.fit(X_tr, y_tr)
    score = model.predict_proba(X_te)[:, 1]  # 각 거래가 사기일 확률
    return {
        "pr_auc": round(average_precision_score(y_te, score), 3),
        "precision_at_recall80": round(precision_at_recall(y_te, score), 3),
        "test_rows": int(len(test)),
        "test_fraud": int(y_te.sum()),
    }


def main(csv_path: str, out_dir: str) -> None:
    df = pd.read_csv(csv_path)
    features = [c for c in df.columns if c != "Class"]

    # ── 질문 1: 중복 현황 ─────────────────────────────────────
    extra_copies = df.duplicated()                # 첫 번째는 원본으로 보고, 두 번째부터 '여분 복사본'
    in_dup_group = df.duplicated(keep=False)      # 복사본이 하나라도 있는 행 전부
    same_features = df.duplicated(subset=features, keep=False)  # 라벨(Class)만 빼고 똑같은 행
    label_conflict_groups = int(
        df[same_features].groupby(features)["Class"].nunique().gt(1).sum()
    )  # 거래 내용은 똑같은데 하나는 정상, 하나는 사기로 표시된 묶음 수

    duplicates = {
        "extra_copies": int(extra_copies.sum()),
        "extra_copies_fraud": int(df.loc[extra_copies, "Class"].sum()),
        "rows_in_duplicate_groups": int(in_dup_group.sum()),
        "label_conflict_groups": label_conflict_groups,
    }

    # ── 질문 2: 1단계 방식 분할에서의 누수 ────────────────────
    train, test = train_test_split(df, test_size=0.2, stratify=df["Class"], random_state=SEED)
    leaked = test.merge(train.drop_duplicates(), how="inner", on=list(df.columns))
    leakage = {
        "test_rows_with_copy_in_train": int(len(leaked)),
        "of_which_fraud": int(leaked["Class"].sum()),
    }

    # ── 질문 3: 중복 유지 vs 제거 후 성능 비교 ─────────────────
    dedup = df.drop_duplicates()                  # 여분 복사본 제거
    d_train, d_test = train_test_split(dedup, test_size=0.2, stratify=dedup["Class"], random_state=SEED)

    comparison = []
    for data_name, tr, te in [("중복 유지(1단계 방식)", train, test), ("중복 제거 후 분할", d_train, d_test)]:
        for model_name, cw in [("로지스틱 회귀", None), ("로지스틱 회귀 + 클래스 가중치", "balanced")]:
            comparison.append({"data": data_name, "model": model_name, **fit_and_score(tr, te, cw)})

    result = {"duplicates": duplicates, "leakage": leakage, "comparison": comparison}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "dedup_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
