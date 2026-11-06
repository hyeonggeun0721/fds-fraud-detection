"""2-2단계부터 여러 스크립트가 함께 쓰는 함수 모음.

같은 데이터, 같은 5겹 분할, 같은 지표를 모든 단계에서 똑같이 쓰기 위해 한곳에 모았다.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 5   # 5겹 교차검증
CV_SEED = 42   # 5조각으로 나누는 방법을 고정 → 모든 모델이 같은 시험지로 평가됨


def load_data(csv_path):
    """데이터를 읽고 중복 거래를 지운 뒤, 문제(X)와 정답(y)으로 나눈다. (2-1 결정: 중복 제거)"""
    df = pd.read_csv(csv_path).drop_duplicates().reset_index(drop=True)
    X = df.drop(columns=["Class"])
    y = df["Class"].to_numpy()
    return X, y


def cv_folds(y):
    """5겹 분할 목록을 돌려준다. 각 원소는 (학습용 행 번호, 테스트용 행 번호)."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    return list(skf.split(np.zeros(len(y)), y))


def precision_at_recall(y_true, score, target=0.80):
    """사기의 target(80%)을 잡도록 기준점을 낮췄을 때, 경보 중 진짜 사기 비율."""
    precision, recall, _ = precision_recall_curve(y_true, score)
    return float(precision[recall >= target].max())


def evaluate(y_true, score):
    """한 조각의 테스트 결과를 두 지표로 요약한다."""
    return {
        "pr_auc": round(float(average_precision_score(y_true, score)), 4),
        "precision_at_recall80": round(precision_at_recall(y_true, score), 4),
        "test_fraud": int(y_true.sum()),
    }


# 2-2에서 확정한 LightGBM 설정 (2-3부터 공통으로 사용)
LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_child_weight=1,   # 2-2 1차 실행의 점수 폭주를 막은 설정
    random_state=0,
    verbose=-1,
)


# 3-3a에서 사용: 학습 조각의 사기 라벨 중 일부만 '확인된 것'으로 남긴다
LABEL_FRACS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]


def hide_labels(y_train, frac, seed):
    """사기 라벨을 frac 비율만 남기고, 나머지 사기는 0(정상)으로 바꾼 '관측된 라벨'을 돌려준다.

    - 확인 안 된 사기는 실제 현장처럼 정상으로 섞여 있다고 가정한다.
    - 같은 seed면 사기 순서를 한 번만 섞고 앞에서부터 남기므로, 10%에 남은 사기는 25%·50%에도 모두 포함된다.
    - LightGBM 스크립트와 오토인코더 스크립트가 같은 seed를 쓰면 정확히 같은 사기가 숨겨진다.
    """
    y_obs = y_train.copy()
    fraud_pos = np.where(y_train == 1)[0]
    order = np.random.default_rng(seed).permutation(fraud_pos)
    n_keep = max(1, int(round(frac * len(fraud_pos))))
    y_obs[order[n_keep:]] = 0
    return y_obs
