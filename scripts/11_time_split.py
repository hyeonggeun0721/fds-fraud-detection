"""4-1: 과거로 학습해서 미래를 맞힐 때도 LightGBM은 0.850인가? (시간순 전진 검증)

사용법: python scripts/11_time_split.py data_raw/creditcard.csv reports

- 거래를 Time(첫 거래로부터 지난 초) 순서로 정렬해 거래 수가 비슷한 5개 시간 구간으로 나눈다
- 전진 검증: 구간 0으로 학습 → 구간 1 테스트, 구간 0~1 → 구간 2, ... 구간 0~3 → 구간 4 (4번)
- Time 특성을 다루는 세 가지 방법을 비교한다
    A. 그대로     : Time(초)을 그대로 넣음 → 테스트 구간의 Time 값은 학습 때 본 적 없는 범위
    B. 제거       : Time을 빼고 V1~V28, Amount만 사용
    C. 하루 중 시각: Time을 0~24시로 바꿔 넣음 (이틀째 같은 시각은 같은 값)
- 대조군: 각 전진 검증과 학습·테스트 크기가 같은 '무작위' 분할(방법 A) → 학습량 차이가 아니라 '시간 순서' 때문에 생긴 차이만 보기 위함
- 4-1b 대조군 '같은 시험지 무작위': 테스트 구간은 시간순과 똑같이 두고, 학습만 다른 모든 구간(과거+미래)에서 같은 크기로 뽑음
  (방법 A·B, 3회 반복) → 테스트 구간의 사기 비율 차이를 없애고 '과거만 봤다'는 효과만 남김
  ※ 테스트 구간 4는 다른 구간이 모두 과거라 학습 데이터가 시간순과 같은 거래가 됨(비교 의미 없음, 기록만)
- LightGBM만 쓴다 (PyTorch와 함께 불러오면 libomp 충돌)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from fds_common import LGBM_PARAMS, evaluate, load_data

N_BLOCKS = 5
RANDOM_SEED = 0     # 무작위 대조군 분할용
N_REPEATS_MIXED = 3  # 4-1b 같은 시험지 대조군 반복 횟수 (시드 = [테스트 구간, 반복 번호])


def top_k_hits(y_true, score):
    k = int(y_true.sum())
    return int(y_true[np.argsort(score)[::-1][:k]].sum())


def time_blocks(time, n_blocks=N_BLOCKS):
    """Time 순서로 거래 수가 비슷한 n개 구간 번호(0~n-1)를 돌려준다.

    거래 수 기준으로 경계 시각을 정한 뒤 '시각'으로 나누므로, 같은 초에 일어난 거래는 반드시 같은 구간에 들어간다.
    """
    cuts = np.quantile(time, np.linspace(0, 1, n_blocks + 1)[1:-1])   # 경계 시각 4개
    return np.searchsorted(cuts, time, side="right")               # 경계보다 같거나 크면 다음 구간


def make_features(X, variant):
    """Time 처리 방법(A/B/C)에 맞춰 특성 표를 만든다."""
    if variant == "A":
        return X
    if variant == "B":
        return X.drop(columns=["Time"])
    if variant == "C":
        out = X.drop(columns=["Time"]).copy()
        out["Hour"] = (X["Time"] % 86400) / 3600     # 하루 = 86,400초 → 0.0~24.0시
        return out
    raise ValueError(variant)


VARIANT_NAMES = {"A": "Time 그대로", "B": "Time 제거", "C": "하루 중 시각"}


def fit_score(X_tr, y_tr, X_te):
    model = LGBMClassifier(**LGBM_PARAMS).fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def main(csv_path: str, out_dir: str) -> None:
    X, y = load_data(csv_path)
    block = time_blocks(X["Time"].to_numpy())

    print(f"거래 {len(y):,}건 / 사기 {y.sum()}건 / 시간 구간 {N_BLOCKS}개")
    blocks_info = []
    for b in range(N_BLOCKS):
        m = block == b
        t = X["Time"][m]
        blocks_info.append({"block": b, "n": int(m.sum()), "fraud": int(y[m].sum()),
                            "fraud_rate": round(float(y[m].mean()), 5),
                            "time_from_h": round(float(t.min()) / 3600, 1), "time_to_h": round(float(t.max()) / 3600, 1)})
        r = blocks_info[-1]
        print(f"  구간 {b}: {r['time_from_h']:>5.1f}~{r['time_to_h']:>5.1f}시간 | 거래 {r['n']:,} | 사기 {r['fraud']} ({r['fraud_rate']:.3%})")

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for test_b in range(1, N_BLOCKS):
        train_idx = np.where(block < test_b)[0]
        test_idx = np.where(block == test_b)[0]
        y_te = y[test_idx]

        for v in ["A", "B", "C"]:
            F = make_features(X, v)
            score = fit_score(F.iloc[train_idx], y[train_idx], F.iloc[test_idx])
            rows.append({"test_block": test_b, "split": "시간순", "variant": v, "n_train": len(train_idx),
                         "train_fraud": int(y[train_idx].sum()), **evaluate(y_te, score),
                         "top_k_hits": top_k_hits(y_te, score)})

        # 대조군: 같은 크기의 무작위 학습·테스트 (방법 A)
        perm = rng.permutation(len(y))
        r_train, r_test = perm[:len(train_idx)], perm[len(train_idx):len(train_idx) + len(test_idx)]
        score = fit_score(X.iloc[r_train], y[r_train], X.iloc[r_test])
        rows.append({"test_block": test_b, "split": "무작위(같은 크기)", "variant": "A", "n_train": len(r_train),
                     "train_fraud": int(y[r_train].sum()), **evaluate(y[r_test], score),
                     "top_k_hits": top_k_hits(y[r_test], score)})

        # 4-1b 대조군: 테스트 구간은 그대로, 학습은 '다른 모든 구간(과거+미래)'에서 같은 크기로 무작위 추출 (3회 반복)
        #   → 시험지가 같으므로 남는 차이는 '과거만 봤다'는 것뿐
        pool = np.where(block != test_b)[0]
        for rep in range(N_REPEATS_MIXED):
            pick = np.random.default_rng([test_b, rep]).choice(pool, size=len(train_idx), replace=False)
            for v in ["A", "B"]:
                F = make_features(X, v)
                score = fit_score(F.iloc[pick], y[pick], F.iloc[test_idx])
                rows.append({"test_block": test_b, "split": "같은 시험지 무작위", "variant": v, "repeat": rep,
                             "n_train": len(pick), "train_fraud": int(y[pick].sum()),
                             "train_future_share": round(float((block[pick] > test_b).mean()), 3),
                             **evaluate(y_te, score), "top_k_hits": top_k_hits(y_te, score)})

        for r in rows[-(4 + 2 * N_REPEATS_MIXED):]:
            name = r["split"] + ("" if r["split"] == "무작위(같은 크기)" else f" {r['variant']}.{VARIANT_NAMES[r['variant']]}")
            if "repeat" in r:
                name += f" #{r['repeat']}"
            print(f"테스트 구간 {test_b} | {name:<18} | 학습 사기 {r['train_fraud']:>3} | 테스트 사기 {r['test_fraud']:>3} | "
                  f"PR-AUC {r['pr_auc']:.3f} | 상위 k 중 사기 {r['top_k_hits']}", flush=True)

    table = pd.DataFrame(rows)
    summary = (table.groupby(["split", "variant"], sort=False)[["pr_auc", "precision_at_recall80", "top_k_hits"]]
               .agg(["mean", "std"]).round(4))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    result = {"n_blocks": N_BLOCKS, "variant_names": VARIANT_NAMES, "blocks": blocks_info,
              "summary": summary.reset_index().to_dict(orient="records"), "folds": rows}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "time_split_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(summary.to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
