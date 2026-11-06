"""4-2: LightGBM은 무엇을 보고 사기라고 판단하나? (SHAP, 5겹 교차검증 최종 모델)

사용법: python scripts/12_shap.py data_raw/creditcard.csv reports

- 설명할 모델: 2-2에서 확정한 최종 모델 그대로 (LGBM_PARAMS, Time 포함, 5겹)
- SHAP 값은 LightGBM 내장 기능 predict(..., pred_contrib=True)로 계산 (TreeSHAP, 추가 설치 없음)
    · 거래 하나의 원시 점수(log-odds) = 기준값 + 특성별 SHAP 값의 합
    · 각 조각의 모델은 자기 테스트 조각만 설명 → 모든 거래가 '학습 때 안 본 모델'로 한 번씩 설명됨
- 전체 수준: 특성별 |SHAP| 평균 (전체 거래 / 사기 거래)
- 거래 하나 수준: 가장 확실히 잡은 사기 / 가장 크게 놓친 사기 / 가장 강한 오경보 + 재현율 80% 기준점 근처의 사례
- SHAP 값 전체는 reports/shap_oof.npz에 저장 (그래프용, git 제외)
- LightGBM만 쓴다 (PyTorch와 함께 불러오면 libomp 충돌)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from fds_common import LGBM_PARAMS, cv_folds, evaluate, load_data

TOP_N = 8   # 사례마다 기록할 영향 큰 특성 수


def explain_case(name, i, X, y, score, raw, shap, base, fold_of):
    """거래 i의 판단 근거: 기준값에서 출발해 어떤 특성이 점수를 얼마나 올리고/내렸는지."""
    order = np.argsort(-np.abs(shap[i]))[:TOP_N]
    return {
        "case": name, "row": int(i), "fold": int(fold_of[i]), "label": int(y[i]),
        "score": round(float(score[i]), 6), "raw_score": round(float(raw[i]), 4), "base_value": round(float(base[i]), 4),
        "top_features": [{"feature": X.columns[j], "value": round(float(X.iat[i, j]), 4),
                          "shap": round(float(shap[i, j]), 4)} for j in order],
        "rest_shap": round(float(shap[i].sum() - shap[i, order].sum()), 4),   # 나머지 특성 합
    }


def main(csv_path: str, out_dir: str) -> None:
    X, y = load_data(csv_path)
    folds = cv_folds(y)
    n, d = X.shape
    shap = np.zeros((n, d), dtype=np.float64)   # 특성별 SHAP 값
    base = np.zeros(n)                          # 기준값 (모델이 아무것도 안 볼 때의 원시 점수)
    raw = np.zeros(n)                           # 원시 점수(log-odds)
    score = np.zeros(n)                         # 확률 점수
    fold_of = np.zeros(n, dtype=int)
    fold_rows = []

    for k, (train_idx, test_idx) in enumerate(folds):
        model = LGBMClassifier(**LGBM_PARAMS).fit(X.iloc[train_idx], y[train_idx])
        X_te = X.iloc[test_idx]
        contrib = model.predict(X_te, pred_contrib=True)          # (거래 수, 특성 수 + 1), 마지막 열 = 기준값
        shap[test_idx] = contrib[:, :d]
        base[test_idx] = contrib[:, d]
        raw[test_idx] = model.predict(X_te, raw_score=True)
        score[test_idx] = model.predict_proba(X_te)[:, 1]
        fold_of[test_idx] = k
        fold_rows.append({"fold": k, **evaluate(y[test_idx], score[test_idx])})
        print(f"조각 {k} | PR-AUC {fold_rows[-1]['pr_auc']:.3f} | 기준값 {contrib[0, d]:.3f}", flush=True)

    # 검산: 기준값 + SHAP 합 = 원시 점수 인가?
    additivity_err = float(np.abs(base + shap.sum(axis=1) - raw).max())
    print(f"검산(기준값 + SHAP 합 − 원시 점수) 최대 오차: {additivity_err:.2e}")

    # 전체 수준 중요도
    fraud = y == 1
    glob = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_all": np.abs(shap).mean(axis=0),           # 전체 거래에서 평균적으로 점수를 얼마나 움직이나
        "mean_abs_fraud": np.abs(shap[fraud]).mean(axis=0),  # 사기 거래에서
        "mean_fraud": shap[fraud].mean(axis=0),              # 사기 거래에서 평균적으로 올리나(+)/내리나(−)
        "mean_normal": shap[~fraud].mean(axis=0),            # 정상 거래에서
    }).sort_values("mean_abs_all", ascending=False).reset_index(drop=True)
    glob["rank"] = np.arange(1, d + 1)
    print(glob.head(12).round(4).to_string(index=False))

    # 거래 하나 수준 사례
    fraud_idx, normal_idx = np.where(fraud)[0], np.where(~fraud)[0]
    k_rank = np.argsort(-score)                       # 점수 높은 순
    cut = np.sort(score[fraud])[::-1][int(np.ceil(0.8 * fraud.sum())) - 1]   # 사기 80%를 잡는 기준점(전체 OOF 기준)
    near_miss = fraud_idx[score[fraud_idx] < cut]
    cases = [
        explain_case("가장 확실히 잡은 사기", fraud_idx[np.argmax(score[fraud_idx])], X, y, score, raw, shap, base, fold_of),
        explain_case("기준점 바로 아래로 놓친 사기", near_miss[np.argmax(score[near_miss])], X, y, score, raw, shap, base, fold_of),
        explain_case("가장 크게 놓친 사기", fraud_idx[np.argmin(score[fraud_idx])], X, y, score, raw, shap, base, fold_of),
        explain_case("가장 강한 오경보", normal_idx[np.argmax(score[normal_idx])], X, y, score, raw, shap, base, fold_of),
    ]
    for c in cases:
        top = ", ".join(f"{f['feature']}={f['value']:.2f}({f['shap']:+.2f})" for f in c["top_features"][:4])
        print(f"{c['case']}: 점수 {c['score']:.4f} | {top}")

    result = {
        "model": "LightGBM (LGBM_PARAMS, Time 포함, 5겹)", "method": "LightGBM pred_contrib (TreeSHAP), 원시 점수(log-odds) 단위",
        "additivity_max_error": additivity_err, "recall80_threshold_oof": round(float(cut), 6),
        "folds": fold_rows, "global": glob.round(5).to_dict(orient="records"), "cases": cases,
        "frauds_above_threshold": int((score[fraud_idx] >= cut).sum()), "normals_above_threshold": int((score[normal_idx] >= cut).sum()),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "shap_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    np.savez_compressed(out / "shap_oof.npz", shap=shap.astype(np.float32), base=base.astype(np.float32),
                        score=score.astype(np.float32), y=y, features=np.array(X.columns))
    print(f"saved shap_results.json, shap_oof.npz | 사기 80% 기준점 {cut:.4f}: 사기 {result['frauds_above_threshold']} / 오경보 {result['normals_above_threshold']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
