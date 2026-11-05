"""두 결과 폴더의 단계별 핵심 수치를 나란히 비교한다 (환경을 바꾼 뒤 결과가 그대로인지 확인용).

사용법: python scripts/compare_results.py <기존 reports 폴더> <새 reports 폴더>
"""
import json
import sys
from pathlib import Path


def load(folder, name):
    p = Path(folder) / name
    return json.loads(p.read_text()) if p.exists() else None


def rows_baseline(d):
    return {r["model"]: r["pr_auc"] for r in d["results"]}


def rows_summary(d, key, metric="pr_auc_mean"):
    return {f"{r[key[0]]} | {r[key[1]]}" if isinstance(key, tuple) else r[key]: r[metric] for r in d["summary"]}


CHECKS = [
    ("1단계 기본 모델 PR-AUC", "baseline_results.json", rows_baseline),
    ("2-1 중복 제거 비교 PR-AUC", "dedup_results.json",
     lambda d: {f"{r['data']} | {r['model']}": r["pr_auc"] for r in d["comparison"]}),
    ("2-1 시드 반복 PR-AUC 평균", "seed_stability.json", lambda d: rows_summary(d, ("data", "model"))),
    ("2-2 LightGBM vs 로지스틱 PR-AUC 평균", "tree_model_results.json", lambda d: rows_summary(d, "model")),
    ("2-3 불균형 처리 PR-AUC 평균", "imbalance_results.json", lambda d: rows_summary(d, "method")),
    ("2-4 기준점: 테스트 재현율 평균", "threshold_results.json", lambda d: rows_summary(d, "policy", "test_recall_mean")),
    ("2-4 보완: 목표 지킨 조각 수", "threshold_margin_results.json",
     lambda d: {f"{r['policy']} | {r['setting']}": r["kept"] for r in d["summary"]}),
    ("3-1 Isolation Forest PR-AUC 평균", "isolation_forest_results.json", lambda d: rows_summary(d, "method")),
]


def main(old_dir, new_dir):
    worst = 0.0
    for title, fname, extract in CHECKS:
        old, new = load(old_dir, fname), load(new_dir, fname)
        print(f"\n## {title}  ({fname})")
        if old is None or new is None:
            print("   (한쪽 결과 없음)")
            continue
        a, b = extract(old), extract(new)
        for k in a:
            if k in b:
                diff = b[k] - a[k]
                worst = max(worst, abs(diff)) if "목표" not in title else worst
                flag = "" if abs(diff) < 0.01 else "  ← 0.01 이상 차이"
                print(f"   {k:<45} 기존 {a[k]:>8.4f}  새 {b[k]:>8.4f}  차이 {diff:+.4f}{flag}")
    print(f"\n가장 큰 차이(조각 수 제외): {worst:.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
