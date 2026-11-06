# FDS — 카드 거래 이상거래탐지

공개 카드 거래 데이터로 사기 거래를 탐지하는 모델을 단계별로 만들고, **"점수가 정말 믿을 만한가"**를 하나씩 검증한 프로젝트입니다.
모든 실험 과정·결정·실패 기록은 [`reports/PROJECT_LOG.md`](reports/PROJECT_LOG.md)에 있습니다.

## 한눈에 보는 결과

| 항목 | 결과 |
|---|---|
| 최종 모델 | LightGBM (`min_child_weight=1`, 불균형 처리 없음) |
| PR-AUC (5겹 교차검증) | **0.850 ± 0.013** (로지스틱 회귀 0.742 ± 0.022, 같은 조각끼리 비교 시 5/5 우세) |
| 사기 80%를 잡을 때 오경보 | 로지스틱 회귀 138건 → **LightGBM 22건** (사기 473건 기준) |
| 기준점 | 운영 목표(재현율 80% 또는 오경보 예산)로 정함. 목표는 평균적으로 지켜지지만 조각별 오차가 큼 |
| 비지도 이상탐지 | 오토인코더(정상만 학습) 0.474, Isolation Forest 0.139 — 모두 LightGBM에 크게 못 미침 |
| 라벨이 적을 때 | 확인된 사기 **8건(2%)**만으로 LightGBM 0.514 > 깨끗한 정상으로 학습한 오토인코더 0.474 |
| 과거로 미래 예측 | 같은 시험지로 비교하면 '과거만 학습'의 손해 ≈ 0 (0.782 vs 0.783). 처음 본 차이(−0.05)는 테스트 구간 차이 |
| 판단 근거 (SHAP) | 사기 판정은 V14가 압도(사기 거래 평균 +6.3). 가장 강한 오경보는 잡은 사기와 근거가 같음 |
| DB 연동 운영 (PostgreSQL) | 최소권한 채점 계정이 새 구간 56,746건 채점 → 경보 56건(사기 53, 오경보 3), 재현율 71.6%. 재현율 80%면 오경보 29건, 90%면 1,220건 |

![정밀도-재현율 곡선](reports/figures/06_pr_curve_tree_model.png)

## 데이터

- Kaggle **Credit Card Fraud Detection** (`mlg-ulb/creditcardfraud`, ULB Machine Learning Group): 2013년 9월 유럽 카드 거래 2일치
- 284,807건 중 사기 492건(0.17%). 변수 V1~V28은 원 제공자가 개인정보 보호를 위해 PCA로 변환한 값, 그 외 `Time`, `Amount`
- 완전히 똑같은 거래 1,081건을 제거해 **283,726건(사기 473건)**으로 사용
- 데이터 파일은 저장소에 포함하지 않습니다. 라이선스는 Kaggle 데이터셋 페이지(Open Database License)를 따릅니다

## 평가 방법

- **지표**: 사기가 0.17%뿐이라 정확도는 의미가 없음(전부 '정상'이라 해도 99.83%). **PR-AUC**와 **재현율 80%일 때 정밀도**를 사용
- **5겹 교차검증**(StratifiedKFold, seed 42): 분할 한 번의 점수는 시험지 운에 따라 PR-AUC가 최대 0.16까지 흔들려서, 모든 비교를 5조각 평균 ± 표준편차로 함
- **같은 조각끼리 짝지은 비교**: 점수 자체는 흔들려도 같은 시험지에서의 모델 차이는 안정적이라, 모델 비교는 조각별 차이와 우세 횟수로 판단

## 단계별 결과

| 단계 | 질문 | 결론 | 그림 |
|---|---|---|---|
| 1 | 데이터는 어떻게 생겼고, 정확도는 왜 쓸모없나? | 사기 0.17%, 결측 없음. 아무것도 안 잡아도 정확도 99.83% → PR-AUC 사용 | 01~03 |
| 2-1 | 중복 거래가 점수를 부풀리나? | 테스트 사기의 약 8%가 학습에 똑같은 복사본 존재 → 중복 제거. 점수는 시험지 운으로 크게 흔들림 → 5겹 교차검증 채택 | 04, 05 |
| 2-2 | 트리 모델은 얼마나 나은가? | LightGBM 0.850 vs 로지스틱 0.742, 5/5 우세. 단, 기본 설정에서는 점수 폭주로 0.464까지 무너짐 → `min_child_weight=1`로 해결 | 06, 07 |
| 2-3 | 불균형 처리(가중치·언더샘플링·SMOTE)가 도움이 되나? | 순위(PR-AUC)는 개선 없음(SMOTE ≈ 처리 없음, 언더샘플링 하락, 가중치는 점수 폭주). 대신 점수 크기를 크게 바꿈 → 처리 없음 채택 | 08, 09 |
| 2-4 | 기준점은 어떻게 정하나? 새 데이터에서도 지켜지나? | 학습 조각만으로 고른 기준점은 목표를 평균적으로 지키지만 조각별로 3/5만 달성 | 10 |
| 2-4 보완 | 여유를 두면? | 재현율 85%로 고르면 15/15 지키지만 오경보 10.4배('절벽'), 오경보 예산은 완만한 맞교환 | 11 |
| 3-1 | 라벨 없이(Isolation Forest) 사기를 찾을 수 있나? | PR-AUC 0.139(무작위의 약 80배)지만 LightGBM 0.850에 크게 못 미침. 고액 정상 거래를 '이상'으로 잡음(특이함 ≠ 사기) | 12 |
| 3-2 | 정상만 배운 오토인코더는? | PR-AUC 0.424로 IF의 약 3배(5/5 우세). 단, 사기가 0.17%만 섞여도 0.049로 붕괴. ROC-AUC로는 IF가 더 높아 순위가 뒤집힘 | 13 |
| 3-3b | 오토인코더를 더 오래 학습하면? | 50에폭 0.474(5/5 개선), 100에폭은 조각별로 엇갈림. 정상 복원 오차는 계속 줄어도 탐지는 나아지지 않음(복원 ≠ 탐지) | 14 |
| 3-3a | 확인된 사기 라벨이 적으면? | LightGBM은 라벨 2%(8건)에서도 0.514로 30/30 우세. 오토인코더는 확인 안 된 사기가 섞이자 0.07 이하로 붕괴 | 15 |
| 4-1 | 과거로 학습해 미래를 맞히면? | 시간순 전진 검증 0.788 vs 같은 크기 무작위 0.854. Time 처리(그대로·제거·하루 중 시각)는 차이 없음 | 16 |
| 4-1b | 그 차이는 정말 시간 순서 때문인가? | 테스트 구간을 똑같이 맞추면 차이 ≈ 0(0.782 vs 0.783) → 차이는 구간마다 다른 사기 비율·난이도 때문. 이틀치 데이터에서는 드리프트가 측정되지 않음 | 16 |
| 4-2 | 모델은 무엇을 보고 판단하나? (SHAP) | 전체 1위 V4, 사기에서는 V14가 압도. V17은 전체 28위지만 사기에서 6위. Time은 7위지만 방향성이 없어 빼도 성능 같음 | 17, 18 |
| 5-1 | 탐지 결과를 DB에 어떻게 담나? | PostgreSQL `fds` 스키마(거래·라벨·실행 기록·점수). 라벨은 분리하고 채점 계정은 접근 불가 → 라벨 누수를 권한으로 차단. COPY 적재 + DB 쪽 검산 | – |
| 5-2 | 과거로 학습한 모델이 새 거래를 채점하면? | 구간 0~3 학습, 구간 4를 시간순 5,000건씩 채점·저장(점수·경보·SHAP 상위 3개 근거). PR-AUC 0.8065로 4-1과 동일. 과거에서 정한 기준점이 새 구간에서는 너무 엄격(재현율 71.6% < 목표 80%) | – |
| 5-2b | 운영 중 기준점을 어떻게 감시하나? | SQL 뷰·함수로 경보 대기열, 시간대별 성능, '목표 재현율에 필요한 기준점'을 바로 확인. 80% → 오경보 29건, 90% → 1,220건(절벽) | – |

## 문제 해결 기록

- **데이터 누수**: 중복 거래가 학습·테스트에 나뉘어 들어가 점수가 부풀려지는 것을 찾아 제거 (2-1)
- **LightGBM 점수 폭주**: 결과가 조각마다 널뛰고 컴퓨터마다 달라지는 것을 단서로, 원시 점수(log-odds)가 약 25,000까지 폭주해 확률이 0/1로 뭉개지는 것을 진단. 잎 값 = −(기울기 합)/(헤시안 합)에서 헤시안 ≈ 0인 잎이 원인 → `min_child_weight=1`로 해결 (2-2)
- **클래스 가중치의 같은 문제**: 가중치 600배가 위의 보호 장치를 무력화하는 것을 확인. 가중치를 √600≈24로 낮추면 안정되지만 처리 없음보다 낮음 (2-3)
- **기준점 고르기의 누수 방지**: 테스트 데이터를 보지 않도록 학습 조각 안에서 한 번 더 교차검증한 점수로 기준점을 고름 (2-4)
- **라이브러리 충돌로 인한 강제 종료**: LightGBM과 PyTorch를 한 프로그램에서 불러오면 서로 다른 OpenMP(libomp) 3개가 함께 올라가 Segmentation fault 발생. 오류 보고서로 원인을 확인하고, 오토인코더 스크립트는 PyTorch만 쓰고 LightGBM·IF 점수는 저장된 파일에서 읽도록 분리 (3-2)
- **잘못된 비교 바로잡기**: 시간순 분할 점수가 무작위보다 낮게 나왔지만, 대조군의 테스트 데이터가 달라 원인을 단정하지 않고 '같은 시험지' 대조 실험을 추가해 차이가 시간 순서가 아니라 테스트 구간 때문임을 확인 (4-1b)
- **라벨 누수를 DB 권한으로 차단**: 채점 계정은 라벨 테이블을 읽을 수 없고, 라벨을 쓰는 뷰·함수는 `security_invoker`로 만들어 '뷰를 통한 우회'까지 막음. 학습·채점 구간이 겹치는 실행 기록은 CHECK 제약이 거부 (5-1, 5-2b)
- **실험 환경 이전**: 인텔 호환 모드 Python 3.9 → 애플 실리콘 Python 3.12로 옮기고, 전 단계를 다시 실행해 결과가 같음을 확인 (3-1 이후)

## 실행

FDS 전용 가상환경(애플 실리콘용 Python 3.12)에서 실행합니다. 다른 파이썬 환경의 라이브러리를 바꾸지 않도록 분리했습니다.

```bash
brew install python@3.12 libomp                  # 처음 한 번만 (libomp는 LightGBM에 필요)
cd ~/fds
/opt/homebrew/bin/python3.12 -m venv .venv        # 처음 한 번만
source .venv/bin/activate                         # 터미널을 열 때마다
pip install -r requirements.txt                   # 처음 한 번만
# 데이터: Kaggle "Credit Card Fraud Detection"의 creditcard.csv를 data_raw/에 둔다

python scripts/01_eda.py              data_raw/creditcard.csv reports   # 1단계 탐색
python scripts/02_baseline.py         data_raw/creditcard.csv reports   # 1단계 기본 모델
python scripts/03_duplicates.py       data_raw/creditcard.csv reports   # 2-1 중복 점검
python scripts/03b_seed_stability.py  data_raw/creditcard.csv reports   # 2-1 시드 반복
python scripts/04_tree_model.py       data_raw/creditcard.csv reports   # 2-2 LightGBM
python scripts/05_imbalance.py        data_raw/creditcard.csv reports   # 2-3 불균형 처리
python scripts/06_threshold.py        data_raw/creditcard.csv reports   # 2-4 기준점
python scripts/06b_threshold_margin.py data_raw/creditcard.csv reports  # 2-4 보완 (약 6분)
python scripts/07_isolation_forest.py data_raw/creditcard.csv reports   # 3-1 Isolation Forest (08이 쓰는 점수 파일도 저장)
python scripts/08_autoencoder.py      data_raw/creditcard.csv reports   # 3-2 오토인코더 (07 먼저 실행)
python scripts/09_ae_epochs.py        data_raw/creditcard.csv reports   # 3-3b 학습 길이 20/50/100에폭
python scripts/10a_labels_lightgbm.py data_raw/creditcard.csv reports   # 3-3a 라벨 부족 (LightGBM)
python scripts/10b_labels_autoencoder.py data_raw/creditcard.csv reports # 3-3a 라벨 부족 (오토인코더)
python scripts/11_time_split.py       data_raw/creditcard.csv reports   # 4-1 시간순 전진 검증 + 4-1b (약 2분)
python scripts/12_shap.py             data_raw/creditcard.csv reports   # 4-2 SHAP (그래프용 shap_oof.npz도 저장)

# 5단계: PostgreSQL 연동 (보안 DB 과제를 복제한 DB security_db_fds 안의 별도 스키마 fds)
createdb security_db_fds && pg_dump -Fc -d security_db | pg_restore -d security_db_fds   # 과제 DB 복제 (처음 한 번)
psql -d security_db_fds -v ON_ERROR_STOP=1 -f db/F001_fds_schema.sql      # 스키마·채점 계정
python scripts/13_load_db.py data_raw/creditcard.csv --db security_db_fds   # 거래·라벨 적재 (COPY + 검산)
python scripts/14_score_batch.py --db security_db_fds                       # 학습(구간 0~3) → 채점(구간 4) → 평가
psql -d security_db_fds -v ON_ERROR_STOP=1 -f db/F002_fds_monitoring.sql  # 경보 대기열·성능 뷰·기준점 점검
psql -d security_db_fds -c "select * from fds.threshold_check(0.8);"
# 되돌리기: psql -d security_db_fds -f db/F000_drop_fds.sql
```

그래프는 결과 JSON으로 다시 그릴 수 있습니다: `python scripts/plot_tree_model.py reports` 등 `plot_*.py`. (`plot_isolation_forest.py`·`plot_shap.py`는 두 번째 인자로 `data_raw/creditcard.csv`)
오토인코더 스크립트(08, 09, 10b)는 마지막 인자로 `mps`(애플 GPU)를 줄 수 있지만, 기록된 결과는 기본값 `cpu`로 낸 것입니다.
LightGBM과 PyTorch는 같은 프로그램에서 함께 불러오면 강제 종료되므로, 두 라이브러리를 쓰는 스크립트를 나눠 두었습니다.
환경을 바꾼 뒤 결과가 그대로인지는 `python scripts/compare_results.py reports <새 결과 폴더>`로 비교합니다.

## 구조

```
scripts/
  fds_common.py        데이터 로드(중복 제거), 5겹 분할, 지표, LightGBM 설정, 라벨 숨기기 — 2-2부터 공통 사용
  01_ ~ 14_*.py        단계별 실험 (13·14는 PostgreSQL 연동) (08·09·10b는 PyTorch만, 나머지는 PyTorch 없이)
  compare_results.py   두 결과 폴더 비교 (환경 변경 후 재현 확인)
  plot_*.py            결과 그래프
db/
  F001_fds_schema.sql      fds 스키마: transactions, fraud_labels, model_runs, scores + 최소권한 채점 계정
  F002_fds_monitoring.sql  경보 대기열·실행별 성능 뷰, threshold_check()
  F000_drop_fds.sql        fds 스키마 완전 제거 (되돌리기)
reports/
  PROJECT_LOG.md       진행 기록 (현재 상태 + 날짜별 히스토리)
  *_results.json       단계별 결과
  figures/             그래프 01~18
data_raw/              원본 데이터 (저장소에 포함하지 않음)
```

## 3단계 결론: 라벨 없이 찾을 수 있나?

![라벨이 적을 때](reports/figures/15_label_scarcity.png)

- 이 데이터에서는 **라벨이 몇 건이라도 있으면 지도학습이 낫다.** 확인된 사기 8건으로 학습한 LightGBM(0.514)이 비지도 방법 중 가장 좋은 결과(오토인코더 0.474)보다 높음
- 오토인코더는 '정상만 배운다'는 전제에 크게 의존한다. 현실에서는 확인 안 된 사기가 정상으로 섞여 있으므로, 이 전제는 쉽게 깨짐
- 비지도에서 '특이함'은 '사기'와 다르다(IF가 고액 정상 거래를 잡음). 또 ROC-AUC로 보면 순위가 뒤집혀서 지표 선택이 결론을 바꿈
- **한계**: 이 데이터의 사기는 서로 비슷한 유형이라 몇 건만 봐도 일반화가 잘 됨. 라벨에 없는 **새 수법**이 나타나는 상황은 확인하지 못했고, 비지도 방법이 필요한 이유는 바로 그런 상황에 있음

## 4단계 결론: 미래에도 통하나, 무엇을 보고 판단하나?

![SHAP 사례](reports/figures/18_shap_cases.png)

- **시간순 검증**: 과거로만 학습해도, 같은 시험지 기준으로는 성능 손해가 측정되지 않음. 처음 보인 차이(−0.05)는 구간마다 사기 비율(0.100~0.187%)과 난이도가 달라서 생긴 것 → '비교는 같은 시험지로'(2-1)를 시간 축에서 재확인
- **한계**: 데이터가 이틀치라 수 주~수 개월에 걸친 수법 변화(드리프트)는 검증할 수 없음
- **판단 근거**: 사기 판정은 V14·V4·V12·V10·V17이 값이 극단적일 때 크게 올림(V14 하나로 +8.5까지). 전체 평균 중요도와 사기에서의 중요도는 다름(V17: 전체 28위, 사기 6위)
- **SHAP 중요도 ≠ 필요성**: Time은 7위지만 사기 쪽으로 일관된 방향이 없어(+0.03), 빼도 성능이 같음
- **오류의 성격**: 기준점 바로 아래로 놓친 사기는 '조금 모자란' 경우(기준점 문제), 가장 크게 놓친 사기는 모든 특성이 정상 범위(정보 부족), 가장 강한 오경보는 잡은 사기와 근거가 같음(구분 불가 또는 라벨 오류 가능성)

## 앞으로

3. ~~비지도 이상탐지~~ — 완료 (아래 '3단계 결론')
4. ~~시간 순서 분할 + 설명가능성(SHAP)~~ — 완료 (아래 '4단계 결론')
5. 보안 DB 프로젝트 연동 — 5-1~5-2b 완료(스키마·적재·배치 채점·운영 뷰). 남은 것: 경보를 보안 이벤트·알림 파이프라인과 SOC 대시보드에 연결
6. 적대적 공격(탐지 회피) 실험

## 환경

- macOS, 애플 실리콘(arm64) **Python 3.12** (Homebrew). `requirements.txt`의 고정 버전: numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, matplotlib 3.11.2, lightgbm 4.7.0, imbalanced-learn 0.14.2, torch 2.15.0, psycopg 3.3.6
- PostgreSQL 15 (Homebrew). 5단계는 보안 DB 과제 DB를 복제한 `security_db_fds`에 별도 스키마 `fds`로 둠. 과제 원본(`security_db`)과 과제 저장소 main 브랜치는 건드리지 않고, 연동 코드는 과제 저장소의 `fds-integration` 브랜치(`~/security_db_fds` worktree)에서 작업
- 2026-11-05에 인텔 호환 모드 Python 3.9 환경에서 옮겨 왔고, 그때까지의 1~3-1단계 스크립트를 다시 실행해 **기존 결과와 같음**을 확인했습니다(점수 폭주로 무효 처리한 2-3 클래스 가중치만 0.0003 차이)
