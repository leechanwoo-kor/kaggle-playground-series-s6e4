# 실험 기록 - Playground Series S6E4

**대회**: Predicting Irrigation Need  
**평가지표**: Balanced Accuracy  
**기간**: 2026-04-01 ~ 2026-04-30

---

## 스코어 추이

| 실험 | OOF | LB | CV-LB 격차 | 비고 |
|------|-----|-----|-----------|------|
| EXP-001 XGB Baseline | 0.96599 | 0.96414 | 0.00185 | |
| EXP-003C XGB 튜닝 | 0.96812 | 0.96597 | 0.00215 | |
| EXP-004 Blend (가중치) | 0.96893 | 0.96588 | 0.00305 | OOF 과적합 |
| EXP-004 CatBoost 단독 | 0.96820 | 0.96516 | 0.00304 | 과적합 |
| EXP-005 XGB+원본 | 0.96824 | 0.96571 | 0.00253 | 원본 효과 없음 |
| EXP-005 균등블렌딩 | 0.96790 | 0.96483 | 0.00307 | LGB/CAT 약화 |
| EXP-006 XGB 단독 | 0.97542 | 0.97498 | 0.00044 | 파이프라인 개선 |
| **EXP-006 Blend+Bias** | **0.97886** | **0.97792** | **0.00094** | **현재 LB 최고** |

---

## 실험 로그

### EXP-001: XGBoost Baseline
- **노트북**: `xgb_baseline.ipynb`
- **설정**: 5-Fold, lr=0.05, max_depth=8, 1000est
- **OOF**: 0.96599 / **LB**: 0.96414

### EXP-002: Feature Engineering
- **노트북**: `exp002_fe.ipynb`
- **시도**: 페어와이즈 조합 (Tier1 4개 → Tier2 6개) + TargetEncoder
- **결론**: 모든 Step에서 악화. FE 효과 없음

### EXP-003: 학습 설정 개선
- **노트북**: `exp003_training_tuning.ipynb`
- **최적**: 10-Fold + lr=0.02 + max_depth=6 + 5000est + ES(200)
- **OOF**: 0.96812 / **LB**: 0.96597
- n_estimators=10000 → 5000에서 수렴 확인, 효과 없음

### EXP-004: 멀티 모델 앙상블
- **노트북**: `exp004_multimodel.ipynb`
- **개별 OOF**: XGB=0.96809, LGB=0.96572, CAT=0.96820
- **블렌딩 OOF**: 0.96893 / **LB**: 0.96588 (과적합)
- LGB 부진으로 블렌딩 효과 제한적

### EXP-005: 원본 데이터 활용 + 모델 튜닝
- **노트북**: `exp005_orig_data.ipynb`
- **원본 데이터** (10K행, weight=0.4) 결합 → XGB 미세 개선, LGB/CAT 효과 없음
- **LGB 튜닝** (lr 0.04→0.02, est 3K→5K) → 여전히 부진 (0.96569)
- **CAT 재조정** (sample_weight 통일) → 오히려 악화 (0.96777), auto_class_weights가 나았음
- **XGB+원본 LB**: 0.96571, **균등블렌딩 LB**: 0.96483

### EXP-006: 상위 노트북 파이프라인 재현 ⭐
- **노트북**: `exp006_full_pipeline.ipynb`
- **핵심 변경**:
  1. 페어와이즈 조합 171개 → 카디널리티 필터 → 135개 (수치형+범주형 전체)
  2. Fold 내 TargetEncoder (train_fold ∪ orig_data로 fit)
  3. 원본 데이터 10K행 sample_weight=0.35로 결합
  4. 3모델(XGB+LGB+CAT) 가중 블렌딩
  5. Logit 공간 Bias Tuning (coordinate descent)
- **개별 OOF**: XGB=0.97542, LGB=0.97514, CAT=0.97738
- **블렌딩 OOF**: 0.97727 (XGB=0.10, LGB=0.05, CAT=0.85)
- **+Bias Tuning OOF**: 0.97886 (bias=[1.76, -0.63, -1.00])
- **LB**: 0.97792 (XGB 단독도 0.97498로 이전 최고 크게 추월)
- **교훈**:
  - 상위 노트북 기법들은 **한꺼번에 적용해야** 효과 발휘 (이전 개별 시도 실패 원인)
  - Fold 내 TargetEncoder가 진정한 누수 방지 + 원본 데이터 활용 핵심
  - CV-LB 격차가 오히려 줄어듦 → 파이프라인이 과적합이 아닌 일반화에 기여

---

## 학습된 교훈

- FE(페어와이즈 조합, TargetEncoder)는 **단독으로는** 효과 없음
- 학습 설정 개선(10-Fold, 낮은 lr)이 가장 확실한 개선 중 하나
- OOF 기반 가중치 최적화는 과적합 위험 → CV-LB 격차 확인 필수
- 원본 데이터는 **단순 concat으로는** 효과 없음 — fold 내 TargetEncoder와 함께 써야 작동
- CatBoost는 auto_class_weights='Balanced'가 sample_weight보다 나음
- **상위 노트북의 기법은 통합 파이프라인으로 동작** — 개별 기법 단독 시도는 시너지 효과 없음
- **Fold 내 TargetEncoder + 원본 데이터 결합**이 EXP-002/005 실패의 근본 원인 해결
- **Bias Tuning은 +0.00160 OOF 개선** — High 클래스에 강한 양의 편향(+1.76) 필요

---

## 상위 노트북과의 격차 분석

**현재 LB 0.97792 vs 상위 0.97890~0.97906 → 격차 ~0.001**

아직 시도 안 한 기법:
- CatBoost iterations 확대 (EXP-006에서 best_iter=1999로 미수렴)
- Multi-seed XGBoost (저비용 다양성)
- 메타스태킹 (Ridge / LGB 메타러너)
- SymmetricTree CatBoost

---

## TODO
- [ ] EXP-007: CatBoost iterations 확대 + Multi-seed XGB
