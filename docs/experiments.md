# 실험 기록 - Playground Series S6E4

**대회**: Predicting Irrigation Need  
**평가지표**: Balanced Accuracy  
**기간**: 2026-04-01 ~ 2026-04-30

---

## 스코어 추이

| 실험 | OOF | LB | CV-LB 격차 | 비고 |
|------|-----|-----|-----------|------|
| EXP-001 XGB Baseline | 0.96599 | 0.96414 | 0.00185 | |
| EXP-003C XGB 튜닝 | 0.96812 | **0.96597** | 0.00215 | **현재 LB 최고** |
| EXP-004 Blend (가중치) | 0.96893 | 0.96588 | 0.00305 | OOF 과적합 |
| EXP-004 CatBoost 단독 | 0.96820 | 0.96516 | 0.00304 | 과적합 |
| EXP-005 XGB+원본 | 0.96824 | 0.96571 | 0.00253 | 원본 효과 없음 |
| EXP-005 균등블렌딩 | 0.96790 | 0.96483 | 0.00307 | LGB/CAT 약화 |

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

---

## 학습된 교훈
- FE(페어와이즈 조합, TargetEncoder)는 효과 없음
- 학습 설정 개선(10-Fold, 낮은 lr)이 가장 확실한 개선
- OOF 기반 가중치 최적화는 과적합 → LB에서 하락
- LightGBM은 이 데이터에서 XGB/CAT 대비 부진
- 원본 데이터 결합은 효과 없음
- CatBoost는 auto_class_weights='Balanced'가 sample_weight보다 나음
- **현재 XGB 단독이 가장 안정적** (LB 0.96597)

---

## 상위 노트북과의 격차 분석

현재 LB 0.96597 vs 상위 0.979 → **격차 약 0.013**

상위 노트북이 쓰지만 우리가 효과를 못 본 기법:
- ~~페어와이즈 조합~~ (EXP-002에서 실패)
- ~~원본 데이터~~ (EXP-005에서 실패)
- ~~단순 블렌딩~~ (EXP-004/005에서 과적합)

아직 시도 안 한 기법:
- 바이어스 튜닝 (로그공간 / Nelder-Mead)
- 상위 노트북의 전체 파이프라인 (171개 조합 + TargetEncoder + 도메인 피처를 한꺼번에)
- 메타스태킹 (Ridge / LGB 메타러너)

---

## TODO
- [ ] 다음 전략 수립
