# 실험 기록 - Playground Series S6E4

**대회**: Predicting Irrigation Need  
**평가지표**: Balanced Accuracy  
**기간**: 2026-04-01 ~ 2026-04-30

---

## 실험 로그

### EXP-001: XGBoost Baseline
- **날짜**: 2026-04-03
- **노트북**: `xgb_baseline.ipynb`
- **모델**: XGBoost (5-Fold StratifiedKFold)
- **피처**: 원본 19개 (LabelEncoding)
- **불균형 처리**: sample_weight (balanced)
- **주요 파라미터**: max_depth=8, lr=0.05, n_estimators=1000, subsample=0.8
- **OOF 점수**: 0.96599
- **LB 점수**: 0.96414
- **비고**: 첫 베이스라인 제출

### EXP-002: Feature Engineering (페어와이즈 조합)
- **날짜**: 2026-04-06
- **노트북**: `exp002_fe.ipynb`
- **모델**: XGBoost (5-Fold, 기존 파라미터)
- **시도**:
  - Step 1: Tier 1 조합 4개 (Growth_Stage×Mulching 등) → OOF 0.96591 (-0.00008)
  - Step 2: +Tier 2 조합 6개 → OOF 0.96569 (-0.00030)
  - Step 3: +TargetEncoder → OOF 0.96525 (-0.00074)
- **결론**: 모든 Step에서 베이스라인 대비 악화. LabelEncoding된 조합 피처는 XGBoost에 노이즈로 작용

### EXP-003: 학습 설정 개선
- **날짜**: 2026-04-06
- **노트북**: `exp003_training_tuning.ipynb`
- **피처**: 원본 19개 유지
- **시도**:
  - A: 10-Fold (기존 파라미터) → OOF 0.96660 (+0.00061)
  - B: 5-Fold + ES(200) + lr=0.02 + max_depth=6 + 5000est → OOF 0.96770 (+0.00171)
  - C: 10-Fold + B 파라미터 → **OOF 0.96812 (+0.00213)**
- **LB 점수**: 0.96597 (CV-LB 격차: 0.00215)
- **비고**: Early stopping 거의 미작동 (best_iter ~4900-5000) → estimators 더 늘릴 여지 있음

---

## 상위 노트북 분석 요약

| 노트북 | 모델 | 핵심 기법 | OOF/LB |
|--------|------|----------|--------|
| lgbm-xgb-cat | XGB+CB+LGB 10-Fold | 도메인 피처, 범주형 쌍 조합(171개), TargetEncoder, 바이어스 튜닝 | 0.97890 |
| ensemble-of-solutions | 상위 솔루션 투표 | 조건부 투표 메타-앙상블 | 0.97771 |
| pairwise-combos | XGB+CB+LGB 5-Fold | 페어와이즈 조합(135개), 도메인 피처(13개), 메타러너 혼합, 바이어스 튜닝 | 0.97906 |

### 공통 패턴
- 3모델 앙상블 (XGBoost + CatBoost + LightGBM)
- 페어와이즈 범주형 조합 피처
- 물리 기반 도메인 피처 (ET_proxy, water_deficit, heat_stress 등)
- TargetEncoder (CV 기반 누수 방지)
- 바이어스 튜닝 (로그공간 / Nelder-Mead)
- 원본 데이터 활용 (가중치 결합)

---

## TODO

### 즉시
- [ ] EXP-003C 제출 → LB 확인 (CV-LB 격차 모니터링)
- [ ] n_estimators 10000으로 확장 (아직 수렴 안 됨)

### 멀티 모델
- [ ] LightGBM 단일 모델 (EXP-003C 설정 기반)
- [ ] CatBoost 단일 모델
- [ ] 3모델 가중 블렌딩 앙상블

### 고급 최적화
- [ ] 원본 데이터 활용 (가중치 결합)
- [ ] 바이어스 튜닝 (로그공간 / Nelder-Mead)
