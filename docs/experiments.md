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
- **CV 점수**: -
- **LB 점수**: 0.96414
- **비고**: 첫 베이스라인 제출

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

### Phase 1: 피처 엔지니어링
- [ ] EXP-002: 도메인 피처 추가 (ET_proxy, water_deficit, heat_stress, drying_force, soil_quality 등)
- [ ] EXP-003: 페어와이즈 범주형 조합 피처 생성
- [ ] EXP-004: TargetEncoder 적용 (CV 기반 누수 방지)

### Phase 2: 멀티 모델
- [ ] EXP-005: LightGBM 단일 모델
- [ ] EXP-006: CatBoost 단일 모델
- [ ] EXP-007: 3모델 가중 블렌딩 앙상블

### Phase 3: 고급 최적화
- [ ] EXP-008: 원본 데이터 활용 (가중치 결합)
- [ ] EXP-009: 메타스태킹 (Ridge / LGB 메타러너)
- [ ] EXP-010: 바이어스 튜닝 (로그공간 / Nelder-Mead)
