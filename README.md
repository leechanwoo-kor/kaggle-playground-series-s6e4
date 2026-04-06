# Playground Series S6E4 - Predicting Irrigation Need

3클래스 분류 (Low / Medium / High) | Balanced Accuracy | ~2026-04-30

```
data/               대회 데이터
data/notebooks/     참고용 상위 노트북
docs/               대회 개요, 데이터 설명, 실험 기록
notebooks/          실험 노트북
submissions/        제출 파일
```

## Setup

```bash
conda activate s6e4
pip install -r requirements.txt
```

## Experiments

| # | 설명 | OOF | LB |
|---|------|-----|-----|
| 001 | XGBoost Baseline (5F, lr=0.05, 1000est) | 0.96599 | 0.96414 |
| 002 | +페어와이즈 조합 + TargetEncoder | 0.96525 | — (악화) |
| 003 | 10-Fold + ES + lr=0.02 + 5000est | **0.96812** | **0.96597** |
