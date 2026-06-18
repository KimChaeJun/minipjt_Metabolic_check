# 대사증후군 위험군 예측 프로젝트

국민건강영양조사(KNHANES) 2023~2024 데이터를 활용하여 대사증후군(Metabolic Syndrome) 위험군을 예측하고, 주요 위험 요인을 분석한 머신러닝 프로젝트입니다.

## 프로젝트 개요

대사증후군은 복부비만, 고혈압, 고혈당, 이상지질혈증 등이 복합적으로 나타나는 질환군으로 심혈관 질환과 당뇨병의 주요 위험 요인으로 알려져 있습니다.

본 프로젝트에서는 국민건강영양조사(KNHANES) 데이터를 활용하여 다음 두 가지 관점에서 대사증후군을 예측하였습니다.

* 생활습관 및 기본 신체정보 기반 조기 위험 예측
* 건강검진 수치 기반 고정밀 위험 예측

---

## 데이터셋

### 국민건강영양조사 (KNHANES)

* 2023년 국민건강영양조사
* 2024년 국민건강영양조사
* 분석 대상: 10,162명

### 사용 변수

#### 생활습관 기반 모델

| 구분      | 변수                         |
| ------- | -------------------------- |
| 인구학적 정보 | 성별(sex), 나이(age)           |
| 신체정보    | BMI                        |
| 생활습관    | 흡연 여부, 음주 여부, 유산소 운동 실천 여부 |

#### 건강검진 기반 모델

| 구분      | 변수              |
| ------- | --------------- |
| 인구학적 정보 | 성별, 나이          |
| 신체계측    | BMI, 허리둘레       |
| 혈압      | 수축기 혈압, 이완기 혈압  |
| 혈액검사    | 공복혈당, 중성지방, HDL |

---

## 대사증후군 정의

다음 5가지 기준 중 3개 이상 충족 시 대사증후군으로 정의하였습니다.

### 복부비만

* 남성 허리둘레 ≥ 90cm
* 여성 허리둘레 ≥ 85cm

### 고혈압

* 수축기 혈압 ≥ 130 mmHg
* 또는 이완기 혈압 ≥ 85 mmHg

### 공복혈당 상승

* 공복혈당 ≥ 100 mg/dL

### 중성지방 상승

* 중성지방 ≥ 150 mg/dL

### HDL 감소

* 남성 < 40 mg/dL
* 여성 < 50 mg/dL

---

## 데이터 분포

| 구분     |     인원 |
| ------ | -----: |
| 정상군    |  7,875 |
| 대사증후군군 |  2,287 |
| 총 인원   | 10,162 |

대사증후군 유병률은 약 22.5%로 나타났습니다.

---

## 통계 분석

### 카이제곱 검정 결과

| 변수        | 결과      |
| --------- | ------- |
| 성별        | 유의함     |
| 흡연 여부     | 유의함     |
| 유산소 운동 여부 | 유의함     |
| 음주 여부     | 유의하지 않음 |

### 주요 결과

* 흡연자의 대사증후군 유병률이 더 높게 나타남
* 유산소 운동 실천자의 대사증후군 유병률이 더 낮게 나타남
* 남성과 여성 간 유의한 차이가 존재함
* 단순 월간 음주 여부는 유의한 차이를 보이지 않음

---

## 머신러닝 모델

### 사용 모델

* Logistic Regression
* Random Forest

### 생활습관 기반 모델 성능

| Metric    | Logistic Regression | Random Forest |
| --------- | ------------------: | ------------: |
| Accuracy  |              0.7339 |        0.7211 |
| Precision |              0.4490 |        0.4386 |
| Recall    |              0.7969 |        0.8493 |
| F1 Score  |              0.5744 |        0.5784 |
| ROC-AUC   |              0.8298 |        0.8303 |

### 주요 영향 변수

#### Logistic Regression Odds Ratio

| 변수      | Odds Ratio |
| ------- | ---------: |
| BMI     |       4.05 |
| Age     |       2.16 |
| Smoking |       1.21 |

#### Random Forest Importance

1. BMI
2. Age
3. Sex
4. Aerobic Activity
5. Smoking
6. Drinking

---

## 프로젝트 구조

```text
minipjt_Metabolic_check
│
├─ data
│   ├─ HN23_ALL
│   ├─ HN24_ALL
│   ├─ metabolic_model.py
│   └─ metabolic_model_with_HE.py
│
├─ results
│   ├─ lifestyle
│   │   ├─ lifestyle_model_report.md
│   │   ├─ assets
│   │   └─ data_csv
│   │
│   └─ with_HE
│       ├─ with_HE_model_report.md
│       ├─ assets
│       └─ data_csv
│
├─ requirements.txt
└─ README.md
```

---

## 실행 방법

### 1. 가상환경 생성

```bash
python -m venv .venv
```

### 2. 가상환경 활성화

#### Windows

```bash
.venv\Scripts\activate
```

#### Git Bash

```bash
source .venv/Scripts/activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 생활습관 기반 모델 실행

```bash
python metabolic_model.py
```

### 5. 건강검진 기반 모델 실행

```bash
python metabolic_model_with_HE.py
```

---

## 기술 스택

### Data Analysis

* Pandas
* NumPy
* SciPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-Learn
* Random Forest
* Logistic Regression

### Environment

* Python 3.11
* VSCode
* Git
* GitHub

---

## 한계점

* 대사증후군 라벨을 동일 데이터의 검진 수치로 생성함
* 건강검진 기반 모델은 일부 데이터 누수(Data Leakage) 가능성이 존재함
* 국민건강영양조사 단면 조사 데이터만 활용함
* 향후 XGBoost, SHAP 기반 설명 가능한 AI(XAI) 분석으로 확장 가능

---

## Repository

GitHub Repository

https://github.com/KimChaeJun/minipjt_Metabolic_check
