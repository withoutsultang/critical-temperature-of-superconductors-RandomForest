# **데이터 사이언스 기말고사 과제 - 초전도체 임계온도 예측 모델**

이 프로젝트는 초전도체의 임계온도(critical temperature)를 예측하는 문제를 해결하기 위해 데이터 분석을 진행한 데이터 사이언스 과제입니다.
**랜덤 포레스트** 모델을 사용하여 데이터를 학습시키고, 최적의 모델을 도출하기 위해 하이퍼파라미터 튜닝 및 성능 분석을 진행했습니다.

## **프로젝트 개요**
초전도체는 특정 온도 이하에서 전기 저항이 사라지는 물질입니다. 본 프로젝트에서는 초전도체의 화학적 조성과 물리적 특성 데이터를 기반으로 임계온도를 예측하기 위한 모델을 개발했습니다.


## **1. 데이터셋 및 프로젝트 구조**
### **1.1 데이터셋 정보**
- **`train.csv`**: 초전도체와 관련된 물리적 속성과 임계온도 데이터  
- **`formula_train.csv`**: 화학적 조성 데이터
- **데이터셋 출처**: Critical temperature of superconductors-Kaggle](https://www.kaggle.com/competitions/critical-temperature-of-superconductors)
- 데이터는 캐글(Kaggle)의 "Critical Temperature of Superconductors" 대회에서 제공된 데이터를 활용했습니다.
### **1.2 데이터셋 설명**
- **train.csv**


| 영어 변수명                   | 한글 변수명               |
|------------------------------|---------------------------|
| number_of_elements           | 원소 개수                 |
| mean_atomic_mass             | 평균 원자 질량            |
| wtd_mean_atomic_mass         | 가중 평균 원자 질량       |
| gmean_atomic_mass            | 기하 평균 원자 질량       |
| wtd_gmean_atomic_mass        | 가중 기하 평균 원자 질량  |
| entropy_atomic_mass          | 원자 질량 엔트로피        |
| wtd_entropy_atomic_mass      | 가중 원자 질량 엔트로피   |
| range_atomic_mass            | 원자 질량 범위            |
| wtd_range_atomic_mass        | 가중 원자 질량 범위       |
| std_atomic_mass              | 원자 질량 표준편차        |
| wtd_std_atomic_mass          | 가중 원자 질량 표준편차   |
| mean_fie                     | 평균 1차 이온화 에너지    |
| wtd_mean_fie                 | 가중 평균 1차 이온화 에너지 |
| gmean_fie                    | 기하 평균 1차 이온화 에너지 |
| wtd_gmean_fie                | 가중 기하 평균 1차 이온화 에너지 |
| entropy_fie                  | 1차 이온화 에너지 엔트로피 |
| wtd_entropy_fie              | 가중 1차 이온화 에너지 엔트로피 |
| range_fie                    | 1차 이온화 에너지 범위    |
| wtd_range_fie                | 가중 1차 이온화 에너지 범위 |
| std_fie                      | 1차 이온화 에너지 표준편차 |
| wtd_std_fie                  | 가중 1차 이온화 에너지 표준편차 |
| mean_atomic_radius           | 평균 원자 반경            |
| wtd_mean_atomic_radius       | 가중 평균 원자 반경       |
| gmean_atomic_radius          | 기하 평균 원자 반경       |
| wtd_gmean_atomic_radius      | 가중 기하 평균 원자 반경  |
| entropy_atomic_radius        | 원자 반경 엔트로피        |
| wtd_entropy_atomic_radius    | 가중 원자 반경 엔트로피   |
| range_atomic_radius          | 원자 반경 범위            |
| wtd_range_atomic_radius      | 가중 원자 반경 범위       |
| std_atomic_radius            | 원자 반경 표준편차        |
| wtd_std_atomic_radius        | 가중 원자 반경 표준편차   |
| mean_Density                 | 평균 밀도                 |
| wtd_mean_Density             | 가중 평균 밀도            |
| gmean_Density                | 기하 평균 밀도            |
| wtd_gmean_Density            | 가중 기하 평균 밀도       |
| entropy_Density              | 밀도 엔트로피             |
| wtd_entropy_Density          | 가중 밀도 엔트로피        |
| range_Density                | 밀도 범위                 |
| wtd_range_Density            | 가중 밀도 범위            |
| std_Density                  | 밀도 표준편차             |
| wtd_std_Density              | 가중 밀도 표준편차        |
| mean_ElectronAffinity        | 평균 전자 친화도          |
| wtd_mean_ElectronAffinity    | 가중 평균 전자 친화도     |
| gmean_ElectronAffinity       | 기하 평균 전자 친화도     |
| wtd_gmean_ElectronAffinity   | 가중 기하 평균 전자 친화도 |
| entropy_ElectronAffinity     | 전자 친화도 엔트로피      |
| wtd_entropy_ElectronAffinity | 가중 전자 친화도 엔트로피 |
| range_ElectronAffinity       | 전자 친화도 범위          |
| wtd_range_ElectronAffinity   | 가중 전자 친화도 범위     |
| std_ElectronAffinity         | 전자 친화도 표준편차      |
| wtd_std_ElectronAffinity     | 가중 전자 친화도 표준편차 |
| mean_FusionHeat              | 평균 융해열               |
| wtd_mean_FusionHeat          | 가중 평균 융해열          |
| gmean_FusionHeat             | 기하 평균 융해열          |
| wtd_gmean_FusionHeat         | 가중 기하 평균 융해열     |
| entropy_FusionHeat           | 융해열 엔트로피           |
| wtd_entropy_FusionHeat       | 가중 융해열 엔트로피      |
| range_FusionHeat             | 융해열 범위               |
| wtd_range_FusionHeat         | 가중 융해열 범위          |
| std_FusionHeat               | 융해열 표준편차           |
| wtd_std_FusionHeat           | 가중 융해열 표준편차      |
| mean_ThermalConductivity     | 평균 열전도도             |
| wtd_mean_ThermalConductivity | 가중 평균 열전도도        |
| gmean_ThermalConductivity    | 기하 평균 열전도도        |
| wtd_gmean_ThermalConductivity| 가중 기하 평균 열전도도   |
| entropy_ThermalConductivity  | 열전도도 엔트로피         |
| wtd_entropy_ThermalConductivity | 가중 열전도도 엔트로피  |
| range_ThermalConductivity    | 열전도도 범위             |
| wtd_range_ThermalConductivity | 가중 열전도도 범위       |
| std_ThermalConductivity      | 열전도도 표준편차         |
| wtd_std_ThermalConductivity  | 가중 열전도도 표준편차    |
| mean_Valence                 | 평균 원자가               |
| wtd_mean_Valence             | 가중 평균 원자가          |
| gmean_Valence                | 기하 평균 원자가          |
| wtd_gmean_Valence            | 가중 기하 평균 원자가     |
| entropy_Valence              | 원자가 엔트로피           |
| wtd_entropy_Valence          | 가중 원자가 엔트로피      |
| range_Valence                | 원자가 범위               |
| wtd_range_Valence            | 가중 원자가 범위          |
| std_Valence                  | 원자가 표준편차           |
| wtd_std_Valence              | 가중 원자가 표준편차      |
| critical_temp                | 임계 온도                 |

- **formula_train.csv**


| **영문 변수명** | **한글 변수명** |
|-----------------|----------------|
| H               | 수소           |
| He              | 헬륨           |
| Li              | 리튬           |
| Be              | 베릴륨         |
| B               | 붕소           |
| C               | 탄소           |
| N               | 질소           |
| O               | 산소           |
| F               | 플루오린       |
| Ne              | 네온           |
| Na              | 나트륨         |
| Mg              | 마그네슘       |
| Al              | 알루미늄       |
| Si              | 규소           |
| P               | 인             |
| S               | 황             |
| Cl              | 염소           |
| Ar              | 아르곤         |
| K               | 칼륨           |
| Ca              | 칼슘           |
| Sc              | 스칸듐         |
| Ti              | 타이타늄       |
| V               | 바나듐         |
| Cr              | 크롬           |
| Mn              | 망가니즈       |
| Fe              | 철             |
| Co              | 코발트         |
| Ni              | 니켈           |
| Cu              | 구리           |
| Zn              | 아연           |
| Ga              | 갈륨           |
| Ge              | 저마늄         |
| As              | 비소           |
| Se              | 셀레늄         |
| Br              | 브로민         |
| Kr              | 크립톤         |
| Rb              | 루비듐         |
| Sr              | 스트론튬       |
| Y               | 이트륨         |
| Zr              | 지르코늄       |
| Nb              | 나이오븀       |
| Mo              | 몰리브덴       |
| Tc              | 테크네튬       |
| Ru              | 루테늄         |
| Rh              | 로듐           |
| Pd              | 팔라듐         |
| Ag              | 은             |
| Cd              | 카드뮴         |
| In              | 인듐           |
| Sn              | 주석           |
| Sb              | 안티모니       |
| Te              | 텔루륨         |
| I               | 아이오딘       |
| Xe              | 제논           |
| Cs              | 세슘           |
| Ba              | 바륨           |
| La              | 란타넘         |
| Ce              | 세륨           |
| Pr              | 프라세오디뮴   |
| Nd              | 네오디뮴       |
| Pm              | 프로메튬       |
| Sm              | 사마륨         |
| Eu              | 유로퓸         |
| Gd              | 가돌리늄       |
| Tb              | 터븀           |
| Dy              | 디스프로슘     |
| Ho              | 홀뮴           |
| Er              | 어븀           |
| Tm              | 툴륨           |
| Yb              | 이터븀         |
| Lu              | 루테튬         |
| Hf              | 하프늄         |
| Ta              | 탄탈럼         |
| W               | 텅스텐         |
| Re              | 레늄           |
| Os              | 오스뮴         |
| Ir              | 이리듐         |
| Pt              | 백금           |
| Au              | 금             |
| Hg              | 수은           |
| Tl              | 탈륨           |
| Pb              | 납             |
| Bi              | 비스무트       |
| Po              | 폴로늄         |
| At              | 아스타틴       |
| Rn              | 라돈           |
| critical_temp   | 임계 온도      |
| material        | 물질           |
### **1.3 프로젝트 단계**
  - 데이터 로드 및 병합
  - 데이터 전처리
  - EDA
  - 랜덤 포레스트 모델 학습 및 하이퍼파라미터 튜닝
  - 결론
---
## **2. 주요 라이브러리**
```python
import pandas as pd
import numpy as np
import platform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')
```
- **`pandas`**: 데이터 처리 및 분석
- **`numpy`**: 수치 계산
- **`matplotlib`, `seaborn`**: 데이터 시각화
- **`sklearn`**: 모델 학습
---

## **3. 데이터 로드 및 전처리**

### **3.1 데이터 로드**
```python
# 데이터 읽기
train = pd.read_csv('/content/data/train.csv')
formula_train = pd.read_csv('/content/data/formula_train.csv')

# 데이터 확인
print(train.head())
print(formula_train.head())
```

### **3.2 데이터 병합 및 전처리**
- `train.csv`와 `formula_train.csv`를 병합해 모델 학습에 사용할 하나의 데이터로 구성
- 결측치 처리를 확인하고, 데이터의 분포를 시각화

```python
# formula_train에서 불필요한 열 제거
formula_train = formula_train.drop(['critical_temp', 'material'], axis=1)

# train 데이터와 formula_train 데이터를 병합
train_all = pd.concat([train, formula_train], axis=1)

# 결측치 확인
print(train_all.isnull().sum())

# 임계 온도 분포 시각화
plt.figure(figsize=(10, 6))
plt.hist(train_all['critical_temp'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Critical Temperature (K)')
plt.ylabel('Frequency')
plt.title('Distribution of Critical Temperature')
plt.show()
```
![image](https://github.com/user-attachments/assets/a0307e01-16de-4810-9eb3-e2fcef377179)

### **설명**
- `critical_temp`는 임계온도를 나타내며, 예측(Target) 변수로 사용됩니다.
- 데이터 병합 후 결측치가 없는 것을 확인했고, `critical_temp`의 분포는 0에 집중되었음을 시각화했습니다.

---

## **4. 탐색적 데이터 분석 (EDA)**

### **4.1 상관 행렬 분석**
- 임계온도와 가장 높은 상관관계를 가지는 변수 분
```python
# 상관 행렬 계산
correlation_matrix = train_all.corr()

# 상위 10개의 변수 시각화
top_features = correlation_matrix['critical_temp'].abs().sort_values(ascending=False).index[:11]
plt.figure(figsize=(12, 8))
sns.heatmap(train_all[top_features].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Top Features Correlation Matrix')
plt.show()
```
![image](https://github.com/user-attachments/assets/62b36614-c0fd-4293-95fa-0dc18f6386aa)


### **설명**
- 상관 행렬 분석 결과, `평균 원자 질량`, `열전도율`, `원자가 범위`와 같은 변수가 `critical_temp`와 높은 상관관계를 보였습니다.

---
## **5. 랜덤 포레스트 모델 학습**

### **5.1 데이터 분할**
```python
X = train_all.drop('critical_temp', axis=1)
y = train_all['critical_temp']

# 학습용 데이터와 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### **5.2 기본 랜덤 포레스트 모델 학습**
```python
# 랜덤 포레스트 모델 정의 및 학습
rf = RandomForestRegressor(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)

# 테스트 데이터 예측 및 성능 평가
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
```

### **설명**
- 데이터는 학습용(80%)과 테스트용(20%)으로 분할
- 초기 랜덤 포레스트 모델에서 `MSE`를 확인해 기본 성능을 평가

---

## **6. 하이퍼파라미터 튜닝**

### **6.1 RandomizedSearchCV를 통한 최적화**
```python
param_distributions = {
    'n_estimators': [10, 20, 30, 40, 50],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=10,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42
)
random_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터 확인
print("Best Parameters:", random_search.best_params_)

# 최적 모델로 테스트 데이터 예측
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Optimized Model MSE: {mse}")
```

### **설명**
- RandomizedSearchCV를 사용해 최적의 하이퍼파라미터(`n_estimators`, `max_depth` 등)를 탐색
- 최적 모델에서 성능(MSE)을 확인한 결과, 초기 모델보다 성능이 향상됨을 확인

---

## **7. 성능 분석**

### **7.1 트리 개수에 따른 MSE 시각화**
```python
n_estimators_range = [10, 20, 30, 40, 50, 60, 70]
mse_values = []

for n in n_estimators_range:
    rf = RandomForestRegressor(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mse_values.append(mean_squared_error(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, mse_values, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Effect of Tree Count on MSE')
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/61f0ba58-9942-4f00-81d4-225058a26179)

### **7.2 변수 중요도 시각화**
```python
feature_importances = best_rf.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

# 상위 10개의 중요 변수 시각화
plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.show()
```
![image](https://github.com/user-attachments/assets/4aea9041-30c2-4ad1-95eb-0144216b4f6f)

---

## **8. 결론 및 향후 개선 방향**
### **결론**
- 랜덤 포레스트 모델을 사용해 초전도체 임계온도를 예측하는 작업을 진행했습니다.
- 하이퍼파라미터 튜닝(RandomizedSearchCV)을 통해 성능을 최적화했습니다.
- 주요 변수 분석을 통해 **구리**과 **칼슘**등 화학적 조성이 중요한 변수임을 확인했습니다.

### **향후 개선 방향**
- **XGBoost**나 **LightGBM** 모델을 도입해 성능 비교
- GridSearch를 사용해 더욱 정교한 하이퍼파라미터 탐색
- 추가적인 데이터 증강 및 변수 생성으로 모델 성능 향상

---

## Kaggle 제출 결과
![image](https://github.com/user-attachments/assets/07f099f6-cc96-4583-8d1b-a359fff4701c)
