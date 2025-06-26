import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. 데이터 로딩 ---
try:
    dataset_path = 'qled_dataset.csv'
    df = pd.read_csv(dataset_path)
    print(f"✅ 데이터셋 '{dataset_path}' 로딩 성공!")
    print(f"데이터셋 크기: {df.shape[0]}개의 샘플, {df.shape[1]}개의 변수")
except FileNotFoundError:
    print(f"❌ 오류: '{dataset_path}' 파일을 찾을 수 없습니다.")
    exit()

# --- 2. 데이터 준비 (입력과 목표 분리) ---
# 입력 변수(X): 두께 정보
feature_names = ['ZnO_thickness', 'QD_thickness', 'CBP_thickness']  # 실제 컬럼명으로 수정
X = df[feature_names]

# 목표 변수(y): EQE 정보
y = df['EQE_max']

# 데이터를 훈련용과 테스트용으로 분리 (80% 훈련, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n✅ 데이터를 훈련용과 테스트용으로 분리했습니다.")

# --- 3. XGBoost 모델 학습 ---
xgb_regressor = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    max_depth=7,
    learning_rate=0.1,
    random_state=42
)

print("--- XGBoost 모델 학습 시작 ---")
xgb_regressor.fit(X_train, y_train)
print("✅ 모델 학습 완료!")

# --- 4. 모델 성능 평가 ---
y_pred = xgb_regressor.predict(X_test)

# R-squared (결정 계수)
r2 = r2_score(y_test, y_pred)
# RMSE (평균 제곱근 오차)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # RMSE 계산

print("\n--- 모델 성능 평가 결과 ---")
print(f"R-squared (결정 계수): {r2:.4f}")
print(f"RMSE (평균 제곱근 오차): {rmse:.4f}")

# --- 5. 예측 결과 시각화 ---
plt.figure(figsize=(8, 8))
sns.regplot(x=y_test, y=y_pred, ci=None, 
            line_kws={'color': 'red', 'linestyle': '--', 'linewidth': 2},
            scatter_kws={'alpha': 0.6, 's': 50})
plt.xlabel("실제 EQE 값 (True Values)")
plt.ylabel("모델이 예측한 EQE 값 (Predictions)")
plt.title("XGBoost 모델 예측 성능")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='gray', linestyle='-')  # y=x 기준선
plt.grid(True)
plt.axis('equal')

# 그래프 저장
plt.savefig('model_performance.png')
print("\n✅ 성능 평가 그래프가 'model_performance.png'로 저장되었습니다.")

# --- 6. 특성 중요도 시각화 ---
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_regressor.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('특성 중요도 (Feature Importance)')
plt.xlabel('중요도')
plt.ylabel('특성')

# 그래프 저장
plt.savefig('feature_importance.png')
print("✅ 특성 중요도 그래프가 'feature_importance.png'로 저장되었습니다.")

# --- 7. 모델 저장 ---
import joblib
model_path = 'xgb_model.joblib'
joblib.dump(xgb_regressor, model_path)
print(f"✅ 학습된 모델이 '{model_path}'로 저장되었습니다.") 