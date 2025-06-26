import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import optuna
import plotly
import joblib

# --- 1. 데이터 로딩 및 모델 재학습 ---
try:
    dataset_path = 'qled_dataset.csv'
    df = pd.read_csv(dataset_path)
    print(f"✅ 데이터셋 '{dataset_path}' 로딩 성공!")
except FileNotFoundError:
    print(f"❌ 오류: '{dataset_path}' 파일을 찾을 수 없습니다.")
    exit()

# 입력 변수와 목표 변수 준비
feature_names = ['ZnO_thickness', 'QD_thickness', 'CBP_thickness']  # 실제 컬럼명으로 수정
X = df[feature_names]
y = df['EQE_max']

# 전체 데이터를 사용하여 모델을 학습
xgb_regressor = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    max_depth=7,
    learning_rate=0.1,
    random_state=42
)
xgb_regressor.fit(X, y)
print("✅ 최적화에 사용할 대리 모델(XGBoost) 준비 완료!")

# --- 2. 베이즈 최적화를 위한 목적 함수 정의 ---
def objective(trial):
    """
    Optuna가 최적화할 목적 함수.
    trial 객체는 각 탐색 시도에 해당하며, 파라미터 값의 범위를 제안합니다.
    """
    # 탐색할 각 파라미터(두께)의 이름과 범위를 지정
    zno_thick = trial.suggest_float('ZnO_thickness', 20, 80)
    qd_thick = trial.suggest_float('QD_thickness', 10, 40)
    cbp_thick = trial.suggest_float('CBP_thickness', 20, 80)
    
    # 제안된 파라미터 조합을 모델이 예측할 수 있는 형태로 변환
    params_to_predict = pd.DataFrame({
        'ZnO_thickness': [zno_thick],
        'QD_thickness': [qd_thick],
        'CBP_thickness': [cbp_thick]
    })
    
    # 학습된 XGBoost 모델로 EQE 값을 예측
    predicted_eqe = xgb_regressor.predict(params_to_predict)[0]
    
    return predicted_eqe

# --- 3. 최적화 실행 ---
print("\n--- 베이즈 최적화 시작 ---")
study = optuna.create_study(direction='maximize')

# 200번의 탐색을 통해 최적의 조합을 찾음
study.optimize(objective, n_trials=200)

print("✅ 베이즈 최적화 완료!")

# --- 4. 최적화 결과 출력 ---
print("\n--- 최종 최적화 결과 ---")
print(f"총 탐색 횟수: {len(study.trials)}")

# 가장 좋았던 결과를 best_trial 변수에 저장
best_trial = study.best_trial

print("\n[찾아낸 최적의 두께 조합]")
for key, value in best_trial.params.items():
    print(f"  - {key}: {value:.2f} nm")

print(f"\n[예상되는 최대 EQE 값]")
print(f"  - {best_trial.value:.2f} %")

# --- 5. 최적화 과정 시각화 ---
print("\n--- 최적화 과정 시각화 ---")

# 1. 각 Trial 별 점수 변화 그래프
fig1 = optuna.visualization.plot_optimization_history(study)
fig1.write_html("optimization_history.html")
print("✅ 최적화 과정 그래프가 'optimization_history.html'로 저장되었습니다.")

# 2. 각 파라미터의 중요도 그래프
fig2 = optuna.visualization.plot_param_importances(study)
fig2.write_html("parameter_importance.html")
print("✅ 파라미터 중요도 그래프가 'parameter_importance.html'로 저장되었습니다.")

# --- 6. 최적화 결과 저장 ---
# 최적의 파라미터 조합을 CSV 파일로 저장
best_params_df = pd.DataFrame([best_trial.params])
best_params_df['predicted_EQE'] = best_trial.value
best_params_df.to_csv('best_parameters.csv', index=False)
print("✅ 최적의 파라미터 조합이 'best_parameters.csv'로 저장되었습니다.") 