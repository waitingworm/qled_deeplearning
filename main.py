import oledpy
import numpy as np
import pandas as pd
from scipy.stats import qmc
import optuna
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os

# --- 0. 데이터 파일 경로 설정 ---
PL_DATA_PATH = '/home/minwoo/Desktop/display_lab/QLED_Deeplearning/PL_TSLee.csv'
NK_DATA_PATH = '/home/minwoo/Desktop/display_lab/QLED_Deeplearning/nkvalue_TSLee.csv'

# 데이터 로드
pl_data = pd.read_csv(PL_DATA_PATH)
nk_data = pd.read_csv(NK_DATA_PATH)

print("✅ 데이터 파일 로드 완료")
print(f"PL 데이터 shape: {pl_data.shape}")
print(f"NK 데이터 shape: {nk_data.shape}")

# --- 1. 시뮬레이션 파라미터 정의 ---
# 실제 소자 구조에 맞게 두께 범위 조정
param_space = {
    'ZnO_thickness': (20, 80),    # ETL (ZnO) 두께 범위
    'QD_thickness': (10, 40),     # EML (QD) 두께 범위
    'CBP_thickness': (20, 80)     # HTL (CBP) 두께 범위
}

# --- 2. 데이터셋 생성 ---
def generate_dataset(n_samples=300):
    # 라틴 하이퍼큐브 샘플링
    sampler = qmc.LatinHypercube(d=len(param_space))
    sample_points = sampler.random(n=n_samples)
    
    # 샘플링된 값들을 실제 파라미터 범위에 맞게 스케일링
    l_bounds = [v[0] for v in param_space.values()]
    u_bounds = [v[1] for v in param_space.values()]
    scaled_samples = qmc.scale(sample_points, l_bounds, u_bounds)
    
    # DataFrame으로 변환
    input_df = pd.DataFrame(scaled_samples, columns=param_space.keys())
    
    # 시뮬레이션 결과 저장
    results = []
    
    for index, row in input_df.iterrows():
        if (index + 1) % 20 == 0:
            print(f"시뮬레이션 진행 중... ({index + 1}/{n_samples})")
        
        try:
            # 실제 소자 구조 정의
            device_structure = [
                ['substrate', 1, 'substrate'],
                ['Ag', 100, 'anode'],           # Ag 전극
                ['CBP', row['CBP_thickness'], 'HTL'],  # HTL
                ['QD', row['QD_thickness'], 'EML'],    # EML
                ['ZnO', row['ZnO_thickness'], 'ETL'],  # ETL
                ['Ag', 100, 'cathode']          # Ag 전극
            ]
            
            # oledpy 시뮬레이션 실행
            result = oledpy.simulate_device(
                structure=device_structure,
                mode='TE',
                wavelength_range=(400, 750),
                pl_data=pl_data,
                nk_data=nk_data
            )
            
            results.append({
                'EQE_max': result['eqe_max'],
                'FWHM': result['fwhm'],
                'peak_wavelength': result['peak_wavelength']
            })
            
        except Exception as e:
            print(f"시뮬레이션 오류 (index {index}): {e}")
            results.append({
                'EQE_max': np.nan,
                'FWHM': np.nan,
                'peak_wavelength': np.nan
            })
    
    # 결과를 DataFrame으로 변환
    output_df = pd.DataFrame(results)
    
    # 입력과 출력을 결합
    full_dataset = pd.concat([input_df, output_df], axis=1)
    full_dataset.dropna(inplace=True)
    
    # CSV 파일로 저장
    full_dataset.to_csv('qled_dataset.csv', index=False)
    print(f"\n✅ 데이터셋 생성 완료! 총 {len(full_dataset)}개의 데이터를 저장했습니다.")
    
    return full_dataset

# --- 3. XGBoost 모델 학습 ---
def train_xgboost_model(dataset):
    X = dataset[param_space.keys()]
    y = dataset['EQE_max']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"\n✅ XGBoost 모델 학습 완료! R-squared 점수: {score:.4f}")
    
    return model

# --- 4. 베이즈 최적화 ---
def optimize_thickness(model):
    def objective(trial):
        # Optuna가 탐색할 파라미터 범위 정의
        ZnO = trial.suggest_float('ZnO_thickness', *param_space['ZnO_thickness'])
        QD = trial.suggest_float('QD_thickness', *param_space['QD_thickness'])
        CBP = trial.suggest_float('CBP_thickness', *param_space['CBP_thickness'])
        
        # 모델로 예측
        X = pd.DataFrame([[ZnO, QD, CBP]], columns=param_space.keys())
        predicted_eqe = model.predict(X)[0]
        
        return predicted_eqe
    
    # 최적화 실행
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    print("\n✅ 베이즈 최적화 완료!")
    print("최적의 두께 조합:")
    for param, value in study.best_params.items():
        print(f"{param}: {value:.2f} nm")
    print(f"예상 최대 EQE: {study.best_value:.2f}%")
    
    return study.best_params

# --- 메인 실행 ---
if __name__ == "__main__":
    # 1. 데이터셋 생성
    dataset = generate_dataset(n_samples=300)
    
    # 2. XGBoost 모델 학습
    model = train_xgboost_model(dataset)
    
    # 3. 베이즈 최적화로 최적 두께 찾기
    best_params = optimize_thickness(model)

