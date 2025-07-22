import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from oledpy.dipole_emission import ThinFilmArchitecture
import warnings
warnings.filterwarnings('ignore')
import contextlib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# --- 모델 변경: 더 빠르고 효율적인 LightGBM을 사용합니다 ---
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from joblib import Parallel, delayed


# Context manager: stdout 억제
@contextlib.contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

# Load optical constants and PL spectrum
# Make sure 'Reference Data' directory exists in the same path as your script
# and contains 'QD_nk_simul.csv' and 'QD_PL_simul.csv'
try:
    # Assuming 'Reference Data' is in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    nk_data = pd.read_csv(os.path.join(script_dir, 'Reference Data', 'QD_nk_simul.csv'))
    pl_data = pd.read_csv(os.path.join(script_dir, 'Reference Data', 'QD_PL_simul.csv'))
except FileNotFoundError:
    print("Error: 'Reference Data' directory or required CSV files not found.")
    print("Please create a folder named 'Reference Data' in the same directory as your script,")
    print("and place 'QD_nk_simul.csv' and 'QD_PL_simul.csv' inside it.")
    sys.exit(1) # Exit if files are not found

# Use subset of PL spectrum for faster calculation (optional)
w_min, w_max = 480, 720
pl_wavelengths = pl_data.iloc[:, 0].to_numpy()
pl_intensities = pl_data.iloc[:, 1].to_numpy()
mask = (pl_wavelengths >= w_min) & (pl_wavelengths <= w_max)
pl_wavelengths = pl_wavelengths[mask]
pl_intensities = pl_intensities[mask]

# Plot PL spectrum
plt.figure(figsize=(6, 4))
plt.plot(pl_wavelengths, pl_intensities, label="PL Spectrum")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity (a.u.)")
plt.title("Photoluminescence (PL) Spectrum")
plt.grid(True)
plt.legend()
plt.show(block=False)

# Function to define the layer structure
def define_layers(ETL_d, HTL_d):
    return [
        {'name': 'Air',    'd': 0,       'doping': 1, 'coherent': 0},
        {'name': 'Ag_T',   'd': 23,  'doping': 1, 'coherent': 1},
        {'name': 'HAT-CN', 'd': 5,  'doping': 1, 'coherent': 1},
        {'name': 'MoO3',   'd': 7,  'doping': 1, 'coherent': 1},
        {'name': 'CBP',    'd': HTL_d,   'doping': 1, 'coherent': 1},
        {'name': 'QD',     'd': 35,  'doping': 1, 'coherent': 1, 'active': 1},
        {'name': 'ZnO',    'd': ETL_d,   'doping': 1, 'coherent': 1},
        {'name': 'Ag_B',   'd': 100,     'doping': 1, 'coherent': 1},
        {'name': 'SiO2',   'd': 100,     'doping': 1, 'coherent': 0},
    ]

# Thickness parameter range
HTL_values = np.arange(10, 101, 2)
ETL_values = np.arange(10, 101, 2)

print(f"시뮬레이션 시작: HTL_values ({len(HTL_values)}개), ETL_values ({len(ETL_values)}개)")
print(f"총 {len(HTL_values) * len(ETL_values)}개의 조합을 계산합니다.")

def run_simulation(params):
    HTL_d, ETL_d = params
    layers = define_layers(ETL_d, HTL_d)
    with suppress_stdout():
        arch = ThinFilmArchitecture(
            layer_dict_list=layers, vac_wavelengths=pl_wavelengths,
            dipole_positions=np.linspace(0, 1, 30), tau=13.8e-9, PLQY=0.97,
            show_progress_bar=False, show_wavelength_progress_bar=False
        )
        arch.load_nk(nk_data)
        arch.set_pl_spectrum(pl_intensities)
        arch.RZ_profile = None
        qd_index = arch.layer_names.index("QD")
        arch.set_active_layer(qd_index)
        arch.thetas = np.array([0])
        custom_u = np.linspace(0, 4, 400)
        arch.init_pds_variables(custom_u=custom_u)
        arch.summarize_device()
        arch.angular_el_spectra()
        emission_intensity = np.trapz(arch.I_EL[:, 0], arch.vac_wavelengths)
    
    return np.abs(emission_intensity)

params_list = [(htl, etl) for htl in HTL_values for etl in ETL_values]

print(f"병렬 시뮬레이션 시작: 총 {len(params_list)}개의 조합을 계산합니다.")
results = Parallel(n_jobs=-1, backend='loky')(
    delayed(run_simulation)(p) for p in tqdm(params_list, desc="전체 시뮬레이션 진행률")
)
EL_map = np.array(results).reshape(len(HTL_values), len(ETL_values))

# Normalize EL map
nor_EL_intensity = EL_map / np.max(EL_map)

# Find max point from simulation
max_idx_sim = np.unravel_index(np.argmax(nor_EL_intensity), nor_EL_intensity.shape)
max_HTL_sim = HTL_values[max_idx_sim[0]]
max_ETL_sim = ETL_values[max_idx_sim[1]]
print(f"\n[INFO] 시뮬레이션 결과 Max EL Intensity: {nor_EL_intensity[max_idx_sim]:.4f}")
print(f"[INFO] 시뮬레이션 결과 Optimal HTL: {max_HTL_sim} nm")
print(f"[INFO] 시뮬레이션 결과 Optimal ETL: {max_ETL_sim} nm")

# Plot contour map for simulation results
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(HTL_values, ETL_values, nor_EL_intensity.T, levels=30, cmap="jet", vmin=0, vmax=1)
plt.colorbar(contour, label="Normalized EL Intensity")
ax.set_xlabel("HTL Thickness (nm)")
ax.set_ylabel("ETL Thickness (nm)")
ax.set_title("EL Intensity Map (Simulation)")
plt.show(block=True)

print("\n--- 머신러닝 모델 학습 시작 ---")

# 1. 데이터 추출/수집
data_rows = []
for i, htl in enumerate(HTL_values):
    for j, etl in enumerate(ETL_values):
        data_rows.append({
            'HTL_Thickness': htl,
            'ETL_Thickness': etl,
            'EL_Intensity': nor_EL_intensity[i, j]
        })
data = pd.DataFrame(data_rows)

print("생성된 데이터프레임의 상위 5개 행:")
print(data.head())
print("\n데이터프레임 정보:")
data.info()

# 2. 데이터 전처리
X = data[['HTL_Thickness', 'ETL_Thickness']]
y = data['EL_Intensity']

# LightGBM은 스케일링에 덜 민감하지만, 일관성을 위해 스케일링을 유지합니다.
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
print("\n데이터 스케일링 완료.")

# 3. 모델 선택 및 학습
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
print(f"훈련 데이터 크기: {X_train.shape[0]}, 테스트 데이터 크기: {X_test.shape[0]}")

# --- EDITED PART: Switched to LightGBM for speed and efficiency ---
print("LightGBM 회귀 모델 학습 중...")
# 정확도를 위해 복잡도가 높은 하이퍼파라미터를 설정합니다.
model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42)
model.fit(X_train, y_train.ravel())
print("모델 학습 완료.")

# 4. 모델 평가
# --- 훈련 세트 성능 평가 ---
y_train_pred_scaled = model.predict(X_train).reshape(-1, 1)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
y_train_original = scaler_y.inverse_transform(y_train)
mse_train = mean_squared_error(y_train_original, y_train_pred)
r2_train = r2_score(y_train_original, y_train_pred)

print(f"\n[Machine Learning Model - 훈련 세트 성능 (LightGBM)]")
print(f"훈련 Mean Squared Error (MSE): {mse_train:.6f}")
print(f"훈련 Root Mean Squared Error (RMSE): {np.sqrt(mse_train):.6f}")
print(f"훈련 R-squared (R2): {r2_train:.6f}")

# --- 테스트 세트 성능 평가 ---
y_test_pred_scaled = model.predict(X_test).reshape(-1, 1)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test)
mse_test = mean_squared_error(y_test_original, y_test_pred)
r2_test = r2_score(y_test_original, y_test_pred)

print(f"\n[Machine Learning Model - 테스트 세트 성능 (LightGBM)]")
print(f"테스트 Mean Squared Error (MSE): {mse_test:.6f}")
print(f"테스트 Root Mean Squared Error (RMSE): {np.sqrt(mse_test):.6f}")
print(f"테스트 R-squared (R2): {r2_test:.6f}")

print("\n--- 훈련 세트 vs 테스트 세트 성능 비교 ---")
print(f"R-squared 차이 (훈련 - 테스트): {(r2_train - r2_test):.6f}")

r2_diff_threshold = 0.05
if (r2_train - r2_test) > r2_diff_threshold and r2_train > 0.95:
    print(f"\n경고: 훈련 세트와 테스트 세트 간의 R2 차이가 커서 과적합(Overfitting) 가능성이 있습니다.")
else:
    print("\n훈련 세트와 테스트 세트 성능이 유사하여 모델이 잘 일반화된 것으로 보입니다.")


# 5. 예측된 EL Intensity Map 그리기
print("\n--- 예측 지도 생성 중 ---")
# 시각화를 위한 그리드는 계산 효율성을 위해 원래대로 유지합니다.
HTL_grid = np.arange(10, 101, 2)
ETL_grid = np.arange(10, 101, 2)

grid_X_rows = []
# Create grid points for prediction
for htl in HTL_grid:
    for etl in ETL_grid:
        grid_X_rows.append({'HTL_Thickness': htl, 'ETL_Thickness': etl})
grid_X = pd.DataFrame(grid_X_rows)

# Scale grid points and predict
grid_X_scaled = scaler_X.transform(grid_X)
predicted_el_scaled = model.predict(grid_X_scaled).reshape(-1, 1)
predicted_el = scaler_y.inverse_transform(predicted_el_scaled).flatten()

# Clip values to be non-negative for physical meaning
predicted_el = np.clip(predicted_el, 0, None)

# Reshape for plotting
predicted_el_map = predicted_el.reshape(len(HTL_grid), len(ETL_grid)).T

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
vmax = np.max(predicted_el_map)
contour = ax.contourf(HTL_grid, ETL_grid, predicted_el_map, levels=30, cmap="jet", vmin=0, vmax=vmax if vmax > 0 else 1)
plt.colorbar(contour, label="Predicted Normalized EL Intensity")
ax.set_xlabel("HTL Thickness (nm)")
ax.set_ylabel("ETL Thickness (nm)")
ax.set_title(f"Predicted EL Intensity Map (Machine Learning - LightGBM)")

# Find and mark the optimal point from the predicted map
max_flat_idx = np.argmax(predicted_el)
predicted_max_HTL = grid_X.iloc[max_flat_idx]['HTL_Thickness']
predicted_max_ETL = grid_X.iloc[max_flat_idx]['ETL_Thickness']

ax.axvline(predicted_max_HTL, color='white', linestyle='--', linewidth=1.5)
ax.axhline(predicted_max_ETL, color='white', linestyle='--', linewidth=1.5)
ax.scatter(predicted_max_HTL, predicted_max_ETL, color='red', marker='X', s=150, zorder=5, label=f'Predicted Max\nHTL: {predicted_max_HTL}nm, ETL: {predicted_max_ETL}nm')
ax.legend()

print(f"\n[Machine Learning Prediction] Optimal HTL: {predicted_max_HTL} nm")
print(f"\n[Machine Learning Prediction] Optimal ETL: {predicted_max_ETL} nm")
print(f"[Machine Learning Prediction] Max EL Intensity (from ML map): {np.max(predicted_el):.4f}")

plt.show(block=True)
