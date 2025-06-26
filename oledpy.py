import numpy as np
import pandas as pd
from scipy.stats import qmc
import os

# 버전 정보
__version__ = '1.0.0'

def simulate_device(structure, mode='TE', wavelength_range=(400, 750), pl_data=None, nk_data=None):
    """
    TE-QLED 시뮬레이션을 실행하는 함수
    
    Parameters:
    -----------
    structure : list
        소자 구조 정보를 담은 리스트
    mode : str
        시뮬레이션 모드 ('TE' 또는 'TM')
    wavelength_range : tuple
        파장 범위 (시작, 끝)
    pl_data : pandas.DataFrame
        PL 데이터
    nk_data : pandas.DataFrame
        nk 데이터
    
    Returns:
    --------
    dict
        시뮬레이션 결과 (EQE_max, FWHM, peak_wavelength)
    """
    try:
        # PL 데이터에서 최대값과 FWHM 계산
        wavelength = pl_data['Wavelength (nm)']
        intensity = pl_data['Intensity']
        
        # 최대값 찾기
        max_intensity = np.max(intensity)
        max_idx = np.argmax(intensity)
        peak_wavelength = wavelength[max_idx]
        
        # FWHM 계산
        half_max = max_intensity / 2
        left_idx = np.argmin(np.abs(intensity[:max_idx] - half_max))
        right_idx = max_idx + np.argmin(np.abs(intensity[max_idx:] - half_max))
        fwhm = wavelength[right_idx] - wavelength[left_idx]
        
        # EQE 계산 (예시: PL 강도와 두께를 고려한 간단한 계산)
        total_thickness = sum(layer[1] for layer in structure)
        eqe_max = max_intensity * (1 - np.exp(-total_thickness/100))  # 간단한 모델
        
        return {
            'eqe_max': eqe_max,
            'fwhm': fwhm,
            'peak_wavelength': peak_wavelength
        }
        
    except Exception as e:
        print(f"시뮬레이션 중 오류 발생: {e}")
        return {
            'eqe_max': 0,
            'fwhm': 0,
            'peak_wavelength': 0
        }

# 버전 정보 출력
print(f"oledpy 라이브러리 버전: {__version__}")

# --- 1. 시뮬레이션 파라미터 정의 ---
# 최적화할 소자 층의 두께 범위를 정의합니다. (단위: nm)
param_space = {
    'ETL_thickness': (20, 80),
    'EML_thickness': (10, 40),
    'HTL_thickness': (20, 80)
}
# 생성할 데이터 샘플 개수
n_samples = 200

# --- 2. 라틴 하이퍼큐브 샘플링 (LHS) ---
# 설계 공간에서 효율적으로 샘플을 추출합니다.
sampler = qmc.LatinHypercube(d=len(param_space))
sample_points = sampler.random(n=n_samples)

# 샘플링된 0-1 사이의 값을 실제 두께 범위로 변환합니다.
l_bounds = [v[0] for v in param_space.values()]
u_bounds = [v[1] for v in param_space.values()]
scaled_samples = qmc.scale(sample_points, l_bounds, u_bounds)

# 나중에 분석하기 쉽도록 DataFrame으로 만들어 줍니다.
input_df = pd.DataFrame(scaled_samples, columns=param_space.keys())
print("✅ LHS 샘플링 완료. 시뮬레이션할 두께 조합을 생성했습니다.")
print(input_df.head())

# --- 3. oledpy 시뮬레이션 실행 ---
# 결과를 저장할 리스트를 미리 만들어 둡니다.
results_list = []

# 여기에 실제 사용할 재료의 광학 상수(.txt 파일 등) 경로를 지정해야 합니다.
# 예시: material_path = 'C:/MyData/materials/'
# material_nk = oledpy.material_load(material_path)

print("\n--- oledpy 시뮬레이션 시작 ---")

for index, row in input_df.iterrows():
    try:
        # 3-1. 소자 구조 정의
        # oledpy에서 사용하는 형식에 맞게 각 층을 리스트로 정의합니다.
        # [재료 이름, 두께 (nm), 층 종류]
        # 'ITO', 'Ag' 등 다른 층들은 고정된 두께를 사용한다고 가정합니다.
        # '재료이름'은 실제 가지고 계신 nk 데이터 파일 이름과 일치해야 합니다.
        device_structure = [
            ['substrate', 1, 'substrate'],
            ['ITO', 150, 'anode'],
            ['HTL_material_name', row['HTL_thickness'], 'HTL'],
            ['EML_material_name', row['EML_thickness'], 'EML'],
            ['ETL_material_name', row['ETL_thickness'], 'ETL'],
            ['Al', 100, 'cathode']
        ]

        # 3-2. 시뮬레이션 실행
        # 이 부분은 oledpy의 실제 함수명과 옵션에 따라 달라질 수 있습니다.
        # TE-QLED 모드로, 400-750nm 파장 범위에서 계산한다고 가정합니다.
        # 실제로는 oledpy.device.Device, oledpy.simulation.run 등의 함수를 사용할 것입니다.
        # 아래는 가상의 함수 호출입니다. 실제 oledpy 문서를 참고하여 수정해야 합니다.
        
        # result_obj = oledpy.calculate_eqe(
        #     structure=device_structure,
        #     materials=material_nk,
        #     mode='TE',
        #     wavelength_range=(400, 750)
        # )

        # 3-3. 결과 추출 (가상 결과)
        # 이 부분은 시뮬레이션 결과 객체(result_obj)에서 실제 값을 추출해야 합니다.
        # 지금은 시뮬레이션이 성공했다고 가정하고 가상의 값을 생성합니다.
        eqe_max = 25 - np.sqrt((row['ETL_thickness'] - 55)**2 + (row['EML_thickness'] - 22)**2) / 10 + np.random.randn() * 0.1
        fwhm = 22 + np.random.rand() * 2
        peak_wavelength = 530 + (row['EML_thickness'] - 25) * 0.5
        
        current_result = {
            'EQE_max': eqe_max,
            'FWHM': fwhm,
            'peak_wavelength': peak_wavelength,
            'simulation_status': 'success'
        }

    except Exception as e:
        # 시뮬레이션 중 특정 두께 조합에서 에러가 발생할 경우를 대비합니다.
        print(f"시뮬레이션 오류 발생 (index: {index}): {e}")
        current_result = {
            'EQE_max': np.nan,
            'FWHM': np.nan,
            'peak_wavelength': np.nan,
            'simulation_status': 'fail'
        }
    
    results_list.append(current_result)
    
    # 진행 상황 출력
    if (index + 1) % 20 == 0:
        print(f"  시뮬레이션 진행률: {index + 1} / {n_samples}")

print("--- oledpy 시뮬레이션 완료 ---")


# --- 4. 데이터셋 완성 및 저장 ---
# 시뮬레이션 결과를 DataFrame으로 변환합니다.
output_df = pd.DataFrame(results_list)

# 초기 입력 DataFrame과 결과 DataFrame을 합칩니다.
full_dataset = pd.concat([input_df, output_df], axis=1)

# 시뮬레이션에 실패한 데이터는 분석에서 제외합니다.
full_dataset.dropna(inplace=True)

# 완성된 데이터셋을 CSV 파일로 저장하여 나중에 계속 사용할 수 있게 합니다.
dataset_path = "qled_dataset.csv"
full_dataset.to_csv(dataset_path, index=False)

print(f"\n✅ 데이터셋 생성 완료! 총 {len(full_dataset)}개의 성공적인 데이터를 확보했습니다.")
print(f"결과가 '{dataset_path}' 파일에 저장되었습니다.")
print("--- 생성된 최종 데이터셋 (상위 5개) ---")
print(full_dataset.head())