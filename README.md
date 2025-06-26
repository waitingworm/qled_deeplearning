# TE-QLED 최적화를 위한 머신러닝 기반 역설계

이 프로젝트는 TE-QLED(Transparent Electrode Quantum Dot Light Emitting Diode)의 최적 두께 조합을 머신러닝을 통해 찾아내는 역설계 시스템입니다.

## 주요 기능

1. 라틴 하이퍼큐브 샘플링을 통한 효율적인 데이터셋 생성
2. oledpy 라이브러리를 활용한 TE-QLED 시뮬레이션
3. XGBoost 기반 대리 모델 학습
4. Optuna를 활용한 베이즈 최적화

## 설치 방법

1. Anaconda 설치 (아직 설치하지 않은 경우):
   - [Anaconda 공식 웹사이트](https://www.anaconda.com/products/distribution)에서 다운로드
   - 설치 가이드에 따라 설치 완료

2. 새로운 Conda 환경 생성 및 활성화:
```bash
# qled_env라는 이름의 새로운 환경 생성
conda create -n qled_env python=3.9

# 환경 활성화
conda activate qled_env
```

3. 필요한 패키지 설치:
```bash
# 기본 패키지 설치
conda install numpy pandas scipy scikit-learn xgboost

# Optuna 설치
conda install -c conda-forge optuna

# oledpy 설치 (pip 사용)
pip install oledpy
```

## 실행 방법

1. Conda 환경 활성화:
```bash
conda activate qled_env
```

2. 데이터셋 생성 및 최적화 실행:
```bash
python main.py
```

## 출력 결과

- `qled_dataset.csv`: 생성된 시뮬레이션 데이터셋
- 최적의 두께 조합과 예상 EQE 값

## 주의사항

- Anaconda가 올바르게 설치되어 있어야 합니다.
- oledpy 라이브러리가 올바르게 설치되어 있어야 합니다.
- 시뮬레이션에 필요한 재료 데이터가 올바른 경로에 있어야 합니다.
- 충분한 메모리와 계산 리소스가 필요합니다.

## 환경 관리 팁

- 환경 비활성화: `conda deactivate`
- 환경 삭제: `conda env remove -n qled_env`
- 설치된 패키지 목록 확인: `conda list`
- 환경 목록 확인: `conda env list` 