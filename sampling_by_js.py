import numpy as np   ## 전체 lhs100회  국소 lhs 50회 중ㄱ간
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────────
# ML 파이프라인: df 변수는 위에서 생성된 데이터프레임 (HTL, ETL, EL_intensity)

# 1) 특징 및 타깃 정의
X = df[["HTL", "ETL"]].values.astype(float)
y = df["EL_intensity"].values.astype(float)

# 2) 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3) 스케일링 + 2차 다항특성 생성
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_p = poly.fit_transform(X_train_s)
X_test_p  = poly.transform(X_test_s)

# 4) 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_p, y_train)

# 5) 예측
y_train_pred = model.predict(X_train_p)
y_test_pred  = model.predict(X_test_p)

# 6) 성능 지표 계산
r2_train   = r2_score(y_train, y_train_pred)
r2_test    = r2_score(y_test,  y_test_pred)
mae_train  = mean_absolute_error(y_train, y_train_pred)
mae_test   = mean_absolute_error(y_test,  y_test_pred)
rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
rmse_test  = mean_squared_error(y_test,  y_test_pred, squared=False)

mean_train = y_train.mean()
mean_test  = y_test.mean()

# 7) 메트릭 테이블
metrics = pd.DataFrame([
    ["Train R²",   f"{r2_train:.3f}",   f"{r2_train*100:.1f}%"],
    ["Test  R²",   f"{r2_test:.3f}",    f"{r2_test*100:.1f}%"],
    ["Train MAE",  f"{mae_train:.4e}",  f"{mae_train/mean_train*100:.1f}%"],
    ["Test  MAE",  f"{mae_test:.4e}",   f"{mae_test/mean_test*100:.1f}%"],
    ["Train RMSE", f"{rmse_train:.4e}", f"{rmse_train/mean_train*100:.1f}%"],
    ["Test  RMSE", f"{rmse_test:.4e}",  f"{rmse_test/mean_test*100:.1f}%"],
], columns=["Metric", "Absolute", "% of mean"])

# 8) 결과 출력
print(metrics.to_string(index=False))

# 9) 패리티 플롯
plt.figure(figsize=(5,5))
plt.scatter(y_test, y_test_pred, alpha=0.7)
mn, mx = y.min(), y.max()
plt.plot([mn, mx], [mn, mx], 'k--')
plt.xlabel("실제 EL intensity")
plt.ylabel("예측 EL intensity")
plt.title("Parity Plot (Test set)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 10) 컨투어 플롯 (예측 표면)
grid_htl = np.linspace(X[:,0].min(), X[:,0].max(), 100)
grid_etl = np.linspace(X[:,1].min(), X[:,1].max(), 100)
GH, GE   = np.meshgrid(grid_htl, grid_etl)
grid = np.vstack([GH.ravel(), GE.ravel()]).T
grid_s = scaler.transform(grid)
grid_p = poly.transform(grid_s)
GI     = model.predict(grid_p).reshape(GH.shape)

plt.figure(figsize=(6,5))
cs = plt.contourf(GH, GE, GI, levels=30, cmap="turbo")
plt.colorbar(cs, label="Predicted EL intensity")
plt.scatter(X[:,0], X[:,1], c="white", s=20, edgecolor="k", alpha=0.6)
plt.xlabel("HTL thickness (nm)")
plt.ylabel("ETL thickness (nm)")
plt.title("ML-predicted EL Surface & Sample Points")
plt.tight_layout()
plt.show()
