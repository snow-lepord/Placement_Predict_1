# Session 06: Ordinary Least Squares (OLS) Linear Regression Analysis
# Optimizing MSE loss and examining residual error distributions.
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import config

os.makedirs(config.REGRESSION_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(config.SPLITS_DIR, "train.csv"))
data = train[["AptitudeTestScore", "CodingTestScore"]].dropna()
X = data[["AptitudeTestScore"]]
y = data["CodingTestScore"]

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

slope = model.coef_[0]
intercept = model.intercept_
mse = mean_squared_error(y, predictions)
rmse = mse ** 0.5
mae = mean_absolute_error(y, predictions)
r2 = r2_score(y, predictions)

# Diagnostic Plots
plt.figure(figsize=(7, 5))
plt.scatter(X, y, alpha=0.2, s=12, color="#4f46e5", label="Observed Scores")
plt.plot(X, predictions, color="#f43f5e", linewidth=2.5, label=f"OLS Line (y = {slope:.2f}x + {intercept:.2f})")
plt.xlabel("Aptitude Test Score")
plt.ylabel("Coding Test Score")
plt.title("Aptitude vs Coding Test Score OLS Fit", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(config.REGRESSION_DIR, "06_linear_regression_fit.png"), dpi=120)
plt.close()

residuals = y - predictions
plt.figure(figsize=(7, 5))
plt.scatter(predictions, residuals, alpha=0.2, s=12, color="#06b6d4")
plt.axhline(0, color="#f43f5e", linewidth=1.5, linestyle="--")
plt.xlabel("Predicted Coding Test Score")
plt.ylabel("Residuals (Observed - Predicted)")
plt.title("Residual Distribution across Predictions", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.REGRESSION_DIR, "06_residual_plot.png"), dpi=120)
plt.close()

with open(os.path.join(config.REGRESSION_DIR, "linear_regression_report.txt"), "w") as f:
    f.write(f"Slope: {round(slope, 3)}\nIntercept: {round(intercept, 3)}\n")
    f.write(f"MSE: {round(mse, 3)}\nRMSE: {round(rmse, 3)}\nMAE: {round(mae, 3)}\nR2 score: {round(r2, 3)}\n")

print(f"Session 06 OLS Regression evaluated. R2: {round(r2, 3)}, MSE: {round(mse, 3)}")
