import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

df = pd.read_excel("Integrated Macro Data.xlsx", parse_dates=["Date"])
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)

df.dropna(inplace=True)

df["Return"].plot(figsize=(10, 6))
plt.title("Consumer Staples Sector: Monthly Return Over Time")
plt.ylabel("Monthly Return")
plt.xlabel("Date")
plt.axhline(df["Return"].mean(), color='red', linestyle='--')  
plt.show()

# Test return normality (just curious)
from scipy.stats import jarque_bera
jb_stat, jb_p = jarque_bera(df["Return"])
print(f"Jarque-Bera stat = {jb_stat:.4f}, p-value = {jb_p:.4f}")

if jb_p < 0.05:
    print("Reject normality — returns are not normally distributed.")
else:
    print("Fail to reject — returns may be normally distributed.")

from scipy.stats import shapiro

stat, p = shapiro(df["Return"])
print(f"Shapiro-Wilk stat = {stat:.4f}, p-value = {p:.4f}")

if p < 0.05:
    print("Reject normality — not normal.")
else:
    print("Fail to reject — possibly normal.")


plt.figure(figsize=(10,6))
sns.histplot(df["Return"], bins=30, kde=True, color='blue')
plt.title("Distribution of Consumer Staples Sector Monthly Returns")
plt.xlabel("Monthly Return")
plt.ylabel("Frequency")
plt.legend()
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(df[["Treasury_Rate", "CPI", "Unemployment", "VIX"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Independent Variables")
plt.tight_layout()
plt.show()

    
# Standarize independent variables 
features = ["Treasury_Rate", "CPI", "Unemployment", "VIX"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
X = pd.DataFrame(X_scaled, columns=features, index=df.index)


X = sm.add_constant(X)
y = df["Return"]

model = sm.OLS(y, X).fit()
print(model.summary())

intercept = model.params["const"]
coefs = model.params.drop("const")

# Scatter plots
for feature in ["Treasury_Rate", "CPI", "Unemployment", "VIX"]:
    beta = coefs[feature]
    equation = f"y = {intercept:.4f} + ({beta:.4f})·{feature}"

    plt.figure(figsize=(8, 5))
    sns.regplot(x=df[feature], y=df["Return"], line_kws={"color": "red"})
    plt.title(f"Return vs {feature} with Fitted Line")
    plt.xlabel(feature)
    plt.ylabel("Return")
    plt.grid(True)


    plt.text(x=df[feature].min(), y=df["Return"].max(),
             s=equation, fontsize=10, color="black", 
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

    plt.tight_layout()
    plt.show()
    
from statsmodels.tsa.stattools import adfuller

# Check whether return is stationary
result = adfuller(df['Return'])

print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
for key, value in result[4].items():
    print('Critical Value (%s): %.3f' % (key, value))

if result[1] < 0.05:
    print("Reject H₀: The series is stationary.")
else:
    print("Fail to reject H₀: The series is non-stationary.")
