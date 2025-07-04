import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import ttest_ind
import seaborn as sns
import statsmodels.api as sm

df = pd.read_excel('S&P500 indexes.xlsx', parse_dates=['Date'])
df.set_index('Date', inplace=True)

returns = df.pct_change().dropna()
spx_ret = returns['SPX']
spx_ret_scaled = spx_ret * 100

# Select best ARCH(p) model based on AIC/BIC
best_aic = np.inf
best_model = None
for p in range(1, 6):
    try:
        model = arch_model(spx_ret_scaled, vol='ARCH', p=p)
        res = model.fit(disp='off')
        print(f"ARCH({p}): AIC = {res.aic:.2f}, BIC = {res.bic:.2f}")
        if res.aic < best_aic:
            best_aic = res.aic
            best_model = res
    except Exception as e:
        print(f"ARCH({p}) failed: {e}")

# Check residuals
std_resid = best_model.resid / best_model.conditional_volatility
plot_acf(std_resid, lags=20)
plt.title('ACF of Standardized Residuals')
plt.show()
plot_acf(std_resid**2, lags=20)
plt.title('ACF of Squared Standardized Residuals')
plt.show()
pval = het_arch(std_resid.dropna(), nlags=5)[1]
print(f"Post-fit ARCH LM Test p-value: {pval:.4f}")
print(best_model.summary())

sns.histplot(std_resid, kde=True, bins=40)
plt.title("Histogram of Standardized Residuals")
plt.show()

# QQ plot
sm.qqplot(std_resid.dropna(), line='s')
plt.title("QQ Plot of Standardized Residuals")
plt.show()

# Define crisis vs normal periods
vol = best_model.conditional_volatility
mean_vol = vol.mean()
std_vol = vol.std()
threshold = mean_vol + 0.5 * std_vol
regime = pd.Series(np.where(vol > threshold, 'Crisis', 'Normal'), index=vol.index)

vol.plot(figsize=(10, 4), title='Estimated Conditional Volatility (ARCH)')
plt.ylabel('Volatility')
plt.grid(True)
plt.show()

# Histogram of volatility to decide threshold
'''
mean_vol = vol.mean()
std_vol = vol.std()
threshold_1 = mean_vol + 0.5 * std_vol
threshold_1_5 = mean_vol + 1 * std_vol
threshold_2 = mean_vol + 2.0 * std_vol

# Plot histogram
plt.figure(figsize=(10, 5))
plt.hist(vol, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(mean_vol, color='blue', linestyle='--', label='Mean')
plt.axvline(threshold_1, color='orange', linestyle='--', label='Mean + 0.5σ')
plt.axvline(threshold_1_5, color='red', linestyle='--', label='Mean + 1σ')
plt.axvline(threshold_2, color='darkred', linestyle='--', label='Mean + 2σ')
plt.title('Histogram of Conditional Volatility (ARCH)')
plt.xlabel('Volatility')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

log_vol = np.log(vol)

# Recalculate thresholds on log scale
mean_log_vol = log_vol.mean()
std_log_vol = log_vol.std()
thresh1 = mean_log_vol + 1 * std_log_vol
thresh1_5 = mean_log_vol + 1.5 * std_log_vol
thresh2 = mean_log_vol + 2 * std_log_vol

# Plot histogram of log(volatility)
plt.figure(figsize=(10, 5))
plt.hist(log_vol, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
plt.axvline(mean_log_vol, color='blue', linestyle='--', label='Mean (log)')
plt.axvline(thresh1, color='orange', linestyle='--', label='Mean + 1σ')
plt.axvline(thresh1_5, color='red', linestyle='--', label='Mean + 1.5σ')
plt.axvline(thresh2, color='darkred', linestyle='--', label='Mean + 2σ')
plt.title('Histogram of Log(Conditional Volatility)')
plt.xlabel('log(Volatility)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()
'''
# Compare average return and volatility by regime
sector_cols = ['Consumer Staples', 'Consumer Discretionary', 'Technology', 'Industrials']
result = []

for col in sector_cols:
    sector = returns[col].dropna()
    aligned_ret, aligned_regime = sector.align(regime, join='inner')
    crisis_ret = aligned_ret[aligned_regime == 'Crisis']
    normal_ret = aligned_ret[aligned_regime == 'Normal']
    
    result.append({
        'Sector': col,
        'Crisis Mean Return': crisis_ret.mean(),
        'Normal Mean Return': normal_ret.mean(),
        'Crisis Volatility': crisis_ret.std(),
        'Normal Volatility': normal_ret.std(),
        'T-test p-value': ttest_ind(crisis_ret, normal_ret, equal_var=False).pvalue
    })

# Display results
result_df = pd.DataFrame(result)
print(result_df)

aligned_spx_ret, aligned_regime_spx = spx_ret.align(regime, join='inner')
crisis_spx_ret = aligned_spx_ret[aligned_regime_spx == 'Crisis']
normal_spx_ret = aligned_spx_ret[aligned_regime_spx == 'Normal']

market_stats = {
    'Market Crisis Mean Return': crisis_spx_ret.mean(),
    'Market Normal Mean Return': normal_spx_ret.mean(),
    'Market Crisis Volatility': crisis_spx_ret.std(),
    'Market Normal Volatility': normal_spx_ret.std()
}

print("\nMarket Return and Volatility by Regime:")
for k, v in market_stats.items():
    print(f"{k}: {v:.4f}")