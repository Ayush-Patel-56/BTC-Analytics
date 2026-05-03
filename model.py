import numpy as np
import pandas as pd
import scipy.stats as stats
from arch import arch_model

def rolling_entropy(x, window=60, bins=20):
    def ent(v):
        p, _ = np.histogram(v, bins=bins, density=True)
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    return x.rolling(window).apply(ent, raw=True)

def update_params(p, sigma2, bar_sigma2, t):
    err = sigma2 - bar_sigma2
    lr  = p['eta'] / (1 + t**0.55)
    p['gamma'] = np.clip(p['gamma'] + lr * err, 0.01, 0.5)
    return p

def simulate_cyber_gbm(S0, mu, sigma_fig_val, H_val, M_val, redundancy_val, info_filter_val,
                       params, bar_sigma2, nu, n_steps=1, dt=1, eps=1e-6):
    """
    Monte Carlo simulation for a single path using stylized Cyber-GBM parameters.
    H_val and M_val act as shock multipliers for the variance process.
    """
    S = np.zeros(n_steps + 1)
    S[0] = S0
    sigma2 = sigma_fig_val ** 2
    
    # Normalization using historical window max
    
    for t in range(1, n_steps + 1):
        crisis  = (H_val > 0.8) or (M_val > 0.8)
        delta_t = params['delta'] if crisis else 0.0
        
        sigma2 = (
            sigma_fig_val**2 * (1 + params['alpha'] * H_val + delta_t * M_val)
            + params['gamma'] * (bar_sigma2 - sigma2)
        )
        sigma2 *= max(1e-12, redundancy_val)
        sigma2 *= 1 + 0.5 * info_filter_val
        sigma2 = max(eps, min(sigma2, 0.5))
        
        Z   = np.random.standard_t(nu) * np.sqrt((nu - 2) / nu)
        S[t]= S[t-1] * np.exp((mu - 0.5 * sigma2) * dt + np.sqrt(sigma2 * dt) * Z)
        params = update_params(params, sigma2, bar_sigma2, t)
        
    return S

def predict_next_hour(prices, n_sims=10000):
    """
    Takes a Series of prices and predicts the 95% range for the next hour.
    Ensures no peeking.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    
    if len(log_ret) < 100:
        raise ValueError("Insufficient data for GARCH fitting.")

    # Fit FIGARCH (Fractionally Integrated GARCH)
    # Using 'studentst' is critical here to capture the 'fat tails' common in crypto
    am = arch_model(log_ret * 100, vol='FIGARCH', p=1, o=0, q=1, dist='studentst')
    res = am.fit(disp='off')
    
    sigma_fig = res.conditional_volatility / 100
    resid = (log_ret * 100 - res.params['mu']) / res.conditional_volatility
    
    # Degrees of freedom (nu) - capped at 4 to maintain stability in simulation
    nu = max(4, stats.t.fit(resid, floc=0, fscale=1)[0])
    
    # Rolling metrics
    H_series = rolling_entropy(resid)
    M_series = log_ret.abs().rolling(60).mean()
    
    # Current values for simulation
    S0 = prices.iloc[-1]
    mu = log_ret.mean()
    
    latest_H = H_series.iloc[-1]
    latest_M = M_series.iloc[-1]
    latest_sigma = sigma_fig.iloc[-1]
    
    # Max values for normalization (using historical window)
    H_max = H_series.max() if H_series.max() > 0 else 1.0
    M_max = M_series.max() if M_series.max() > 0 else 1.0
    
    H_val = min(latest_H / H_max, 1.0)
    M_val = min(latest_M / M_max, 1.0)
    
    bar_sigma2 = (sigma_fig**2).mean()
    
    # Redundancy and info filter
    # redundancy = 1 + 0.1 * np.log1p(prices.rolling(5).var() / prices.rolling(20).var())
    # info_filter = (H_series > H_series.mean()).astype(float)
    
    r_window = prices.iloc[-20:]
    redundancy_val = 1 + 0.1 * np.log1p(r_window.iloc[-5:].var() / r_window.var())
    if np.isnan(redundancy_val): redundancy_val = 1.0
    
    info_filter_val = 1.0 if latest_H > H_series.mean() else 0.0
    
    # Base params
    α0, δ0 = 0.5, 0.3
    if α0 * H_max + δ0 * M_max >= 1:
        fac = 0.95 / (α0 * H_max + δ0 * M_max)
        α0 *= fac
        δ0 *= fac
    base_params = {'alpha': α0, 'delta': δ0, 'gamma': 0.2, 'kappa': 0.1, 'eta': 1e-3}
    
    # MC Simulation
    sim_results = []
    for _ in range(n_sims):
        path = simulate_cyber_gbm(
            S0, mu, latest_sigma, H_val, M_val, redundancy_val, info_filter_val,
            base_params.copy(), bar_sigma2, nu, n_steps=1
        )
        sim_results.append(path[1])
    
    low95, high95 = np.percentile(sim_results, [2.5, 97.5])
    
    return {
        'current_price': S0,
        'predicted_low': low95,
        'predicted_high': high95,
        'mu': mu,
        'sigma': latest_sigma,
        'nu': nu
    }

def evaluate_prediction(actual, low, high):
    """
    Calculates coverage and Winkler score for a single prediction.
    """
    coverage = int(low <= actual <= high)
    width = high - low
    alpha = 0.05
    if actual < low:
        winkler = width + (2/alpha) * (low - actual)
    elif actual > high:
        winkler = width + (2/alpha) * (actual - high)
    else:
        winkler = width
    return coverage, winkler, width
