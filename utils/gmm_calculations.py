import numpy as np
from scipy import stats

def compute_sample_moments(x, z, y):
    """
    Compute sample moments for GMM.
    S_xx: (1/n) sum x_i x_i'  (K x K)
    S_xz: (1/n) sum x_i z_i'  (K x L)
    S_xy: (1/n) sum x_i y_i   (K x 1)
    """
    n = len(x)
    S_xx = (1/n) * (x.T @ x)
    S_xz = (1/n) * (x.T @ z)
    S_xy = (1/n) * (x.T @ y)
    return S_xx, S_xz, S_xy

def compute_gmm_1step(S_xx, S_xz, S_xy):
    """
    Compute 1-step GMM estimator with W = I.
    δ̂¹ = (S_xz' S_xx^{-1} S_xz)^{-1} S_xz' S_xx^{-1} S_xy
    """
    S_xx_inv = np.linalg.inv(S_xx)
    temp = S_xz.T @ S_xx_inv @ S_xz
    temp_inv = np.linalg.inv(temp)
    delta_hat = temp_inv @ (S_xz.T @ S_xx_inv @ S_xy)
    return delta_hat.flatten()

def compute_residuals(z, y, delta_hat):
    """
    Compute residuals: ε̂_i = y_i - z_i' δ̂
    """
    return y - z @ delta_hat

def compute_g_n(x, residuals):
    """
    Compute g_n(δ̂) = (1/n) sum x_i ε̂_i
    """
    n = len(x)
    return (1/n) * (x.T @ residuals)

def compute_S_hat(residuals, x):
    """
    Compute Ŝ = (1/n) Σ ε̂_i² x_i x_i'
    The covariance matrix of the moment conditions.
    """
    n = len(x)
    eps_squared = residuals ** 2
    # Outer product: for each i, eps_i^2 * x_i x_i'
    # Sum over i: (1/n) sum eps_i^2 x_i x_i'
    S_hat = (1/n) * sum(eps_squared[i] * np.outer(x[i], x[i]) for i in range(n))
    return S_hat

def compute_gmm_2step(S_xx, S_xz, S_xy, W2):
    """
    Compute 2-step GMM estimator with given weighting matrix W2.
    δ̂² = (S_xz' W2 S_xz)^{-1} S_xz' W2 S_xy
    """
    try:
        temp = S_xz.T @ W2 @ S_xz
        temp_inv = np.linalg.inv(temp)
        delta_hat = temp_inv @ (S_xz.T @ W2 @ S_xy)
        return delta_hat.flatten(), True
    except np.linalg.LinAlgError:
        return None, False

def compute_J_stat(g_n, W2, n):
    """
    Compute J-statistic: J² = n * g_n' W2 g_n
    """
    try:
        J2 = n * (g_n.T @ W2 @ g_n)
        return J2.item(), True
    except:
        return None, False

def compute_asymptotic_variance_1step(S_xx, S_xz, S_hat):
    """
    Compute asymptotic variance for 1-step GMM.
    V1 = (S_xz' S_xx^{-1} S_xz)^{-1} S_xz' S_xx^{-1} S_hat S_xx^{-1} S_xz (S_xz' S_xx^{-1} S_xz)^{-1}
    """
    S_xx_inv = np.linalg.inv(S_xx)
    A = S_xz.T @ S_xx_inv @ S_xz
    A_inv = np.linalg.inv(A)
    B = S_xz.T @ S_xx_inv @ S_hat @ S_xx_inv @ S_xz
    V = A_inv @ B @ A_inv
    return V

def compute_asymptotic_variance_2step(S_xz, W2):
    """
    Compute asymptotic variance for 2-step GMM.
    V2 = (S_xz' W2 S_xz)^{-1}
    """
    A = S_xz.T @ W2 @ S_xz
    V = np.linalg.inv(A)
    return V

def compute_j_test_p_value(J2, df):
    """
    Compute p-value for J-test: P(chi2_df > J2)
    """
    if df <= 0:
        return None  # Not overidentified
    p_value = 1 - stats.chi2.cdf(J2, df)
    return p_value