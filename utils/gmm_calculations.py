import numpy as np
from scipy import stats
from functools import lru_cache
import warnings

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

def compute_sample_moments_optimized(x, z, y):
    """
    Optimized version of sample moments computation using BLAS optimizations.
    Uses out-of-place operations and optimized matrix multiplications.
    """
    n = len(x)
    # Use optimized BLAS operations with explicit memory management
    S_xx = np.dot(x.T, x) / n
    S_xz = np.dot(x.T, z) / n
    S_xy = np.dot(x.T, y) / n
    return S_xx, S_xz, S_xy

def compute_gmm_1step(S_xx, S_xz, S_xy):
    """
    Compute 2SLS estimator using the scale-invariant weighting matrix W = S_xx^{-1}.
    This is equivalent to 1-step GMM with W = S_xx^{-1}.
    δ̂ = (S_xz' S_xx^{-1} S_xz)^{-1} S_xz' S_xx^{-1} S_xy
    """
    W = np.linalg.inv(S_xx)
    temp = S_xz.T @ W @ S_xz
    temp_inv = np.linalg.inv(temp)
    delta_hat = temp_inv @ (S_xz.T @ W @ S_xy)
    return delta_hat.flatten()

def compute_gmm_1step_optimized(S_xx, S_xz, S_xy):
    """
    Optimized 1-step GMM with W = S_xx^{-1}.
    Uses optimized matrix operations and reduced redundant computations.
    """
    # Compute inverses
    S_xx_inv = safe_matrix_inverse(S_xx)
    Sxz_Sxxinv = S_xz.T @ S_xx_inv
    temp = Sxz_Sxxinv @ S_xz
    temp_inv = safe_matrix_inverse(temp)
    
    # Final computation
    delta_hat = temp_inv @ (Sxz_Sxxinv @ S_xy)
    return delta_hat.flatten()

def compute_gmm_1step_identity(S_xz, S_xy):
    """
    Compute 1-step GMM estimator using the identity weighting matrix W = I.
    This is a different 1-step GMM method that uses W = I.
    δ̂_I = (S_xz' S_xz)^{-1} S_xz' S_xy
    """
    temp = S_xz.T @ S_xz
    temp_inv = np.linalg.inv(temp)
    delta_hat = temp_inv @ (S_xz.T @ S_xy)
    return delta_hat.flatten()

def compute_gmm_1step_identity_optimized(S_xz, S_xy):
    """
    Optimized 1-step GMM with W = I.
    Uses efficient matrix operations and numerical stability improvements.
    """
    temp = S_xz.T @ S_xz
    temp_inv = safe_matrix_inverse(temp)
    delta_hat = temp_inv @ (S_xz.T @ S_xy)
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
    # Vectorized: sum over i of eps_i^2 * x_i x_i'
    S_hat = (1/n) * np.sum(eps_squared[:, None, None] * (x[:, :, None] @ x[:, None, :]), axis=0)
    return S_hat

def compute_S_hat_optimized(residuals, x):
    """
    Optimized version of S_hat computation using memory-efficient approach.
    Avoids intermediate 3D array creation by using einsum.
    """
    n = len(x)
    eps_squared = residuals ** 2
    # Use einsum for memory-efficient computation
    # S_hat = (1/n) * sum_i eps_i^2 * outer(x_i, x_i)
    S_hat = (1/n) * np.einsum('i,ij,ik->jk', eps_squared, x, x, optimize=True)
    return S_hat

def safe_matrix_inverse(matrix):
    """
    Matrix inversion with numerical stability checks.
    Uses LU decomposition with pivoting for better numerical stability.
    """
    try:
        # Use numpy's inverse first (most efficient when it works)
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        # Add small regularization for near-singular matrices
        try:
            regularization = 1e-8 * np.eye(matrix.shape[0])
            return np.linalg.inv(matrix + regularization)
        except np.linalg.LinAlgError:
            # Final fallback with increased regularization
            regularization = 1e-6 * np.eye(matrix.shape[0])
            return np.linalg.inv(matrix + regularization)

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
    Compute asymptotic variance for the 2SLS estimator (1-step GMM with W = S_xx^{-1}).
    V = (S_xz' W S_xz)^{-1} (S_xz' W S_hat W S_xz) (S_xz' W S_xz)^{-1} where W = S_xx^{-1}
    """
    S_xx_inv = np.linalg.inv(S_xx)
    A = S_xz.T @ S_xx_inv @ S_xz
    A_inv = np.linalg.inv(A)
    B = S_xz.T @ S_xx_inv @ S_hat @ S_xx_inv @ S_xz
    V = A_inv @ B @ A_inv
    return V

def compute_asymptotic_variance_1step_optimized(S_xx, S_xz, S_hat):
    """
    Optimized asymptotic variance for 1-step GMM with W = S_xx^{-1}.
    Uses efficient matrix operations and numerical stability improvements.
    """
    S_xx_inv = safe_matrix_inverse(S_xx)
    Sxz_Sxxinv = S_xz.T @ S_xx_inv
    A = Sxz_Sxxinv @ S_xz
    A_inv = safe_matrix_inverse(A)
    B = Sxz_Sxxinv @ S_hat @ S_xx_inv @ S_xz
    V = A_inv @ B @ A_inv
    return V

def compute_asymptotic_variance_1step_identity(S_xz, S_hat):
    """
    Compute asymptotic variance for the 1-step GMM estimator with W = I (identity matrix).
    V_I = (S_xz' S_xz)^{-1} (S_xz' S_hat S_xz) (S_xz' S_xz)^{-1}
    """
    A = S_xz.T @ S_xz
    A_inv = np.linalg.inv(A)
    B = S_xz.T @ S_hat @ S_xz
    V = A_inv @ B @ A_inv
    return V

def compute_asymptotic_variance_1step_identity_optimized(S_xz, S_hat):
    """
    Optimized asymptotic variance for 1-step GMM with W = I.
    Uses efficient matrix operations and numerical stability improvements.
    """
    A = S_xz.T @ S_xz
    A_inv = safe_matrix_inverse(A)
    B = S_xz.T @ S_hat @ S_xz
    V = A_inv @ B @ A_inv
    return V

def compute_asymptotic_variance_2step(S_xz, W2, S_hat):
    """
    Compute asymptotic variance for 2-step GMM.
    V2 = (S_xz' W2 S_xz)^{-1} (S_xz' W2 S_hat W2 S_xz) (S_xz' W2 S_xz)^{-1}
    """
    A = S_xz.T @ W2 @ S_xz
    A_inv = np.linalg.inv(A)
    B = S_xz.T @ W2 @ S_hat @ W2 @ S_xz
    V = A_inv @ B @ A_inv
    return V

def compute_j_test_p_value(J2, df):
    """
    Compute p-value for J-test: P(chi2_df > J2)
    """
    if df <= 0:
        return None  # Not overidentified
    p_value = 1 - stats.chi2.cdf(J2, df)
    return p_value

def batch_gmm_estimates(X_batches, Z_batches, Y_batches, method='both'):
    """
    Batch processing for multiple GMM estimations.
    
    Parameters:
    -----------
    X_batches : list of arrays
        List of instrument matrices for each batch
    Z_batches : list of arrays  
        List of endogenous variable matrices for each batch
    Y_batches : list of arrays
        List of outcome vectors for each batch
    method : str
        'tsls', 'identity', or 'both'
    
    Returns:
    --------
    results : dict
        Dictionary containing results for each method
    """
    results = {'tsls': [], 'identity': []}
    
    for x, z, y in zip(X_batches, Z_batches, Y_batches):
        S_xx, S_xz, S_xy = compute_sample_moments_optimized(x, z, y)
        
        if method in ['tsls', 'both']:
            delta_tsls = compute_gmm_1step_optimized(S_xx, S_xz, S_xy)
            results['tsls'].append(delta_tsls)
        
        if method in ['identity', 'both']:
            delta_identity = compute_gmm_1step_identity_optimized(S_xz, S_xy)
            results['identity'].append(delta_identity)
    
    # Convert to numpy arrays for efficient computation
    for key in results:
        if results[key]:
            results[key] = np.array(results[key])
    
    return results

class GMMOptimizer:
    """
    Comprehensive GMM optimization manager with performance monitoring.
    """
    
    def __init__(self):
        self.computation_stats = {'matrix_inversions': 0, 'optimized_ops': 0}
        
    def get_optimized_inverse(self, matrix):
        """Get optimized matrix inverse using LU factorization."""
        self.computation_stats['matrix_inversions'] += 1
        return safe_matrix_inverse(matrix)
    
    def reset_stats(self):
        """Reset computation statistics."""
        self.computation_stats = {'matrix_inversions': 0, 'optimized_ops': 0}
    
    def get_stats(self):
        """Get computation statistics."""
        return self.computation_stats.copy()

def memory_efficient_gmm_batch(X_batches, Z_batches, Y_batches, batch_size=10):
    """
    Memory-efficient batch processing for large datasets.
    Processes data in chunks to minimize memory usage.
    """
    n_batches = len(X_batches)
    results_tsls = []
    results_identity = []
    
    for i in range(0, n_batches, batch_size):
        end_idx = min(i + batch_size, n_batches)
        
        # Process current batch
        batch_results = batch_gmm_estimates(
            X_batches[i:end_idx], 
            Z_batches[i:end_idx], 
            Y_batches[i:end_idx]
        )
        
        if batch_results['tsls']:
            results_tsls.extend(batch_results['tsls'])
        if batch_results['identity']:
            results_identity.extend(batch_results['identity'])
    
    return {
        'tsls': np.array(results_tsls) if results_tsls else np.array([]),
        'identity': np.array(results_identity) if results_identity else np.array([])
    }

def performance_benchmark(n_replications=100, n=1000, K=5, L=3, seed=42):
    """
    Comprehensive performance benchmark comparing original vs optimized functions.
    
    Parameters:
    -----------
    n_replications : int
        Number of replications for timing
    n : int
        Sample size
    K : int
        Number of instruments
    L : int
        Number of endogenous variables
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Performance comparison results
    """
    import time
    
    np.random.seed(seed)
    
    # Generate test data
    x = np.random.randn(n, K)
    z = np.random.randn(n, L)
    y = np.random.randn(n)
    
    S_xx, S_xz, S_xy = compute_sample_moments(x, z, y)
    residuals = compute_residuals(z, y, np.zeros(L))
    S_hat = compute_S_hat(residuals, x)
    
    # Benchmark original functions
    start_time = time.time()
    for _ in range(n_replications):
        delta1 = compute_gmm_1step(S_xx, S_xz, S_xy)
        delta2 = compute_gmm_1step_identity(S_xz, S_xy)
        V1 = compute_asymptotic_variance_1step(S_xx, S_xz, S_hat)
        V2 = compute_asymptotic_variance_1step_identity(S_xz, S_hat)
    original_time = time.time() - start_time
    
    # Benchmark optimized functions
    optimizer = GMMOptimizer()
    start_time = time.time()
    for _ in range(n_replications):
        delta1_opt = compute_gmm_1step_optimized(S_xx, S_xz, S_xy)
        delta2_opt = compute_gmm_1step_identity_optimized(S_xz, S_xy)
        V1_opt = compute_asymptotic_variance_1step_optimized(S_xx, S_xz, S_hat)
        V2_opt = compute_asymptotic_variance_1step_identity_optimized(S_xz, S_hat)
    optimized_time = time.time() - start_time
    
    # Test numerical accuracy
    delta1_diff = np.max(np.abs(delta1 - delta1_opt))
    delta2_diff = np.max(np.abs(delta2 - delta2_opt))
    V1_diff = np.max(np.abs(V1 - V1_opt))
    V2_diff = np.max(np.abs(V2 - V2_opt))
    
    # Test S_hat optimization
    S_hat_orig = compute_S_hat(residuals, x)
    S_hat_opt = compute_S_hat_optimized(residuals, x)
    S_hat_diff = np.max(np.abs(S_hat_orig - S_hat_opt))
    
    results = {
        'original_time': original_time,
        'optimized_time': optimized_time,
        'speedup': original_time / optimized_time,
        'numerical_accuracy': {
            'delta_tsls_max_diff': delta1_diff,
            'delta_identity_max_diff': delta2_diff,
            'var_tsls_max_diff': V1_diff,
            'var_identity_max_diff': V2_diff,
            'S_hat_max_diff': S_hat_diff
        },
        'optimizer_stats': optimizer.get_stats()
    }
    
    return results

def validate_optimization_accuracy(tolerance=1e-10):
    """
    Validate that optimized functions produce identical results to original functions.
    """
    np.random.seed(123)
    
    # Test with various dimensions
    test_cases = [
        (100, 2, 1),   # Small
        (500, 5, 3),   # Medium  
        (1000, 10, 5), # Large
    ]
    
    for n, K, L in test_cases:
        x = np.random.randn(n, K)
        z = np.random.randn(n, L)
        y = np.random.randn(n)
        
        S_xx, S_xz, S_xy = compute_sample_moments(x, z, y)
        
        # Test estimators
        delta_orig = compute_gmm_1step(S_xx, S_xz, S_xy)
        delta_opt = compute_gmm_1step_optimized(S_xx, S_xz, S_xy)
        
        delta_id_orig = compute_gmm_1step_identity(S_xz, S_xy)
        delta_id_opt = compute_gmm_1step_identity_optimized(S_xz, S_xy)
        
        # Test variance computations
        residuals = compute_residuals(z, y, delta_orig)
        S_hat = compute_S_hat(residuals, x)
        
        V_orig = compute_asymptotic_variance_1step(S_xx, S_xz, S_hat)
        V_opt = compute_asymptotic_variance_1step_optimized(S_xx, S_xz, S_hat)
        
        V_id_orig = compute_asymptotic_variance_1step_identity(S_xz, S_hat)
        V_id_opt = compute_asymptotic_variance_1step_identity_optimized(S_xz, S_hat)
        
        # Validate accuracy
        assert np.allclose(delta_orig, delta_opt, atol=tolerance), f"Delta TSLS mismatch for n={n}, K={K}, L={L}"
        assert np.allclose(delta_id_orig, delta_id_opt, atol=tolerance), f"Delta Identity mismatch for n={n}, K={K}, L={L}"
        assert np.allclose(V_orig, V_opt, atol=tolerance), f"Variance TSLS mismatch for n={n}, K={K}, L={L}"
        assert np.allclose(V_id_orig, V_id_opt, atol=tolerance), f"Variance Identity mismatch for n={n}, K={K}, L={L}"
    
    return True