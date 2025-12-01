import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import scipy.stats as stats
from utils.gmm_calculations import *

def generate_data(K, L, n, seed, delta_true=None, dgp_type="Homoskedastic", hetero_level=0.1):
    np.random.seed(seed)

    # Generate x_i: multivariate normal, mean 0, covariance I_K
    x = np.random.multivariate_normal(np.zeros(K), np.eye(K), n)

    # Generate Π: L x K matrix, random but full rank
    # Ensure full rank by regenerating if necessary
    Pi = np.random.randn(L, K)
    while np.linalg.matrix_rank(Pi) < min(L, K):
        Pi = np.random.randn(L, K)

    # Noise for z: v_i ~ N(0, sigma_v I_L)
    if dgp_type == "High Endogeneity":
        sigma_v = 0.01
    elif dgp_type == "Low Endogeneity":
        sigma_v = 1.0
    else:
        sigma_v = 0.1
    v = np.random.multivariate_normal(np.zeros(L), sigma_v * np.eye(L), n)

    # z_i = Pi @ x_i + v_i
    z = x @ Pi.T + v

    # If delta_true not provided, generate random
    if delta_true is None:
        delta_true = np.random.uniform(-5, 5, L)

    # ε_i
    if dgp_type == "Heteroskedastic (Linear)":
        var_eps = hetero_level * (1 + np.abs(x[:, 0]))
        eps = np.random.normal(0, np.sqrt(var_eps), n)
    elif dgp_type == "Heteroskedastic (Quadratic)":
        var_eps = hetero_level * (1 + x[:, 0]**2)
        eps = np.random.normal(0, np.sqrt(var_eps), n)
    elif dgp_type == "Heteroskedastic (Exponential)":
        var_eps = hetero_level * np.exp(np.abs(x[:, 0]))
        eps = np.random.normal(0, np.sqrt(var_eps), n)
    elif dgp_type == "Invalid Instruments (Not Exogenous)":
        # Create invalid instruments by correlating x with error term
        # ε_i = eps_base + alpha * x_i0 + eps_pure
        eps_base = np.random.normal(0, 0.05, n)
        alpha = 0.3  # Strength of correlation between instrument and error term
        eps_pure = np.random.normal(0, 0.1, n)
        eps = eps_base + alpha * x[:, 0] + eps_pure
    else:
        eps = np.random.normal(0, 0.1, n)

    # y_i = z_i @ delta_true + eps
    y = z @ delta_true + eps

    return x, z, y, delta_true
@st.cache_data(show_spinner=True)
def run_monte_carlo_simulations(K, L, n, M, seed, dgp_type, hetero_level, gmm_method, delta_true):
    # Generate all data and compute 1-step TSLS estimates vectorized
    x_all, z_all, y_all, delta_1_all = generate_and_estimate_tensor(n, M, K, L, seed, dgp_type, hetero_level, delta_true)

    # Compute moments for identity method
    S_xz_all = np.einsum('mni,mnj->mij', x_all, z_all) / n
    S_xy_all = np.einsum('mni,mn->mi', x_all, y_all) / n

    # Compute 1-step identity estimates vectorized
    delta_1_I_all = compute_vectorized_1step_identity(S_xz_all, S_xy_all)

    # Select the appropriate 1-step estimates based on method
    if gmm_method == "W = S_xx⁻¹ (TSLS Equivalent)":
        delta_hats_1 = delta_1_all
        delta_hats_1_I = delta_1_I_all
        delta_1_selected = delta_1_all
    else:  # W = I
        delta_hats_1 = delta_1_I_all
        delta_hats_1_I = delta_1_all
        delta_1_selected = delta_1_I_all

    # Compute residuals for selected method: (M, n)
    residuals_1_all = y_all - np.einsum('mnj,mj->mn', z_all, delta_1_selected)

    # Compute 2-step estimates vectorized
    S_xx_all = np.einsum('mni,mnj->mij', x_all, x_all) / n
    delta_2_all, J2_all, p_values_all = compute_vectorized_2step(S_xx_all, S_xz_all, S_xy_all, residuals_1_all, x_all)

    return delta_hats_1, delta_hats_1_I, delta_2_all, J2_all, p_values_all

def parse_pasted_data(data_str):
    try:
        df = pd.read_csv(StringIO(data_str))
        # Assume columns: x1, x2, ..., z1, z2, ..., y
        # Need to infer K, L from columns
        cols = df.columns
        x_cols = [c for c in cols if c.startswith('x')]
        z_cols = [c for c in cols if c.startswith('z')]
        y_cols = [c for c in cols if c.startswith('y')]
        if len(y_cols) != 1:
            raise ValueError("Must have exactly one y column")
        K = len(x_cols)
        L = len(z_cols)
        if K == 0 or L == 0:
            raise ValueError("Must have x and z columns")
        x = df[x_cols].values
        z = df[z_cols].values
        y = df[y_cols[0]].values
        n = len(df)
        return x, z, y, K, L, n
    except Exception as e:
        raise ValueError(f"Error parsing data: {str(e)}")

def check_identification(x, z, L):
    Sigma_xz = np.mean(x[:, :, None] * z[:, None, :], axis=0)  # K x L
    rank = np.linalg.matrix_rank(Sigma_xz)
    return rank == L, rank

def matrix_to_latex(matrix, name):
    if matrix.ndim == 1:
        return rf"{name} = \begin{{pmatrix}} {' & '.join([f'{x:.4f}' for x in matrix])} \end{{pmatrix}}"
    elif matrix.ndim == 2:
        rows = []
        for row in matrix:
            rows.append(' & '.join([f'{x:.4f}' for x in row]))
        return rf"{name} = \begin{{pmatrix}} {' \\\\ '.join(rows)} \end{{pmatrix}}"
    else:
        return f"{name} = {matrix}"

st.title("GMM Data Generator and Demonstrator")

st.markdown("""
This app demonstrates the Generalized Method of Moments (GMM) estimation for linear models with endogenous variables.
It allows you to generate synthetic data or paste your own dataset, visualize the data and data generating process,
perform 1-step and 2-step GMM estimation, and compare the estimators through simulations with Hansen J-tests.
""")

st.sidebar.header("Parameters")

K = st.sidebar.slider("K (number of instruments)", 1, 10, 2)
L = st.sidebar.slider("L (number of endogenous variables)", 1, 10, 1)
n = st.sidebar.slider("n (sample size)", 100, 10000, 1000)
seed = st.sidebar.number_input("Seed", value=42, min_value=0)
M = st.sidebar.slider("M (number of simulations)", 100, 5000, 1000)
dgp_type = st.sidebar.selectbox("Data Generating Process", ["Homoskedastic", "Heteroskedastic (Linear)", "Heteroskedastic (Quadratic)", "Heteroskedastic (Exponential)", "High Endogeneity", "Low Endogeneity", "Invalid Instruments (Not Exogenous)"])

if "Heteroskedastic" in dgp_type:
    hetero_level = st.sidebar.slider("Heteroskedasticity Level", 0.0, 5.0, 0.1, 0.01)
else:
    hetero_level = 0.0

# 1-Step GMM Method Selection
gmm_method = st.sidebar.selectbox(
    "1-Step GMM Method",
    ["W = S_xx⁻¹ (TSLS Equivalent)", "W = I (Identity Matrix)"],
    index=0
)

data_option = st.sidebar.radio("Data Source", ["Generate", "Paste"])

if data_option == "Generate":
    x, z, y, delta_true = generate_data(K, L, n, seed, dgp_type=dgp_type, hetero_level=hetero_level)
    st.sidebar.write(f"δ_true: {delta_true}")
else:
    pasted_data = st.sidebar.text_area("Paste CSV data (columns: x1,x2,...,z1,z2,...,y)")
    if pasted_data:
        try:
            x, z, y, K_parsed, L_parsed, n_parsed = parse_pasted_data(pasted_data)
            if K_parsed != K or L_parsed != L or n_parsed != n:
                st.sidebar.warning(f"Parsed dimensions: K={K_parsed}, L={L_parsed}, n={n_parsed}. Adjust sliders if needed.")
            K, L, n = K_parsed, L_parsed, n_parsed
            delta_true = None  # Not known for pasted data
        except ValueError as e:
            st.sidebar.error(e)
            x, z, y, delta_true = None, None, None, None
    else:
        x, z, y, delta_true = None, None, None, None

if x is not None:
    identified, rank = check_identification(x, z, L)

    # Pre-compute GMM estimates if identified using optimized functions
    if identified:
        S_xx, S_xz, S_xy = compute_sample_moments_optimized(x, z, y)

        # Compute only the selected 1-step GMM method using optimized versions
        if gmm_method == "W = S_xx⁻¹ (TSLS Equivalent)":
            delta_hat = compute_gmm_1step_optimized(S_xx, S_xz, S_xy)
            residuals_1 = compute_residuals(z, y, delta_hat)
            g_n = compute_g_n(x, residuals_1)
            S_hat = compute_S_hat_optimized(residuals_1, x)
            delta_hat_I = residuals_1_I = g_n_I = None
        else:  # W = I
            delta_hat_I = compute_gmm_1step_identity_optimized(S_xz, S_xy)
            residuals_1_I = compute_residuals(z, y, delta_hat_I)
            g_n_I = compute_g_n(x, residuals_1_I)
            # Note: For 2-step GMM, we still need S_hat from the W=S_xx^{-1} method
            # as it's the conventional approach
            delta_hat_temp = compute_gmm_1step_optimized(S_xx, S_xz, S_xy)
            residuals_temp = compute_residuals(z, y, delta_hat_temp)
            S_hat = compute_S_hat_optimized(residuals_temp, x)
            delta_hat = residuals_1 = g_n = None

        # Ensure delta_hat_I is always computed for tab3 using optimized version
        if delta_hat_I is None:
            delta_hat_I = compute_gmm_1step_identity_optimized(S_xz, S_xy)
            residuals_1_I = compute_residuals(z, y, delta_hat_I)
            g_n_I = compute_g_n(x, residuals_1_I)

        try:
            W2 = np.linalg.inv(S_hat)
            inversion_success = True
            delta_hat_2, success_2 = compute_gmm_2step(S_xx, S_xz, S_xy, W2)
            if success_2:
                residuals_2 = compute_residuals(z, y, delta_hat_2)
                g_n_2 = compute_g_n(x, residuals_2)
                J2, success_J = compute_J_stat(g_n_2, W2, n)
            else:
                residuals_2 = None
                g_n_2 = None
                J2 = None
        except np.linalg.LinAlgError:
            inversion_success = False
            W2 = None
            delta_hat_2 = None
            residuals_2 = None
            g_n_2 = None
            J2 = None
    else:
        S_xx = S_xz = S_xy = delta_hat = residuals_1 = g_n = S_hat = W2 = delta_hat_2 = residuals_2 = g_n_2 = J2 = None
        delta_hat_I = residuals_1_I = g_n_I = None
        inversion_success = False

    tab1, tab2, tab3, tab4 = st.tabs(["Data & DGP", "1-Step GMM", "2-Step GMM", "Comparison and J Test"])

    with tab1:
        # Summary statistics
        st.header("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("X (Instruments)")
            st.write(pd.DataFrame(x).describe())
        with col2:
            st.subheader("Z (Endogenous)")
            st.write(pd.DataFrame(z).describe())
        with col3:
            st.subheader("Y (Outcome)")
            st.write(pd.DataFrame(y, columns=['y']).describe())

        # Table preview
        st.header("Data Preview")
        df = pd.DataFrame(np.column_stack([x, z, y]), columns=[f'x{i+1}' for i in range(K)] + [f'z{i+1}' for i in range(L)] + ['y'])
        st.dataframe(df.head(10))

        # δ_true
        if delta_true is not None:
            st.header("True Parameters")
            st.write(f"δ_true: {delta_true}")

        # Identification status
        st.header("Identification Status")
        if identified:
            st.success(f"Model is identified (rank of Σ_xz = {rank} = L)")
        else:
            st.error(f"Model is not identified (rank of Σ_xz = {rank} < L)")

        # Data distributions
        st.header("Data Distributions")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(x.flatten(), bins=30, alpha=0.7)
        axes[0].set_title('Instruments X')
        axes[1].hist(z.flatten(), bins=30, alpha=0.7)
        axes[1].set_title('Endogenous Z')
        axes[2].hist(y, bins=30, alpha=0.7)
        axes[2].set_title('Outcome Y')
        st.pyplot(fig)

        # Data Generating Process Explanation
        st.header("Data Generating Process Explanation")
        st.markdown("""
        The data is generated according to the following process:

        - **Instruments (X)**: Each x_i is drawn from a multivariate normal distribution with mean 0 and identity covariance matrix: x_i ~ N(0, I_K)

        - **Endogenous Variables (Z)**: Z is generated as z_i = Π x_i + v_i, where Π is a randomly generated L x K matrix with full rank, and v_i ~ N(0, σ_v^2 I_L). The variance σ_v depends on the endogeneity level: 1.0 for "Low Endogeneity", 0.01 for "High Endogeneity", and 0.1 otherwise.

        - **Parameters (δ)**: If not provided, δ_true is randomly generated from a standard normal distribution.

        - **Error Term (ε)**: The error term depends on the DGP type:
          - Homoskedastic: ε_i ~ N(0, 0.1)
          - Heteroskedastic (Linear): ε_i ~ N(0, hetero_level * (1 + |x_{i1}|))
          - Heteroskedastic (Quadratic): ε_i ~ N(0, hetero_level * (1 + x_{i1}^2))
          - Heteroskedastic (Exponential): ε_i ~ N(0, hetero_level * exp(|x_{i1}|))
          - Invalid Instruments (Not Exogenous): ε_i = ε_base + α * x_{i1} + ε_pure, where ε_base ~ N(0, 0.05), α = 0.3, ε_pure ~ N(0, 0.1)

        - **Outcome (Y)**: y_i = z_i' δ_true + ε_i

        This process creates endogenous variables through the correlation between x and z, and allows for different types of heteroskedasticity and endogeneity levels.
        """)

    with tab2:
        if identified:
            st.header(f"1-Step GMM - {gmm_method}")

            # Display moments
            st.subheader("Sample Moments")
            st.latex(matrix_to_latex(S_xx, r"\mathbf{S}_{xx}"))
            st.latex(matrix_to_latex(S_xz, r"\mathbf{S}_{xz}"))
            st.latex(matrix_to_latex(S_xy, r"\mathbf{S}_{xy}"))

            # Select the appropriate method based on user choice
            if gmm_method == "W = S_xx⁻¹ (TSLS Equivalent)":
                if delta_hat is None:
                    # Compute on demand if not already computed using optimized functions
                    delta_hat = compute_gmm_1step_optimized(S_xx, S_xz, S_xy)
                    residuals_1 = compute_residuals(z, y, delta_hat)
                    g_n = compute_g_n(x, residuals_1)
                    S_hat = compute_S_hat_optimized(residuals_1, x)

                current_delta = delta_hat
                current_residuals = residuals_1
                current_g_n = g_n
                method_name = "TSLS"

                # Compute standard errors using optimized version
                try:
                    V1 = compute_asymptotic_variance_1step_optimized(S_xx, S_xz, S_hat)
                    se_1 = np.sqrt(np.diag(V1) / n)
                    st.subheader("Standard Errors (Asymptotic)")
                    for i, se in enumerate(se_1):
                        st.write(f"SE(δ_{i+1}): {se:.6f}")
                except:
                    st.error("Failed to compute standard errors")
                    se_1 = None
            else:  # W = I
                if delta_hat_I is None:
                    # Compute on demand if not already computed using optimized function
                    delta_hat_I = compute_gmm_1step_identity_optimized(S_xz, S_xy)
                    residuals_1_I = compute_residuals(z, y, delta_hat_I)
                    g_n_I = compute_g_n(x, residuals_1_I)

                current_delta = delta_hat_I
                current_residuals = residuals_1_I
                current_g_n = g_n_I
                method_name = "I"

                # Compute standard errors using optimized version
                try:
                    V1_I = compute_asymptotic_variance_1step_identity_optimized(S_xz, S_hat)
                    se_1_I = np.sqrt(np.diag(V1_I) / n)
                    st.subheader("Standard Errors (Asymptotic)")
                    for i, se in enumerate(se_1_I):
                        st.write(f"SE(δ_{i+1}): {se:.6f}")
                except:
                    st.error("Failed to compute standard errors")
                    se_1_I = None

            st.subheader(f"1-Step GMM Estimator ({gmm_method})")
            st.latex(matrix_to_latex(current_delta, rf"\hat{{\delta}}^{{1}}_{{{method_name}}}"))

            # If delta_true available, show difference
            if delta_true is not None:
                st.write(f"True δ: {delta_true}")
                st.write(f"Estimation Error: {current_delta - delta_true}")

            # Residuals
            st.subheader("Residuals")
            st.write("Summary of Residuals:")
            st.write(pd.Series(current_residuals).describe())

            # Residual distribution
            st.subheader("Residual Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(current_residuals, bins=30, alpha=0.7)
            ax.set_title(f'Residuals from 1-Step GMM ({gmm_method})')
            st.pyplot(fig)

            # g_n
            st.subheader(f"g_n(δ̂¹_{method_name})")
            st.latex(matrix_to_latex(current_g_n, rf"\mathbf{{g}}_n(\hat{{\delta}}^{{1}}_{{{method_name}}})"))

            # Compute intermediates for numerical display
            if gmm_method == "W = S_xx⁻¹ (TSLS Equivalent)":
                S_xx_inv = np.linalg.inv(S_xx)
                temp_1step = S_xz.T @ S_xx_inv @ S_xz
                temp_1step_inv = np.linalg.inv(temp_1step)
                inner_1step = S_xz.T @ S_xx_inv @ S_xy
                
                # Expanders
                with st.expander("Formulas and Derivations"):
                    st.markdown("**Moment Conditions:**")
                    st.latex(r"E[x_i (y_i - z_i' \delta)] = 0")
                    st.markdown("**Sample Moments:**")
                    st.latex(r"\mathbf{S}_{xx} = \frac{1}{n} \sum x_i x_i'")
                    st.latex(r"\mathbf{S}_{xz} = \frac{1}{n} \sum x_i z_i'")
                    st.latex(r"\mathbf{S}_{xy} = \frac{1}{n} \sum x_i y_i")
                    st.markdown("**1-Step GMM Estimator (W = S_xx⁻¹):**")
                    st.latex(r"\hat{\delta}^{1}_{TSLS} = (\mathbf{S}_{xz}' \mathbf{S}_{xx}^{-1} \mathbf{S}_{xz})^{-1} \mathbf{S}_{xz}' \mathbf{S}_{xx}^{-1} \mathbf{S}_{xy}")
                    st.markdown("**Residuals:**")
                    st.latex(r"\hat{\epsilon}_i = y_i - z_i' \hat{\delta}^{1}_{TSLS}")
                    st.markdown("**g_n(δ):**")
                    st.latex(r"\mathbf{g}_n(\delta) = \frac{1}{n} \sum x_i (y_i - z_i' \delta)")
                    st.markdown("**Asymptotic Variance:**")
                    st.latex(r"V = (\mathbf{S}_{xz}' \mathbf{S}_{xx}^{-1} \mathbf{S}_{xz})^{-1} (\mathbf{S}_{xz}' \mathbf{S}_{xx}^{-1} \hat{\mathbf{S}} \mathbf{S}_{xx}^{-1} \mathbf{S}_{xz}) (\mathbf{S}_{xz}' \mathbf{S}_{xx}^{-1} \mathbf{S}_{xz})^{-1}")

                    st.markdown("**Numerical Calculations:**")
                    st.markdown("**Sample Moments:**")
                    st.latex(matrix_to_latex(S_xx, r"\mathbf{S}_{xx}"))
                    st.latex(matrix_to_latex(S_xz, r"\mathbf{S}_{xz}"))
                    st.latex(matrix_to_latex(S_xy, r"\mathbf{S}_{xy}"))
                    st.markdown("**1-Step GMM Estimator Computation:**")
                    st.latex(rf"\mathbf{{S}}_{{xx}}^{{-1}} = {matrix_to_latex(S_xx_inv, '')}")
                    st.latex(rf"\mathbf{{S}}_{{xz}}' \mathbf{{S}}_{{xx}}^{{-1}} \mathbf{{S}}_{{xz}} = {matrix_to_latex(temp_1step, '')}")
                    st.latex(rf"(\mathbf{{S}}_{{xz}}' \mathbf{{S}}_{{xx}}^{{-1}} \mathbf{{S}}_{{xz}})^{{-1}} = {matrix_to_latex(temp_1step_inv, '')}")
                    st.latex(rf"\mathbf{{S}}_{{xz}}' \mathbf{{S}}_{{xx}}^{{-1}} \mathbf{{S}}_{{xy}} = {matrix_to_latex(inner_1step, '')}")
                    st.latex(rf"\hat{{\delta}}^{{1}}_{{TSLS}} = {matrix_to_latex(current_delta, '')}")
                    st.markdown("**g_n(δ̂¹_TSLS):**")
                    st.latex(rf"\mathbf{{g}}_n(\hat{{\delta}}^{{1}}_{{TSLS}}) = {matrix_to_latex(current_g_n, '')}")

                with st.expander("Explanations"):
                    st.markdown("The 1-step GMM estimator with W = S_xx⁻¹ is equivalent to the Two-Stage Least Squares (2SLS) estimator.")
                    st.markdown("This weighting matrix is scale-invariant and provides a natural starting point for GMM estimation.")
                    st.markdown("The residuals represent the fitted errors, and g_n(δ̂¹_TSLS) should be close to zero if the estimator is consistent.")
                    st.markdown("Standard errors are computed using the asymptotic variance formula that accounts for heteroskedasticity.")
            else:  # W = I
                temp_1step_I = S_xz.T @ S_xz
                temp_1step_I_inv = np.linalg.inv(temp_1step_I)
                inner_1step_I = S_xz.T @ S_xy
                
                # Expanders
                with st.expander("Formulas and Derivations"):
                    st.markdown("**Moment Conditions:**")
                    st.latex(r"E[x_i (y_i - z_i' \delta)] = 0")
                    st.markdown("**Sample Moments:**")
                    st.latex(r"\mathbf{S}_{xx} = \frac{1}{n} \sum x_i x_i'")
                    st.latex(r"\mathbf{S}_{xz} = \frac{1}{n} \sum x_i z_i'")
                    st.latex(r"\mathbf{S}_{xy} = \frac{1}{n} \sum x_i y_i")
                    st.markdown("**1-Step GMM Estimator (W = I):**")
                    st.latex(r"\hat{\delta}^{1}_{I} = (\mathbf{S}_{xz}' \mathbf{S}_{xz})^{-1} \mathbf{S}_{xz}' \mathbf{S}_{xy}")
                    st.markdown("**Residuals:**")
                    st.latex(r"\hat{\epsilon}_i = y_i - z_i' \hat{\delta}^{1}_{I}")
                    st.markdown("**g_n(δ):**")
                    st.latex(r"\mathbf{g}_n(\delta) = \frac{1}{n} \sum x_i (y_i - z_i' \delta)")
                    st.markdown("**Asymptotic Variance:**")
                    st.latex(r"V_I = (\mathbf{S}_{xz}' \mathbf{S}_{xz})^{-1} (\mathbf{S}_{xz}' \hat{\mathbf{S}} \mathbf{S}_{xz}) (\mathbf{S}_{xz}' \mathbf{S}_{xz})^{-1}")

                    st.markdown("**Numerical Calculations:**")
                    st.markdown("**Sample Moments:**")
                    st.latex(matrix_to_latex(S_xx, r"\mathbf{S}_{xx}"))
                    st.latex(matrix_to_latex(S_xz, r"\mathbf{S}_{xz}"))
                    st.latex(matrix_to_latex(S_xy, r"\mathbf{S}_{xy}"))
                    st.markdown("**1-Step GMM Estimator Computation:**")
                    st.latex(rf"\mathbf{{S}}_{{xz}}' \mathbf{{S}}_{{xz}} = {matrix_to_latex(temp_1step_I, '')}")
                    st.latex(rf"(\mathbf{{S}}_{{xz}}' \mathbf{{S}}_{{xz}})^{{-1}} = {matrix_to_latex(temp_1step_I_inv, '')}")
                    st.latex(rf"\mathbf{{S}}_{{xz}}' \mathbf{{S}}_{{xy}} = {matrix_to_latex(inner_1step_I, '')}")
                    st.latex(rf"\hat{{\delta}}^{{1}}_{{I}} = {matrix_to_latex(current_delta, '')}")
                    st.markdown("**g_n(δ̂¹_I):**")
                    st.latex(rf"\mathbf{{g}}_n(\hat{{\delta}}^{{1}}_{{I}}) = {matrix_to_latex(current_g_n, '')}")

                with st.expander("Explanations"):
                    st.markdown("The 1-step GMM estimator with W = I (identity matrix) is a simple alternative weighting scheme.")
                    st.markdown("This method gives equal weight to all moment conditions, unlike the TSLS equivalent which uses S_xx⁻¹.")
                    st.markdown("The residuals represent the fitted errors, and g_n(δ̂¹_I) should be close to zero if the estimator is consistent.")
                    st.markdown("Standard errors are computed using the appropriate asymptotic variance formula for the identity weighting matrix.")
        else:
            st.warning("Model is not identified. Cannot perform GMM estimation.")


    with tab3:
        if identified:
            st.header("2-Step GMM")

            if inversion_success:
                st.subheader("2-Step GMM Estimator")
                st.latex(matrix_to_latex(delta_hat_2, r"\hat{\delta}^2"))

                # If delta_true available, show difference
                if delta_true is not None:
                    st.write(f"True δ: {delta_true}")
                    st.write(f"Estimation Error: {delta_hat_2 - delta_true}")

                # Residuals for 2-step
                st.subheader("Residuals (2-Step)")
                st.write("Summary of Residuals:")
                st.write(pd.Series(residuals_2).describe())

                # Residual distribution
                st.subheader("Residual Distribution (2-Step)")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(residuals_2, bins=30, alpha=0.7)
                ax.set_title('Residuals from 2-Step GMM')
                st.pyplot(fig)

                # g_n for δ̂²
                st.subheader("g_n(δ̂²)")
                st.latex(matrix_to_latex(g_n_2, r"\mathbf{g}_n(\hat{\delta}^2)"))

                # J²
                if success_J:
                    st.subheader("J² Statistic")
                    st.write(f"J² = {J2:.4f}")
                else:
                    st.error("Failed to compute J²")
            else:
                st.error("Ŝ is singular, cannot compute 2-step GMM.")

            # Display intermediates
            st.subheader("Intermediates")
            st.latex(matrix_to_latex(S_hat, r"\hat{\mathbf{S}}"))
            if inversion_success:
                st.latex(matrix_to_latex(W2, r"\mathbf{W}_2"))

            # Compute intermediates for 2-step numerical display
            if inversion_success:
                temp_2step = S_xz.T @ W2 @ S_xz
                temp_2step_inv = np.linalg.inv(temp_2step)
                inner_2step = S_xz.T @ W2 @ S_xy
                J2_computed = n * (g_n_2.T @ W2 @ g_n_2)

            # Expanders for 2-step
            with st.expander("Formulas and Derivations (2-Step)"):
                st.markdown("**Ŝ:**")
                st.latex(r"\hat{\mathbf{S}} = \frac{1}{n} \sum (\hat{\epsilon}_i^1)^2 \mathbf{x}_i \mathbf{x}_i'")
                st.markdown("**W₂:**")
                st.latex(r"\mathbf{W}_2 = \hat{\mathbf{S}}^{-1}")
                st.markdown("**2-Step GMM Estimator:**")
                st.latex(r"\hat{\delta}^2 = (\mathbf{S}_{xz}' \mathbf{W}_2 \mathbf{S}_{xz})^{-1} \mathbf{S}_{xz}' \mathbf{W}_2 \mathbf{S}_{xy}")
                st.markdown("**J² Statistic:**")
                st.latex(r"J^2 = n \mathbf{g}_n(\hat{\delta}^2)' \mathbf{W}_2 \mathbf{g}_n(\hat{\delta}^2)")

                if inversion_success:
                    st.markdown("**Numerical Calculations:**")
                    st.latex(matrix_to_latex(S_hat, r"\hat{\mathbf{S}}"))
                    st.latex(matrix_to_latex(W2, r"\mathbf{W}_2"))
                    st.markdown("**2-Step GMM Estimator Computation:**")
                    st.latex(rf"\mathbf{{S}}_{{xz}}' \mathbf{{W}}_2 \mathbf{{S}}_{{xz}} = {matrix_to_latex(temp_2step, '')}")
                    st.latex(rf"(\mathbf{{S}}_{{xz}}' \mathbf{{W}}_2 \mathbf{{S}}_{{xz}})^{{-1}} = {matrix_to_latex(temp_2step_inv, '')}")
                    st.latex(rf"\mathbf{{S}}_{{xz}}' \mathbf{{W}}_2 \mathbf{{S}}_{{xy}} = {matrix_to_latex(inner_2step, '')}")
                    st.latex(rf"\hat{{\delta}}^2 = {matrix_to_latex(delta_hat_2, '')}")
                    st.markdown("**g_n(δ̂²):**")
                    st.latex(rf"\mathbf{{g}}_n(\hat{{\delta}}^2) = {matrix_to_latex(g_n_2, '')}")
                    st.markdown("**J² Statistic Computation:**")
                    st.latex(rf"J^2 = {n} \times ({matrix_to_latex(g_n_2.T @ W2 @ g_n_2, '')}) = {J2:.4f}")

            with st.expander("Explanations (2-Step)"):
                st.markdown("The 2-step GMM uses an optimal weighting matrix W₂ based on the residuals from the 1-step estimator.")
                st.markdown("This improves efficiency compared to the 1-step GMM.")
                st.markdown("The J² statistic tests overidentification: under the null, J² ~ χ² with degrees of freedom K - L.")
        else:
            st.warning("Model is not identified. Cannot perform GMM estimation.")

    with tab4:
        if data_option == "Generate" and delta_true is not None and identified:
            st.header("Comparison and J Test")

            delta_hats_1, delta_hats_1_I, delta_hats_2, J2s, p_values = run_monte_carlo_simulations(K, L, n, M, seed, dgp_type, hetero_level, gmm_method, delta_true)

            # Compute averages, biases, SEs
            # W = S_xx^-1 method
            avg_delta_1 = np.mean(delta_hats_1, axis=0)
            bias_1 = avg_delta_1 - delta_true
            se_1 = np.std(delta_hats_1, axis=0, ddof=1)
            
            # W = I method
            avg_delta_1_I = np.mean(delta_hats_1_I, axis=0)
            bias_1_I = avg_delta_1_I - delta_true
            se_1_I = np.std(delta_hats_1_I, axis=0, ddof=1)

            avg_delta_2 = np.nanmean(delta_hats_2, axis=0)
            bias_2 = avg_delta_2 - delta_true
            se_2 = np.nanstd(delta_hats_2, axis=0, ddof=1)

            # Asymptotic variances (from one simulation, say the last) using optimized functions
            S_xx_asym, S_xz_asym, S_xy_asym = compute_sample_moments_optimized(x, z, y)
            delta_hat_1_asym = compute_gmm_1step_optimized(S_xx_asym, S_xz_asym, S_xy_asym)
            delta_hat_1_I_asym = compute_gmm_1step_identity_optimized(S_xz_asym, S_xy_asym)
            residuals_1_asym = compute_residuals(z, y, delta_hat_1_asym)
            S_hat_asym = compute_S_hat_optimized(residuals_1_asym, x)
            residuals_1_I_asym = compute_residuals(z, y, delta_hat_1_I_asym)
            S_hat_I_asym = compute_S_hat_optimized(residuals_1_I_asym, x)

            # Asymptotic standard errors for both 1-step methods using optimized versions
            V1 = compute_asymptotic_variance_1step_optimized(S_xx_asym, S_xz_asym, S_hat_asym)
            asym_se_1 = np.sqrt(np.diag(V1) / n)

            V1_I = compute_asymptotic_variance_1step_identity_optimized(S_xz_asym, S_hat_I_asym)
            asym_se_1_I = np.sqrt(np.diag(V1_I) / n)

            if inversion_success:
                V2 = compute_asymptotic_variance_2step(S_xz_asym, W2, S_hat_asym)
                asym_se_2 = np.sqrt(np.diag(V2) / n)
            else:
                asym_se_2 = np.full(L, np.nan)

            # 1-Step GMM Results Tables - Display only selected method
            if gmm_method == "W = S_xx⁻¹ (TSLS Equivalent)":
                st.subheader("1-Step GMM Results (W = S_xx⁻¹)")
                one_step_data = []
                for j in range(L):
                    one_step_data.append({
                        "Parameter": f"δ_{j+1}",
                        "Estimated": avg_delta_1[j],
                        "True Value": delta_true[j],
                        "Bias": bias_1[j],
                        "SE (Emp)": se_1[j],
                        "SE (Asym)": asym_se_1[j]
                    })
                df_one_step = pd.DataFrame(one_step_data)
                st.dataframe(df_one_step)
            else:  # W = I
                st.subheader("1-Step GMM Results (W = I)")
                one_step_I_data = []
                for j in range(L):
                    one_step_I_data.append({
                        "Parameter": f"δ_{j+1}",
                        "Estimated": avg_delta_1_I[j],
                        "True Value": delta_true[j],
                        "Bias": bias_1_I[j],
                        "SE (Emp)": se_1_I[j],
                        "SE (Asym)": asym_se_1_I[j]
                    })
                df_one_step_I = pd.DataFrame(one_step_I_data)
                st.dataframe(df_one_step_I)

            # 2-Step GMM Results Table
            st.subheader("2-Step GMM Results")
            two_step_data = []
            for j in range(L):
                two_step_data.append({
                    "Parameter": f"δ_{j+1}",
                    "Estimated": avg_delta_2[j],
                    "True Value": delta_true[j],
                    "Bias": bias_2[j],
                    "SE (Emp)": se_2[j],
                    "SE (Asym)": asym_se_2[j]
                })
            df_two_step = pd.DataFrame(two_step_data)
            st.dataframe(df_two_step)

            # Direct Comparison Table
            st.subheader("Direct Comparison: Selected 1-Step vs 2-Step")
            comparison_data = []
            for j in range(L):
                # Compare 2-step vs selected 1-step
                if gmm_method == "W = S_xx⁻¹ (TSLS Equivalent)":
                    bias_diff = bias_2[j] - bias_1[j]
                    eff_gain_emp = ((se_1[j] / se_2[j]) - 1) * 100 if se_2[j] != 0 else np.nan
                    eff_gain_asym = ((asym_se_1[j] / asym_se_2[j]) - 1) * 100 if asym_se_2[j] != 0 else np.nan
                    method_label = "W=S_xx⁻¹"
                else:
                    bias_diff = bias_2[j] - bias_1_I[j]
                    eff_gain_emp = ((se_1_I[j] / se_2[j]) - 1) * 100 if se_2[j] != 0 else np.nan
                    eff_gain_asym = ((asym_se_1_I[j] / asym_se_2[j]) - 1) * 100 if asym_se_2[j] != 0 else np.nan
                    method_label = "W=I"

                comparison_data.append({
                    "Parameter": f"δ_{j+1}",
                    f"Bias: 2-Step vs {method_label}": bias_diff,
                    f"Eff Gain: 2-Step vs {method_label} (Emp %)": eff_gain_emp,
                    f"Eff Gain: 2-Step vs {method_label} (Asym %)": eff_gain_asym
                })
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison)

            # Hansen J-Test
            st.subheader("Hansen J-Test")
            if K > L:
                avg_J2 = np.nanmean(J2s)
                avg_p_value = np.nanmean(p_values)
                df_j = K - L
                critical_5 = stats.chi2.ppf(0.95, df_j)
                critical_1 = stats.chi2.ppf(0.99, df_j)
                st.write(f"Average J²: {avg_J2:.4f}")
                st.write(f"Average p-value: {avg_p_value:.4f}")
                st.write(f"Degrees of freedom: {df_j}")
                st.write(f"5% Critical value: {critical_5:.4f}")
                st.write(f"1% Critical value: {critical_1:.4f}")
                if avg_p_value > 0.05:
                    st.success("Fail to reject null: moments are satisfied.")
                else:
                    st.error("Reject null: moments are not satisfied.")
            else:
                st.write("Model is exactly identified (K = L), J-test not applicable.")

            # Explanations
            with st.expander("Explanations"):
                st.markdown("**Exactly Identified Case (K = L):**")
                st.markdown("The model has exactly as many moment conditions as parameters. The GMM estimator is unique and efficient. The J-test is not applicable since there are no degrees of freedom for overidentification.")
                st.markdown("**Overidentified Case (K > L):**")
                st.markdown("There are more moment conditions than parameters. The 2-step GMM is more efficient than 1-step. The J-test checks if the overidentifying restrictions hold: J² ~ χ²_{K-L} under the null.")

            # Plots
            st.subheader("Plots")
            fig, axes = plt.subplots(1, L, figsize=(6*L, 4))
            if L == 1:
                axes = [axes]
            for j in range(L):
                if gmm_method == "W = S_xx⁻¹ (TSLS Equivalent)":
                    axes[j].hist(delta_hats_1[:, j], alpha=0.4, label='1-Step (W=S_xx⁻¹)', bins=30, color='blue')
                else:
                    axes[j].hist(delta_hats_1_I[:, j], alpha=0.4, label='1-Step (W=I)', bins=30, color='green')
                axes[j].hist(delta_hats_2[:, j], alpha=0.4, label='2-Step', bins=30, color='orange')
                axes[j].axvline(delta_true[j], color='red', linestyle='--', linewidth=2, label='True')
                axes[j].set_title(f'δ_{j+1}')
                axes[j].legend()
            st.pyplot(fig)
        else:
            st.warning("Comparison requires generated data with true parameters and identified model.")
