import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from utils.gmm_calculations import *

def generate_data(K, L, n, seed, delta_true=None):
    np.random.seed(seed)

    # Generate x_i: multivariate normal, mean 0, covariance I_K
    x = np.random.multivariate_normal(np.zeros(K), np.eye(K), n)

    # Generate Π: L x K matrix, random but full rank
    # To ensure full rank, make it identity if K >= L, else something
    if K >= L:
        Pi = np.random.randn(L, K)
        # Ensure full rank by making it have rank L
        U, s, Vt = np.linalg.svd(Pi)
        s[s < 1e-10] = 1e-10  # avoid zero singular values
        Pi = U @ np.diag(s) @ Vt
    else:
        # If K < L, can't have full rank, but for simulation, perhaps set to have rank K
        Pi = np.random.randn(L, K)

    # Noise for z: v_i ~ N(0, 0.1 I_L)
    v = np.random.multivariate_normal(np.zeros(L), 0.1 * np.eye(L), n)

    # z_i = Pi @ x_i + v_i
    z = x @ Pi.T + v

    # If delta_true not provided, generate random
    if delta_true is None:
        delta_true = np.random.randn(L)

    # ε_i ~ N(0, 0.1)
    eps = np.random.normal(0, 0.1, n)

    # y_i = z_i @ delta_true + eps
    y = z @ delta_true + eps

    return x, z, y, delta_true

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

data_option = st.sidebar.radio("Data Source", ["Generate", "Paste"])

if data_option == "Generate":
    x, z, y, delta_true = generate_data(K, L, n, seed)
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

    # Pre-compute GMM estimates if identified
    if identified:
        S_xx, S_xz, S_xy = compute_sample_moments(x, z, y)
        delta_hat = compute_gmm_1step(S_xx, S_xz, S_xy)
        residuals_1 = compute_residuals(z, y, delta_hat)
        g_n = compute_g_n(x, residuals_1)
        S_hat = compute_S_hat(residuals_1, x)
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
        inversion_success = False

    tab1, tab2, tab3, tab4 = st.tabs(["Data & DGP", "1-Step GMM", "2-Step GMM", "Comparison & Hansen J-Test"])

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

    with tab2:
        if identified:
            st.header("1-Step GMM Estimation")

            # Display moments
            st.subheader("Sample Moments")
            st.latex(matrix_to_latex(S_xx, r"\mathbf{S}_{xx}"))
            st.latex(matrix_to_latex(S_xz, r"\mathbf{S}_{xz}"))
            st.latex(matrix_to_latex(S_xy, r"\mathbf{S}_{xy}"))

            st.subheader("1-Step GMM Estimator")
            st.latex(matrix_to_latex(delta_hat, r"\hat{\delta}^1"))

            # If delta_true available, show difference
            if delta_true is not None:
                st.write(f"True δ: {delta_true}")
                st.write(f"Estimation Error: {delta_hat - delta_true}")

            # Residuals
            st.subheader("Residuals")
            st.write("Summary of Residuals:")
            st.write(pd.Series(residuals_1).describe())

            # Residual distribution
            st.subheader("Residual Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(residuals_1, bins=30, alpha=0.7)
            ax.set_title('Residuals from 1-Step GMM')
            st.pyplot(fig)

            # g_n
            st.subheader("g_n(δ̂¹)")
            st.latex(matrix_to_latex(g_n, r"\mathbf{g}_n(\hat{\delta}^1)"))

            # Expanders
            with st.expander("Formulas and Derivations"):
                st.markdown("**Moment Conditions:**")
                st.latex(r"E[x_i (y_i - z_i' \delta)] = 0")
                st.markdown("**Sample Moments:**")
                st.latex(r"\mathbf{S}_{xx} = \frac{1}{n} \sum x_i x_i'")
                st.latex(r"\mathbf{S}_{xz} = \frac{1}{n} \sum x_i z_i'")
                st.latex(r"\mathbf{S}_{xy} = \frac{1}{n} \sum x_i y_i")
                st.markdown("**1-Step GMM Estimator:**")
                st.latex(r"\hat{\delta}^1 = (\mathbf{S}_{xz}' \mathbf{S}_{xx}^{-1} \mathbf{S}_{xz})^{-1} \mathbf{S}_{xz}' \mathbf{S}_{xx}^{-1} \mathbf{S}_{xy}")
                st.markdown("**Residuals:**")
                st.latex(r"\hat{\epsilon}_i = y_i - z_i' \hat{\delta}^1")
                st.markdown("**g_n(δ):**")
                st.latex(r"\mathbf{g}_n(\delta) = \frac{1}{n} \sum x_i (y_i - z_i' \delta)")

            with st.expander("Explanations"):
                st.markdown("The 1-step GMM estimator uses the identity matrix as the weighting matrix (W = I).")
                st.markdown("It solves the sample moment conditions by minimizing the quadratic form.")
                st.markdown("The residuals represent the fitted errors, and g_n(δ̂¹) should be close to zero if the estimator is consistent.")
        else:
            st.warning("Model is not identified. Cannot perform GMM estimation.")

    with tab3:
        if identified:
            st.header("2-Step GMM Estimation")

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

            with st.expander("Explanations (2-Step)"):
                st.markdown("The 2-step GMM uses an optimal weighting matrix W₂ based on the residuals from the 1-step estimator.")
                st.markdown("This improves efficiency compared to the 1-step GMM.")
                st.markdown("The J² statistic tests overidentification: under the null, J² ~ χ² with degrees of freedom K - L.")
        else:
            st.warning("Model is not identified. Cannot perform GMM estimation.")

    with tab4:
        if data_option == "Generate" and delta_true is not None and identified:
            st.header("Comparison & Hansen J-Test")

            # Run M simulations
            delta_hats_1 = []
            delta_hats_2 = []
            J2s = []
            p_values = []

            for m in range(M):
                # Generate new data with same parameters but different seed
                x_sim, z_sim, y_sim, _ = generate_data(K, L, n, seed + m + 1, delta_true)

                # Compute moments
                S_xx_sim, S_xz_sim, S_xy_sim = compute_sample_moments(x_sim, z_sim, y_sim)

                # 1-step
                delta_hat_1_sim = compute_gmm_1step(S_xx_sim, S_xz_sim, S_xy_sim)
                delta_hats_1.append(delta_hat_1_sim)

                # Residuals 1
                residuals_1_sim = compute_residuals(z_sim, y_sim, delta_hat_1_sim)

                # S_hat
                S_hat_sim = compute_S_hat(residuals_1_sim, x_sim)

                # W2
                try:
                    W2_sim = np.linalg.inv(S_hat_sim)
                    inversion_success_sim = True
                except np.linalg.LinAlgError:
                    inversion_success_sim = False

                if inversion_success_sim:
                    # 2-step
                    delta_hat_2_sim, success_2_sim = compute_gmm_2step(S_xx_sim, S_xz_sim, S_xy_sim, W2_sim)
                    if success_2_sim:
                        delta_hats_2.append(delta_hat_2_sim)

                        # Residuals 2
                        residuals_2_sim = compute_residuals(z_sim, y_sim, delta_hat_2_sim)

                        # g_n
                        g_n_sim = compute_g_n(x_sim, residuals_2_sim)

                        # J2
                        J2_sim, success_J_sim = compute_J_stat(g_n_sim, W2_sim, n)
                        if success_J_sim:
                            J2s.append(J2_sim)
                            df = K - L
                            p_val = compute_j_test_p_value(J2_sim, df)
                            p_values.append(p_val)
                        else:
                            J2s.append(np.nan)
                            p_values.append(np.nan)
                    else:
                        delta_hats_2.append(np.full(L, np.nan))
                        J2s.append(np.nan)
                        p_values.append(np.nan)
                else:
                    delta_hats_2.append(np.full(L, np.nan))
                    J2s.append(np.nan)
                    p_values.append(np.nan)

            # Convert to arrays
            delta_hats_1 = np.array(delta_hats_1)
            delta_hats_2 = np.array(delta_hats_2)
            J2s = np.array(J2s)
            p_values = np.array(p_values)

            # Compute averages, biases, SEs
            avg_delta_1 = np.mean(delta_hats_1, axis=0)
            bias_1 = avg_delta_1 - delta_true
            se_1 = np.std(delta_hats_1, axis=0, ddof=1)

            avg_delta_2 = np.nanmean(delta_hats_2, axis=0)
            bias_2 = avg_delta_2 - delta_true
            se_2 = np.nanstd(delta_hats_2, axis=0, ddof=1)

            # Asymptotic variances (from one simulation, say the last)
            S_xx_asym, S_xz_asym, S_xy_asym = compute_sample_moments(x, z, y)
            delta_hat_1_asym = compute_gmm_1step(S_xx_asym, S_xz_asym, S_xy_asym)
            residuals_1_asym = compute_residuals(z, y, delta_hat_1_asym)
            S_hat_asym = compute_S_hat(residuals_1_asym, x)
            V1 = compute_asymptotic_variance_1step(S_xx_asym, S_xz_asym, S_hat_asym)
            asym_se_1 = np.sqrt(np.diag(V1) / n)

            if inversion_success:
                V2 = compute_asymptotic_variance_2step(S_xz_asym, W2)
                asym_se_2 = np.sqrt(np.diag(V2) / n)
            else:
                asym_se_2 = np.full(L, np.nan)

            # Comparison table
            st.subheader("Comparison Table")
            comparison_data = []
            for j in range(L):
                comparison_data.append({
                    "Parameter": f"δ_{j+1}",
                    "True Value": delta_true[j],
                    "1-Step Avg": avg_delta_1[j],
                    "1-Step Bias": bias_1[j],
                    "1-Step SE (Emp)": se_1[j],
                    "1-Step SE (Asym)": asym_se_1[j],
                    "2-Step Avg": avg_delta_2[j],
                    "2-Step Bias": bias_2[j],
                    "2-Step SE (Emp)": se_2[j],
                    "2-Step SE (Asym)": asym_se_2[j]
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
            fig, axes = plt.subplots(1, L, figsize=(5*L, 4))
            if L == 1:
                axes = [axes]
            for j in range(L):
                axes[j].hist(delta_hats_1[:, j], alpha=0.5, label='1-Step', bins=30)
                axes[j].hist(delta_hats_2[:, j], alpha=0.5, label='2-Step', bins=30)
                axes[j].axvline(delta_true[j], color='red', linestyle='--', label='True')
                axes[j].set_title(f'δ_{j+1}')
                axes[j].legend()
            st.pyplot(fig)
        else:
            st.warning("Comparison requires generated data with true parameters and identified model.")