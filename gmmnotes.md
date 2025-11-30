Generalized Method of Moments

Oriol González-Casasús
Universitat Pompeu Fabra and Barcelona School of Economics
Advanced Econometric Methods I

Introduction to GMM

The most important assumption made for the OLS is the orthogonality between the error term and the regressors.

Without it, the OLS is not even consistent.

Note: The condition is $\mathbb{E}[x_i \epsilon_i] = 0$.

The Generalized Method of Moments (GMM) is a way to exploit orthogonality conditions to form objective functions for estimation.

This clearly includes OLS, but many other estimators.

By its nature, it can be seen as a less parametric approach than ML.

We only exploit certain moments, not the entire distribution.

Offers some robustness.

Motivating Example: Reverse Causality

We will consider the linear regression model $y_{i}=z_{i}^{\prime}\delta+\epsilon_{i}$ under threats to the validity of $\mathbb{E}[z_{i}\epsilon_{i}]=0$.

Suppose that $z_{i}=\gamma y_{i}+u_{i}$.

E.g., inflation-unemployment or crime-police.

Writing up the system gives

$$\begin{bmatrix}y_{i}\\ z_{i}\end{bmatrix}=\begin{bmatrix}\frac{1}{1-\delta\gamma}&\frac{\delta}{1-\delta\gamma}\\ \frac{\gamma}{1-\delta\gamma}&\frac{1}{1-\delta\gamma}\end{bmatrix}\begin{bmatrix}\epsilon_{i}\\ u_{i}\end{bmatrix}$$

It then follows that $\mathbb{E}[z_{i}\epsilon_{i}]=\frac{\gamma}{1-\delta\gamma}\mathbb{E}[\epsilon_{i}^{2}]+\frac{1}{1-\delta\gamma}\mathbb{E}[u_{i}\epsilon_{i}]$.

Typically, the second term will be zero ($\mathbb{E}[u_{i}\epsilon_{i}]=0$), but the first term is not negligible.

As a consequence, the OLS estimator is inconsistent:

$$\hat{\delta}_{OLS}\xrightarrow{p}\delta+\frac{\mathbb{E}[z_{i}\epsilon_{i}]}{\mathbb{E}[z_{i}^{2}]}\ne\delta$$

Motivating Example: Omitted Variables

Consider the model $y_{i}=z_{i}^{\prime}\delta+q_{i}^{\prime}\gamma+\eta_{i}$ but for some reason we can't observe $q_i$.

We instead estimate the model $y_{i}=z_{i}^{\prime}\delta+\epsilon_{i}$.

The error term is now $\epsilon_{i}=q_{i}^{\prime}\gamma+\eta_{i}$.

Derivation:

$$\begin{aligned}
  \mathbb{E}[z_{i}\epsilon_{i}] &= \mathbb{E}[z_i(q_i^{\prime}\gamma + \eta_i)] \\
  &= \mathbb{E}[z_{i}q_{i}^{\prime}]\gamma + \mathbb{E}[z_i \eta_i]
  \end{aligned}$$

Even if strict exogeneity holds for the original error ($\mathbb{E}[z_i \eta_i] = 0$), the term $\mathbb{E}[z_{i}q_{i}^{\prime}]\gamma$ is generally non-zero unless the included ($z_i$) and excluded ($q_i$) regressors are uncorrelated.

Thus, OLS becomes inconsistent.

Motivating Example: Measurement Error

Consider the model $y_{i}=\delta z_{i}^{*}+\eta_{i}$.

Suppose that $Z_{i}^{*}$ is not observed, but we instead observe a noisy version $Z_{i}=Z_{i}^{*}+u_{i}$.

Suppose we ignore the measurement error and estimate $y_{i}=\delta z_{i}+\epsilon_{i}$.

Now $\epsilon_{i}=\eta_{i}-\delta u_{i}$.

Derivation:

$$\begin{aligned}
  \mathbb{E}[z_{i}\epsilon_{i}] &= \mathbb{E}[(z_{i}^{*}+u_{i})(\eta_{i}-\delta u_{i})] \\
  &= \mathbb{E}[z_i^* \eta_i] - \delta \mathbb{E}[z_i^* u_i] + \mathbb{E}[u_i \eta_i] - \delta \mathbb{E}[u_i^2]
  \end{aligned}$$

Assuming measurement noise $u_i$ is uncorrelated with $z_i^*$ and $\eta_i$, this simplifies to $-\delta \mathbb{E}[u_i^2] = -\delta \sigma_u^2 \neq 0$.

The OLS estimator of $y_{i}$ on $Z_{i}$ is inconsistent.

Model and Assumptions

The model we study is a generalization of the previously considered.

Linearity: $y_{i}=z_{i}^{\prime}\delta+\epsilon_{i}$, where $Z_{i}$ is $L$-dimensional.

Ergodic stationarity: $\{w_{i}:=(y_{i},z_{i},x_{i})\}$ is jointly stationary and ergodic, where $X_{i}$ is a $K$-dimensional vector of instruments.

Orthogonality conditions: $\mathbb{E}[x_{ik}\epsilon_{i}]=0$ for all $i, k$.

Implies $\mathbb{E}[x_i(y_i - z_i^{\prime}\delta)] = 0$.

Rank condition: The $K\times L$ matrix $\Sigma_{xz}:=\mathbb{E}[x_{i}z_{i}^{\prime}]$ is of full column rank.

Necessary for unique solution: $K \ge L$.

MDS: Let $g_{i}:=x_{i}\epsilon_{i}$. $\{g_{i}\}$ is an MDS with $S:=Avar(\overline{g})=\mathbb{E}[g_{i}g_{i}^{\prime}]$ nonsingular.

Fourth moments: $\mathbb{E}[x_{ik}^{2}z_{il}^{2}]$ exists and is finite for all $k,l$.

Method of Moments Principle

Let $g(w_{i};\delta)=g_{i}=x_{i}(y_{i}-z_{i}^{\prime}\delta)$. By the orthogonality conditions, $\mathbb{E}[g(w_{i};\delta)]=0$.

The basic principle is to choose $\delta$ such that the corresponding sample moments are also zero:

$$g_{n}(\delta):=\frac{1}{n}\sum_{i=1}^{n}g(w_{i};\delta)=0$$

Derivation:

$$\begin{aligned}
  \frac{1}{n}\sum_{i=1}^{n}x_{i}(y_i - z_i^{\prime}\delta) &= 0 \\
  \frac{1}{n}\sum_{i=1}^{n}x_{i}y_{i} - \left(\frac{1}{n}\sum_{i=1}^{n}x_{i}z_{i}^{\prime}\right)\delta &= 0 \\
  S_{xy} - S_{xz}\delta &= 0
  \end{aligned}$$

Thus, $S_{xz}\delta=s_{xy}$ is a system of $K$ linear equations in $L$ unknowns.

Method of Moments (Exact Identification)

If the system is exactly identified ($K=L$), $S_{xz}$ is square and invertible (for large $n$).

The unique solution is the Instrumental Variables (IV) estimator:

$$\hat{\delta}_{iv}=\mathcal{S}_{xz}^{-1}S_{xy}=\left(\frac{1}{n}\sum_{i=1}^{n}x_{i}z_{i}^{\prime}\right)^{-1}\frac{1}{n}\sum_{i=1}^{n}x_{i}y_{i}$$

Special Case (OLS): If $x_i = z_i$ (regressors are exogenous), then:

$$\hat{\delta} = (Z^{\prime}Z)^{-1}Z^{\prime}y = \hat{\delta}_{OLS}$$

Generalized Method of Moments

If $K>L$ (overidentified), we generally cannot satisfy $S_{xy} - S_{xz}\delta = 0$ exactly.

We aim to make $g_n(\delta)$ as close to zero as possible using a weighted quadratic form.

Norm: $||v||_W^2 = v^{\prime}Wv$.

The GMM estimator minimizes:

$$\hat{\delta}(\hat{W}):=\arg\min_{\delta} J(\delta,\hat{W}); \quad J(\delta,\hat{W}) = n g_{n}(\delta)^{\prime}\hat{W}g_{n}(\delta)$$

where $\hat{W} \xrightarrow{p} W$ (positive definite).

GMM Solution

Objective function:

$$J(\delta,\hat{W})=n(s_{xy}-S_{xz}\delta)^{\prime}\hat{W}(s_{xy}-S_{xz}\delta)$$

First Order Condition (FOC):

$$\frac{\partial J}{\partial \delta} = -2 S_{xz}^{\prime}\hat{W}(s_{xy}-S_{xz}\delta) = 0$$

Solving for $\delta$:

$$S_{xz}^{\prime}\hat{W}S_{xy}=S_{xz}^{\prime}\hat{W}S_{xz}\delta$$

$$\hat{\delta}(\hat{W})=(S_{xz}^{\prime}\hat{W}S_{xz})^{-1}S_{xz}^{\prime}\hat{W}S_{xy}$$

Note: If $K=L$, $S_{xz}$ is invertible, so $(S_{xz}^{\prime}\hat{W}S_{xz})^{-1} = S_{xz}^{-1}\hat{W}^{-1}(S_{xz}^{\prime})^{-1}$. The expression simplifies to $S_{xz}^{-1}S_{xy}$, which is the IV estimator.

Asymptotic Distribution

Proposition 1:
$\sqrt{n}(\hat{\delta}(\hat{W})-\delta)\xrightarrow{d} N(0,Avar(\hat{\delta}(\hat{W})))$ where

$$Avar(\hat{\delta}(\hat{W}))=(\Sigma_{xz}^{\prime}W\Sigma_{xz})^{-1}\Sigma_{xz}^{\prime}WSW\Sigma_{xz}(\Sigma_{xz}^{\prime}W\Sigma_{xz})^{-1}$$

Sketch of Derivation:

Start with $\hat{\delta}(\hat{W})=(S_{xz}^{\prime}\hat{W}S_{xz})^{-1}S_{xz}^{\prime}\hat{W}S_{xy}$.

Substitute $S_{xy} = S_{xz}\delta + \frac{1}{n}\sum x_i \epsilon_i$ (since $y = z\delta + \epsilon$):

$$\hat{\delta}(\hat{W}) = \delta + (S_{xz}^{\prime}\hat{W}S_{xz})^{-1}S_{xz}^{\prime}\hat{W}\left(\frac{1}{n}\sum x_i \epsilon_i\right)$$

Rearrange and multiply by $\sqrt{n}$:

$$\sqrt{n}(\hat{\delta}(\hat{W}) - \delta) = (S_{xz}^{\prime}\hat{W}S_{xz})^{-1}S_{xz}^{\prime}\hat{W}\left(\frac{1}{\sqrt{n}}\sum x_i \epsilon_i\right)$$

Limits:

$S_{xz} \xrightarrow{p} \Sigma_{xz}$

$\hat{W} \xrightarrow{p} W$

$\frac{1}{\sqrt{n}}\sum x_i \epsilon_i \xrightarrow{d} N(0, S)$ (by CLT)

Result follows by Slutsky's theorem.

Summary of Variance Components

$\Sigma_{xz} = \mathbb{E}[x_i z_i^{\prime}] \approx S_{xz}$

$W \approx \hat{W}$

$S = \mathbb{E}[\epsilon_i^2 x_i x_i^{\prime}]$ (Long-run variance of moments)

Heteroskedasticity robust: $S \approx \frac{1}{n}\sum \hat{\epsilon}_i^2 x_i x_i^{\prime}$

Homoskedasticity: $S = \sigma^2 \Sigma_{xx} \approx \hat{\sigma}^2 S_{xx}$

Efficient GMM (Proposition 3)

We want to choose $W$ to minimize the variance. The lower bound is $(\Sigma_{xz}^{\prime}S^{-1}\Sigma_{xz})^{-1}$, achieved when $W = S^{-1}$.

Proof Details:
We want to show:

$$(\Sigma_{xz}^{\prime}W\Sigma_{xz})^{-1}\Sigma_{xz}^{\prime}WSW\Sigma_{xz}(\Sigma_{xz}^{\prime}W\Sigma_{xz})^{-1} \ge (\Sigma_{xz}^{\prime}S^{-1}\Sigma_{xz})^{-1}$$

Inverting both sides (reversing inequality):

$$\Sigma_{xz}^{\prime}W\Sigma_{xz}(\Sigma_{xz}^{\prime}WSW\Sigma_{xz})^{-1}\Sigma_{xz}^{\prime}W\Sigma_{xz} \le \Sigma_{xz}^{\prime}S^{-1}\Sigma_{xz}$$

Let $S^{-1} = C^{\prime}C$ (Cholesky decomposition).
Define:

$A = C\Sigma_{xz}$

$D = C^{-1^{\prime}}W\Sigma_{xz}$

Note that $\Sigma_{xz}^{\prime}W\Sigma_{xz} = A^{\prime}D$ and $\Sigma_{xz}^{\prime}WSW\Sigma_{xz} = D^{\prime}D$.

The inequality becomes:

$$A^{\prime}D(D^{\prime}D)^{-1}D^{\prime}A \le A^{\prime}A$$

$$A^{\prime}[I - D(D^{\prime}D)^{-1}D^{\prime}]A \ge 0$$

The term in brackets is $M_D$, the orthogonal projection matrix onto the null space of $D$. Since projection matrices are positive semi-definite, $A^{\prime}M_D A \ge 0$.

Testing Overidentifying Restrictions (Hansen's J Test)

Proposition 4: $J(\hat{\delta}(\hat{S}^{-1}),\hat{S}^{-1})\xrightarrow{d}\chi_{K-L}^{2}$.

Derivation steps:

$J = n g_n(\hat{\delta})^{\prime} \hat{S}^{-1} g_n(\hat{\delta})$.

Taylor expansion implies $g_n(\hat{\delta}) \approx B \bar{g}$, where $B = I - \Sigma_{xz}(\Sigma_{xz}^{\prime}S^{-1}\Sigma_{xz})^{-1}\Sigma_{xz}^{\prime}S^{-1}$.

Let $\hat{S}^{-1} = C^{\prime}C$. Then $CB = M_A C$ where $A = C\Sigma_{xz}$.

Substitute back:

$$J = n \bar{g}^{\prime} B^{\prime} C^{\prime} C B \bar{g} = n (C\bar{g})^{\prime} M_A (C\bar{g})$$

Since $\sqrt{n}C\bar{g} \xrightarrow{d} N(0, I_K)$ (because $C S C^{\prime} = I$), $J$ is a quadratic form of standard normals projected by $M_A$.

The distribution is $\chi^2$ with degrees of freedom equal to $\text{rank}(M_A)$.

Rank Calculation:

$$\begin{aligned}
\text{rank}(M_A) &= \text{trace}(M_A) = \text{trace}(I_K - A(A^{\prime}A)^{-1}A^{\prime}) \\
&= \text{trace}(I_K) - \text{trace}(A(A^{\prime}A)^{-1}A^{\prime}) \\
&= K - \text{trace}((A^{\prime}A)^{-1}A^{\prime}A) \\
&= K - \text{trace}(I_L) = K - L
\end{aligned}$$

C Test and LR Principle

C Test: $C = J - J_1 \xrightarrow{d} \chi_{K-K_1}^2$. Tests orthogonality of a subset of instruments.

LR Test: $LR = J_{restricted} - J_{unrestricted}$.

Behaves like Wald test asymptotically.

Invariance: LR numerical value doesn't depend on how restrictions are formulated (unlike Wald).

Requirement: Must use the same weighting matrix $\hat{S}^{-1}$ for both restricted and unrestricted models to ensure $LR \ge 0$ and valid distribution.

Homoskedasticity (2SLS)

If errors are homoskedastic, $S = \sigma^2 \Sigma_{xx}$.

The efficient GMM estimator becomes 2SLS:

$$\hat{\delta}_{2SLS} = (Z^{\prime}P_X Z)^{-1} Z^{\prime}P_X y$$

where $P_X = X(X^{\prime}X)^{-1}X^{\prime}$.

Interpretation:

Regress $Z$ on $X$ to get $\hat{Z} = P_X Z$.

Regress $y$ on $\hat{Z}$.

Sargan Statistic: The J-statistic simplifies to the Sargan statistic under homoskedasticity.

$$\text{Sargan} = \frac{(y - Z\delta)^{\prime} P_X (y - Z\delta)}{\hat{\sigma}^2}$$