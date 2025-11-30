Generalized Method of Moments

Oriol González-Casasús
Universitat Pompeu Fabra and Barcelona School of Economics
Advanced Econometric Methods I

Generalized Method of Moments

The most important assumption made for the OLS is the orthogonality between the error term and the regressors.

Without it, the OLS is not even consistent.

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

$$\begin{bmatrix}y_{i}\\ z_{i}\end{bmatrix}=\begin{bmatrix}\frac{1}{1-\delta\gamma}&\frac{\delta}{1-\delta\gamma}\\ \frac{\gamma}{1-\delta\gamma}&\frac{1}{1-\delta\gamma}\end{bmatrix}\begin{bmatrix}\epsilon_{i}\\ U_{i}\end{bmatrix}$$

It then follows that $\mathbb{E}[z_{i}\epsilon_{i}]=\frac{\gamma}{1-\delta\gamma}\mathbb{E}[\epsilon_{i}^{2}]+\frac{1}{1-\delta\gamma}\mathbb{E}[u_{i}\epsilon_{i}]$.

Typically, the second term will be zero, but the first term is not negligible.

As a consequence, the OLS estimator is inconsistent:

$$\hat{\delta}\xrightarrow{p}\delta+\frac{\mathbb{E}[z_{i}\epsilon_{i}]}{\mathbb{E}[z_{i}^{2}]}\ne\delta$$

Motivating Example: Omitted Variables

Consider the model $y_{i}=z_{i}^{\prime}\delta+q_{i}^{\prime}\gamma+\eta_{i}$ but for some reason we can't observe $q_i$.

We instead estimate the model $y_{i}=z_{i}^{\prime}\delta+\epsilon_{i}$.

The error term is now $\epsilon_{i}=q_{i}^{\prime}\gamma+\eta_{i}$.

It is easy to verify that $\mathbb{E}[z_{i}\epsilon_{i}]=\mathbb{E}[z_{i}q_{i}^{\prime}]\gamma$ which is generally non-zero except when the included and excluded regressors are uncorrelated.

Thus, OLS becomes inconsistent.

Motivating Example: Measurement Error

Consider the model $y_{i}=\delta z_{i}^{*}+\eta_{i}$.

Suppose that $Z_{i}^{*}$ is not observed, but we instead observe a noisy version $Z_{i}=Z_{i}^{*}+u_{i}$.

Suppose we ignore the measurement error and estimate $y_{i}=\delta z_{i}+\epsilon_{i}$.

Now $\epsilon_{i}=\eta_{i}-\delta u_{i}$.

It is easy to verify that $\mathbb{E}[z_{i}\epsilon_{i}]=\mathbb{E}[(z_{i}^{*}+u_{i})(\eta_{i}-\delta u_{i})]\ne0$.

The OLS estimator of $y_{i}$ on $Z_{i}$ is inconsistent.

Model and Assumptions

The model we study is a generalization of the previously considered.

Linearity: $y_{i}=z_{i}^{\prime}\delta+\epsilon_{i}$, where $Z_{i}$ is $L$-dimensional.

Ergodic stationarity: $\{w_{i}:=(y_{i},z_{i},x_{i})\}$ is jointly stationary and ergodic, where $X_{i}$ is a $K$-dimensional vector of instruments.

Orthogonality conditions: $\mathbb{E}[x_{ik}\epsilon_{i}]=0$ for all $i, k$.

Rank condition: The $K\times L$ matrix $\Sigma_{xz}:=\mathbb{E}[x_{i}z_{i}^{\prime}]$ is of full column rank.

MDS: Let $g_{i}:=x_{i}\epsilon_{i}$. $\{g_{i}\}$ is an MDS with $S:=Avar(\overline{g})=\mathbb{E}[g_{i}g_{i}^{\prime}]$ nonsingular.

Fourth moments: $\mathbb{E}[x_{ik}^{2}z_{il}^{2}]$ exists and is finite for all $k,l$.

Discussion of Assumptions

Same as before.

Same as before, now we add one more variable.

Here orthogonality conditions are about the instruments, not the included regressors.

$X_{i}$ and $Z_{i}$ may share variables.

It's important to include in $X_{i}$ all predetermined regressors, since the orthogonality conditions are all that's exploited by GMM.

By the orthogonality conditions, $\mathbb{E}[x_{i}(y_{i}-z_{i}^{\prime}\delta)]=0$ is a system of $K$ equations in $L$ unknowns. The rank condition is necessary and sufficient for a unique solution. Clearly, a necessary condition for identification is $K\ge L$.

The system can be overidentified ($K>L$), exactly identified ($K=L$), or underidentified ($K<L$).

Same as before.

Same as before.

Method of Moments Principle

Let $g(w_{i};\delta)=g_{i}=x_{i}(y_{i}-z_{i}^{\prime}\delta)$. By the orthogonality conditions, $\mathbb{E}[g(w_{i};\delta)]=0$.

The basic principle of the method of moments is to choose $\delta$ such that the corresponding sample moments are also zero:

$$g_{n}(\delta):=\frac{1}{n}\sum_{i=1}^{n}g(w_{i};\delta)=0$$

Note $g_{n}(\delta)=\frac{1}{n}\sum_{i=1}^{n}x_{i}y_{i}-\frac{1}{n}\sum_{i=1}^{n}x_{i}z_{i}^{\prime}\delta:=\mathcal{S}_{xy}-S_{xz}\delta$.

Thus, $S_{xz}\delta=s_{xy}$ is a system of $K$ linear equations in $L$ unknowns.

To cover the case $K>L$, we need the generalized method of moments.

Method of Moments

If the system is exactly identified, then $K=L$ and $\Sigma_{xz}$ is square and invertible.

Since under Assumption 2 $S_{xz}=\Sigma_{xz}+o_{p}(1)$, $S_{xz}$ is invertible for sufficiently large $n$ with probability one.

Therefore, when the sample size is large, the system has a unique solution given by

$$\hat{\delta}_{iv}=\mathcal{S}_{xz}^{-1}S_{xy}=\left(\frac{1}{n}\sum_{i=1}^{n}x_{i}z_{i}^{\prime}\right)^{-1}\frac{1}{n}\sum_{i=1}^{n}x_{i}y_{i}$$

This estimator is called the instrumental variables (IV) estimator.

If $x_{i}=z_{i}$, then $\hat{\delta}_{iv}$ reduces to the OLS estimator.

Generalized Method of Moments

If the system is overidentified, $K>L$, we can't generally find an $L$-dimensional vector $\delta$ to satisfy the $K$ equations.

Instead of setting $g_{n}(\delta)$ to zero, our aim should be to set it as close to zero as possible.

We measure the distance between two vectors by the weighted quadratic form $(\xi-\eta)^{\prime}W(\xi-\eta)$, where $W$ is symmetric and positive definite.

Let $\hat{W}$ be a $K\times K$ symmetric positive definite matrix such that $\hat{W}\xrightarrow{p} W$ with $W$ symmetric positive definite.

The GMM estimator of $\delta$ is

$$\hat{\delta}(\hat{W}):=\arg\min J(\delta,\hat{W}); \quad J(\delta,\hat{W})=ng_{n}(\delta)^{\prime}\hat{W}g_{n}(\delta)$$

This defines a class of estimators depending on $\hat{W}$.

Generalized Method of Moments (Derivation)

The objective function is a special case of a minimum distance problem.

In our case, since $g_{n}(\delta)$ is linear in $\delta$, the objective function is quadratic in $\delta$:

$$J(\delta,\hat{W})=n(s_{xy}-S_{xz}\delta)^{\prime}\hat{W}(s_{xy}-S_{xz}\delta)$$

The FOC for minimization yields

$$S_{xz}^{\prime}\hat{W}S_{xy}=S_{xz}^{\prime}\hat{W}S_{xz}\delta$$

Under Assumptions 2 and 4, $S_{xz}^{\prime}\hat{W}S_{xz}$ is nonsingular for sufficiently large $n$.

The unique solution is then the GMM estimator

$$\hat{\delta}(\hat{W})=(S_{xz}^{\prime}\hat{W}S_{xz})^{-1}S_{xz}^{\prime}\hat{W}S_{xy}$$

If $K=L$ then $S_{xz}$ is a square matrix and $\hat{\delta}(\hat{W})=\hat{\delta}_{iv}$ for any $\hat{W}$.

Asymptotic Distribution

Following the same arguments as for the linear regression, we have the next result.

Proposition 1

Under Assumptions 1-4, $\hat{\delta}(\hat{W})\xrightarrow{p}\delta$.

If Assumption 3 is strengthened to Assumption 5, then $\sqrt{n}(\hat{\delta}(\hat{W})-\delta)\xrightarrow{d} N(0,Avar(\hat{\delta}(\hat{W})))$ where

$$Avar(\hat{\delta}(\hat{W}))=(\Sigma_{xz}^{\prime}W\Sigma_{xz})^{-1}\Sigma_{xz}^{\prime}WSW\Sigma_{xz}(\Sigma_{xz}^{\prime}W\Sigma_{xz})^{-1}$$

Key observation: GMM is consistent and asymptotically normal for any $\hat{W}$, BUT the asymptotic variance depends on $W$.

Thus, $\hat{W}$ will play an important role for efficiency.

Sketch of the Proof

The sampling error is $\hat{\delta}(\hat{W})-\delta=(S_{xz}^{\prime}\hat{W}S_{xz})^{-1}S_{xz}^{\prime}\hat{W}\overline{g}$ where $\overline{g}=g_{n}(\delta)$.

By Assumptions 2, 3 and 5, $S_{xz}\xrightarrow{p}\Sigma_{xz}, \overline{g}\xrightarrow{p} 0, \sqrt{n}\overline{g}\xrightarrow{d} N(0,S)$.

The result immediately follows by Slutsky's theorem.

Variance Estimation

Using the exact same arguments as for the OLS, we can derive the following.

Proposition 2

For any consistent estimator $\hat{\delta}$, let $\hat{\epsilon}_{i}=y_{i}-z_{i}^{\prime}\hat{\delta}$. Under Assumptions 1-4, $\hat{\sigma}^{2}=\frac{1}{n}\sum_{i=1}^{n}\hat{\epsilon}_{i}^{2}\xrightarrow{p}\mathbb{E}[\epsilon_{i}^{2}]$.
Under Assumptions 1-6, $\hat{S}=\frac{1}{n}\sum_{i=1}^{n}\hat{\epsilon}_{i}^{2}x_{i}x_{i}^{\prime}\xrightarrow{p}\mathbb{E}[\epsilon_{i}^{2}x_{i}x_{i}^{\prime}]=S$.

An immediate implication is that

$$\hat{Avar}(\hat{\delta}(\hat{W}))=(S_{xz}^{\prime}\hat{W}S_{xz})^{-1}S_{xz}^{\prime}\hat{W}\hat{S}\hat{W}S_{xz}(S_{xz}^{\prime}\hat{W}S_{xz})^{-1}\xrightarrow{p} Avar(\hat{\delta}(\hat{W}))$$

Inference

Based on asymptotic normality:

Under $H_{0}:\delta_{l}=\overline{\delta}$

$$T=\frac{\sqrt{n}(\hat{\delta}(\hat{W})_{l}-\overline{\delta})}{\sqrt{[\hat{Avar}(\hat{\delta}(\hat{W}))]_{ll}}}\xrightarrow{d} N(0,1)$$

Under $H_{0}:R\delta=r$,

$$W=n(R\hat{\delta}(\hat{W})-r)^{\prime}[R\hat{Avar}(\hat{\delta}(\hat{W}))R^{\prime}]^{-1}(R\hat{\delta}(\hat{W})-r)\xrightarrow{d}\chi_{d}^{2}$$

Under $H_{0}:a(\delta)=0$,

$$W=na(\hat{\delta}(\hat{W}))^{\prime}[A(\hat{\delta}(\hat{W}))\hat{Avar}(\hat{\delta}(\hat{W}))A(\hat{\delta}(\hat{W}))^{\prime}]^{-1}a(\hat{\delta}(\hat{W}))\xrightarrow{d}\chi_{d}^{2}$$

Based on the bootstrap: use pairs bootstrap by drawing with replacement $\{(y_{i},z_{i},x_{i})\}$. Importantly, we draw indices $i$, not single variables.

Efficient GMM

Naturally, we want to choose $W$ that yields the lowest variance.

Proposition 3

A lower bound for $Avar(\hat{\delta}(\hat{W}))$ is given by $(\Sigma_{xz}^{\prime}S^{-1}\Sigma_{xz})^{-1}$, which is achieved if $\hat{W}$ is such that $\hat{W}\xrightarrow{p} W=S^{-1}$.

Proof.

The result immediately follows if we could show, for any positive definite $W$,

$$(\Sigma_{xz}^{\prime}W\Sigma_{xz})^{-1}\Sigma_{xz}^{\prime}WSW\Sigma_{xz}(\Sigma_{xz}^{\prime}W\Sigma_{xz})^{-1}\ge(\Sigma_{xz}^{\prime}S^{-1}\Sigma_{xz})^{-1}$$

or, equivalently,

$$\Sigma_{xz}^{\prime}W\Sigma_{xz}(\Sigma_{xz}^{\prime}WSW\Sigma_{xz})^{-1}\Sigma_{xz}^{\prime}W\Sigma_{xz}\le\Sigma_{xz}^{\prime}S^{-1}\Sigma_{xz}$$

Let $S^{-1}=C^{\prime}C$, and define $A=C\Sigma_{xz}$, $D=C^{-1^{\prime}}W\Sigma_{xz}$. Then $A^{\prime}A-A^{\prime}D(D^{\prime}D)^{-1}D^{\prime}A=A^{\prime}M_{D}A\ge0$.

The efficient GMM estimator is $\hat{\delta}(\hat{S}^{-1})=(S_{xz}^{\prime}\hat{S}^{-1}S_{xz})^{-1}S_{xz}^{\prime}\hat{S}^{-1}S_{xy}$.

Implementing the Efficient GMM

To implement the efficient GMM, we need $\hat{S}$, but this in turn depends on the residuals, which rely on a preliminary consistent estimate of $\delta$.

Several options exist:

2-step efficient GMM: In the first step, compute $\hat{\delta}(\hat{W})$ by using $\hat{W}=I$ or $\hat{W}=S_{xx}^{-1}$. In the second step, compute $\hat{\delta}(\hat{S}^{-1})$ using $e_{i}=y_{i}-z_{i}^{\prime}\hat{\delta}(\hat{W})$.

Iterative GMM: Iterate the 2-step process until convergence.

Continuous-updating GMM: Solve $\hat{\delta}=\arg\min_{\delta}J(\delta,\hat{S}(\delta)^{-1})$.

Asymptotic (Local) Power

It seems intuitively obvious that test statistics associated with the efficient GMM should be preferred.

At least in large samples; in small samples, the efficient GMM might perform poorly because $\hat{S}$ estimates fourth moments.

Take for instance the t-test for $H_{0}:\delta_{l}=\overline{\delta}$.

Consider a sequence of local alternatives $\delta_{l}=\overline{\delta}+\gamma/\sqrt{n}$.

The power under any fixed alternative approaches 1 regardless of $W$, so power is of little use to choose $W$.

It's easy to show that $T\xrightarrow{d} N(\mu,1)$, where $\mu=\gamma/\sqrt{[Avar(\hat{\delta}(\hat{W}))]_{ll}}$.

Evidently, the larger is $|\mu|$, the higher is the asymptotic power, $\mathbb{P}(|N(\mu,1)|>z_{1-\alpha/2})$.

But $|\mu|$ decreases with the asymptotic variance, so the asymptotic power against local alternatives is maximized by the efficient GMM.

Testing Overidentifying Restrictions

In an exactly identified system, it's possible to choose $\delta$ such that $J(\delta,W)=0$.

In an overidentified system, we would expect $J(\delta,W)$ to be close to zero.

It turns out that, for a particular choice of $W$, $J(\delta,W)$ has a nice asymptotic behavior.

Proposition 4 (Hansen's J Test)

Under Assumptions 1-6, $J(\hat{\delta}(\hat{S}^{-1}),\hat{S}^{-1})\xrightarrow{d}\chi_{K-L}^{2}$.

Intuition: Since $\sqrt{n}\overline{g}\xrightarrow{d} N(0,S)$ and $\hat{S}\xrightarrow{p} S$, $J(\delta,\hat{S}^{-1})=n\overline{g}^{\prime}\hat{S}^{-1}\overline{g}\xrightarrow{d}\chi_{K}^{2}$. If $\delta$ is replaced by $\hat{\delta}(\hat{S}^{-1})$, then the degrees of freedom are reduced to $K-L$ because we need to estimate $L$ parameters to form the sample average of $g_i$.

Sketch of the Proof (Hansen's J Test)

The sampling error of the efficient GMM is $\hat{\delta}(\hat{S}^{-1})-\delta=(S_{xz}^{\prime}\hat{S}^{-1}S_{xz})^{-1}S_{xz}^{\prime}\hat{S}^{-1}\overline{g}$.

The sample moment condition then reads as $g_{n}(\hat{\delta}(\hat{S}^{-1}))=B\overline{g}$, where

$$B=I_{K}-S_{xz}(S_{xz}^{\prime}\hat{S}^{-1}S_{xz})^{-1}S_{xz}^{\prime}\hat{S}^{-1}$$

Let $\hat{S}^{-1}=C^{\prime}C$ and note that $CB=M_{A}C$ for $A=CS_{xz}$.

It follows that $J(\hat{\delta}(\hat{S}^{-1}),\hat{S}^{-1})=n\overline{g}^{\prime}C^{\prime}M_{A}C\overline{g}$.

We deduce the result since $\sqrt{n}C\overline{g}\xrightarrow{d} N(0,I_{K})$ and $\text{rank}(M_{A})=K-L$ with probability approaching one.

Remarks (J Test)

This is a specification test about all the restrictions of the model (Assumptions 1-6).

Only when we are confident about Assumptions 1-2, 4-6 we can interpret a large $J$ as evidence for endogeneity.

The $J$ test is not consistent against some failures of the orthogonality conditions.

This is because we lost $L$ degrees of freedom.

More specifically, since $BS_{xz}=0$, $B$ is not of full column rank. Thus, $\sqrt{n}B\overline{g}$ (hence $J$) may remain finite even if $\mathbb{E}[g_{i}]\ne0$ and the elements of $\sqrt{n}\overline{g}$ diverge.

Testing a Subset of Orthogonality Conditions

Suppose we can divide the $K$ instruments into two groups.

$K_{1}$ variables $X_{i1}$ are known to satisfy the orthogonality conditions, and the $K-K_{1}$ remaining $X_{i2}$ are suspect.

Assume wlog $x_{i}=(x_{i1}^{\prime},x_{i2}^{\prime})^{\prime}$.

The part of the model we wish to test is $\mathbb{E}[x_{i2}\epsilon_{i}]=0$.

This restriction is testable if $K_{1}\ge L$.

The basic idea is to compare two $J$ statistics from two separate GMM estimators, one using only $X_{i1}$ and another using $x_i$.

If the inclusion of $X_{i2}$ significantly increases $J$, we have evidence against $\mathbb{E}[x_{i2}\epsilon_{i}]=0$.

Notation and Main Result (C Test)

We can partition the relevant objects as

$$g_{n}(\delta)=\begin{bmatrix}g_{1n}(\delta)\\ g_{2n}(\delta)\end{bmatrix} \quad (K_{1}\times 1) \text{ and } ((K-K_{1})\times 1)$$

$$S_{(K\times K)}=\begin{bmatrix}S_{11}&S_{12}\\ S_{21}&S_{22}\end{bmatrix}$$

where $S_{11}=\mathbb{E}[\epsilon_{i}^{2}x_{i1}x_{i1}^{\prime}]$, $S_{12}=\mathbb{E}[\epsilon_{i}^{2}x_{i1}x_{i2}^{\prime}]$ and $S_{22}=\mathbb{E}[\epsilon_{i}^{2}x_{i2}x_{i2}^{\prime}]$.

We deduce that:

$$\hat{\delta}=(S_{xz}^{\prime}\hat{S}^{-1}S_{xz})^{-1}S_{xz}^{\prime}\hat{S}^{-1}s_{xy}$$

$$J=ng_{n}(\hat{\delta})^{\prime}\hat{S}^{-1}g_{n}(\hat{\delta})$$

$$\overline{\delta}=(S_{x_{1}z}^{\prime}\hat{S}_{11}^{-1}S_{x_{1}z})^{-1}S_{x_{1}z}^{\prime}\hat{S}_{11}^{-1}S_{x_{1}y}$$

$$J_{1}=ng_{1n}(\overline{\delta})^{\prime}\hat{S}_{11}^{-1}g_{1n}(\overline{\delta})$$

Proposition 5

Under Assumptions 1-6, $\mathcal{C}:=J-J_{1}\xrightarrow{d}\chi_{K-K_{1}}^{2}$.

Some Remarks (C Test)

The $C$ test is still a specification test, which relies on the other assumptions holding true.

The choices of $\hat{S}$ and $\hat{S}_{11}$ don't matter asymptotically, but may make $\mathcal{C}$ negative in finite samples.

Solution: use the same $\hat{S}$ throughout.

The proof follows similar steps as for the $J$ test and is left as an exercise.

See Chapter 3, Analytical Exercise 7 in Hayashi for hints.

Hypothesis Testing by the Likelihood-Ratio Principle

For the ML, we showed there existed a trinity of tests: Wald, LR and LM.

We showed a few slides ago how to test $H_{0}:a(\delta)=0$ using the Wald statistic.

Despite the LR is inherently a test for ML, we can define an analogous test based on a pseudo-likelihood.

$\frac{1}{n}\sum_{i=1}^{n}\log f(y_{i}|x_{i};\delta)$

The objective function now is not a log-likelihood but a minimum distance problem $J(\delta,W)$.

However, we can similarly use the LR principle to define

$$LR=J(\overline{\delta}(\hat{S}^{-1}),\hat{S}^{-1})-J(\hat{\delta}(\hat{S}^{-1}),\hat{S}^{-1})$$

where $\overline{\delta}(\hat{S}^{-1})=\arg\min_{\delta:H_{0}}J(\delta,\hat{S}^{-1})$ is the restricted efficient GMM estimator.

Wald versus LR statistic

It turns out that the LR statistic for GMM behaves similarly as for the MLE.

Proposition 6

Suppose Assumptions 1-6 hold and consider testing $H_{0}:a(\delta)=0$. Then,

$W\xrightarrow{d}\chi_{d}^{2}$ and $LR\xrightarrow{d}\chi_{d}^{2}$.

$LR-W=o_{p}(1)$.

If $a(\delta)=R\delta-r$, then $W=LR$.

The result says that both the W and LR statistics are asymptotically equivalent.

Note that (3) is stronger than (2), which in turn is stronger than (1).

The formal proof requires some tools that you will learn in AEMII.

Wald versus LR statistic (Continued)

The advantage of LR over W is invariance.

The numerical value doesn't depend on how the restrictions are represented by $a(\cdot)$.

We need $\hat{W}=\hat{S}^{-1}$, otherwise LR is not asymptotically $\chi_{d}^{2}$.

In contrast, the Wald test is $\chi_{d}^{2}$ regardless of $\hat{W}$.

As for the C test, the same $\hat{S}$ should be used to guarantee nonnegativity in finite samples.

We should also use the same $\hat{S}$ for the Wald test to fulfill (3).

Parts (2) and (3) imply that the outcome of the tests, not just the probability of rejection, will be the same in large samples if the hypothesis is true.

LR for the Linear Regression Model

Since the linear regression from earlier in the course is a special case of a GMM framework, we can derive the LR statistic.

In that case, we have $x_{i}=Z_{i}$. As a consequence:

The unrestricted efficient GMM estimator is the OLS.

$J(\hat{\delta}(\hat{S}^{-1}),\hat{S}^{-1})=0$.

Thus, $LR=J(\overline{\delta}(\hat{S}^{-1}),\hat{S}^{-1})$.

Still fulfills the previous equivalences to the Wald test.

Implications of Conditional Homoskedasticity

So far we didn't assume anything about the conditional error variance.

It turns out that if $\mathbb{E}[\epsilon_{i}^{2}|x_{i}]=\sigma^{2}$ some results simplify.

Most simplifications come from the fact that $S=\sigma^{2}\Sigma_{xx}$.

Under homoskedasticity:

$S$ can be estimated by $\hat{\sigma}^{2}S_{xx}$, hence we don't need Assumption 6.

Efficient GMM becomes 2SLS.

$J$ becomes the Sargan statistic, and $C$ becomes the difference of two Sargan statistics.

Minimizing $J$ is equivalent to minimizing SSR.

However, the $nR^{2}$ test we used for OLS to test homoskedasticity is not so simple (or valid) anymore.

Efficient GMM Becomes 2SLS

In the efficient GMM, the weighting matrix is $\hat{S}^{-1}$.

Under homoskedasticity, $\hat{S}=\hat{\sigma}^{2}S_{xx}$.

The efficient GMM then becomes

$$\hat{\delta}(\hat{S}^{-1})=(S_{xz}^{\prime}(\hat{\sigma}^{2}S_{xx})^{-1}S_{xz})^{-1}S_{xz}^{\prime}(\hat{\sigma}^{2}S_{xx})^{-1}S_{xy}=(S_{xz}^{\prime}S_{xx}^{-1}S_{xz})^{-1}S_{xz}^{\prime}S_{xx}^{-1}S_{xy}$$

This is nothing but the 2SLS estimator $\hat{\delta}_{2sls}=\hat{\delta}(S_{xx}^{-1})$.

It doesn't depend on $\hat{\sigma}^{2}$.

An alternative representation is $\hat{\delta}(S_{xx}^{-1})=(Z^{\prime}P_{X}Z)^{-1}Z^{\prime}P_{X}Y$.

This representation explains the name: (1) regress $Z_{i}$ on $X_{i}$ to obtain $\hat{Z}=P_{X}Z$, (2) regress $y_{i}$ on $\hat{Z}_{i}$.

When $K=L$ this further boils down to the IV estimator $\hat{\delta}_{iv}=S_{xz}^{-1}s_{xy}=(Z^{\prime}X)^{-1}Z^{\prime}Y$.

In sum, under homoskedasticity, there's no need for a multi-step estimation due to $\hat{W}$.

Variance Estimation and Sargan Statistic

The asymptotic variance simplifies to

$$Avar(\hat{\delta}_{2sls})=(\Sigma_{xz}^{\prime}(\sigma^{2}\Sigma_{xx})^{-1}\Sigma_{xz})^{-1}=\sigma^{2}(\Sigma_{xz}^{\prime}\Sigma_{xx}^{-1}\Sigma_{xz})^{-1}$$

A natural estimator is $\hat{Avar}(\hat{\delta}_{2sls})=\hat{\sigma}^{2}(S_{xz}^{\prime}S_{xx}^{-1}S_{xz})^{-1}$, with $\hat{\sigma}^{2}=\frac{1}{n}\sum_{i=1}^{n}(y_{i}-z_{i}^{\prime}\hat{\delta}_{2sls})^{2}$.

Easy to prove consistency.

With this choice, $T$ remains $N(0,1)$ and $W$ remains $\chi_{d}^{2}$ asymptotically.

Moreover, the GMM distance becomes the Sargan statistic:

$$J(\hat{\delta}_{2sls},(\hat{\sigma}^{2}S_{xx})^{-1})=n\frac{(S_{xy}-S_{xz}\hat{\delta}_{2sls})^{\prime}S_{xx}^{-1}(S_{xy}-S_{xz}\hat{\delta}_{2sls})}{\hat{\sigma}^{2}}\xrightarrow{d}\chi_{K-L}^{2}$$

Connection Between J and SSR

When all the regressors are predetermined and errors homoskedastic,

$$\begin{aligned}
  J(\delta,(\hat{\sigma}^{2}S_{xx})^{-1}) &= \frac{(y-Z\delta)^{\prime}P_{x}(y-Z\delta)}{\hat{\sigma}^{2}} = \frac{y^{\prime}P_{x}y-2y^{\prime}P_{x}Z\delta+\delta^{\prime}Z^{\prime}P_{x}Z\delta}{\hat{\sigma}^{2}} \\
  &= \frac{y^{\prime}P_{x}y-2y^{\prime}Z\delta+\delta^{\prime}Z^{\prime}Z\delta}{\hat{\sigma}^{2}} = \frac{(y-Z\delta)^{\prime}(y-Z\delta)}{\hat{\sigma}^{2}}-\frac{y^{\prime}y-y^{\prime}P_{x}y}{\hat{\sigma}^{2}} \\
  &= \frac{(y-Z\delta)^{\prime}(y-Z\delta)}{\hat{\sigma}^{2}}-\frac{(y-\hat{y})^{\prime}(y-\hat{y})}{\hat{\sigma}^{2}}
  \end{aligned}$$

since $P_{X}Z=Z$ when $Z_{i}\subset X_{i}$, where $\hat{y}=P_{x}y$ is the vector of fitted values from OLS.

Since the last term doesn't depend on $\delta$, minimizing $J$ amounts to minimizing the sum of squared residuals $(y-Z\delta)^{\prime}(y-Z\delta)$.

The efficient GMM is OLS.

The restricted efficient GMM is the restricted OLS.

$W=LR=(SSR_{H_{0}}-SSR)/\hat{\sigma}^{2}$.

Nonlinear GMM

So far we applied GMM to linear models $y_{i}=z_{i}^{\prime}\delta+\epsilon_{i}$.

In that case, $g(w_{i};\delta)=x_{i}(y_{i}-z_{i}^{\prime}\delta)$ is still a linear function.

GMM can be readily applied to more complex nonlinear models $a(y_{i},z_{i};\theta)=\epsilon_{i}$.

Simply set $g(w_{i};\theta)=x_{i}a(y_{i},z_{i};\theta)$.

Example: Nonlinear consumption Euler equation

$$\mathbb{E}\left[R_{t+1}\frac{\beta u^{\prime}(c_{t+1})}{u^{\prime}(c_{t})}|I_t\right]=1 \Rightarrow \mathbb{E}[x_{i}(R_{t+1}\beta(\frac{c_{t+1}}{c_{t}})^{-\alpha}-1)]=0,$$

for $u(c)=c^{1-\alpha}/(1-\alpha)$ and any $x_{i}\in I_{t}$.

Nonlinear GMM (Continued)

Recall the GMM objective function is

$$Q_{n}(\theta)=-\frac{1}{2}g_{n}(\theta)^{\prime}\hat{W}g_{n}(\theta), \quad \text{with } g_{n}(\theta)=\frac{1}{n}\sum_{i=1}^{n}g(w_{i};\theta).$$

As for the ML, we can apply the MVT to the FOC.

We now apply it to $g_{n}(\theta)$, not its first derivative.

Thus, unlike for ML, the objective function needs to be continuously differentiable only once, not twice.

This reflects that the sample average enters the objective function differently for GMM.

Nonlinear GMM (Derivation)

From the FOC, letting $G_{n}(\theta)=\partial g_{n}(\theta)/\partial\theta^{\prime}$ be the Jacobian of $g_{n}(\theta)$,

$$0=\frac{\partial Q_{n}(\hat{\theta})}{\partial\theta}=-G_{n}(\hat{\theta})^{\prime}\hat{W}g_{n}(\hat{\theta})$$

Applying the MVT to $g_{n}(\theta)$ — not to $\partial Q_{n}(\theta)/\partial\theta$ as in ML — we obtain

$$g_{n}(\hat{\theta})=g_{n}(\theta_{0})+G_{n}(\overline{\theta})(\hat{\theta}-\theta_{0})$$

Putting things together and solving for $(\hat{\theta}-\theta_{0})$ yields

$$\sqrt{n}(\hat{\theta}-\theta_{0})=-[G_{n}(\hat{\theta})^{\prime}\hat{W}G_{n}(\overline{\theta})]^{-1}G_{n}(\hat{\theta})^{\prime}\hat{W}\frac{1}{\sqrt{n}}\sum_{i=1}^{n}g(w_{i};\theta_{0})$$

This general derivation is valid for a wider class of estimators called "Extremum estimators".

You will learn more about this next quarter.

LM Statistic

For the MLE, we derived the trinity of tests (W, LR, LM) and showed they were asymptotically equivalent.

It turns out the same holds for the GMM.

The sampling error for GMM has a common format as for ML.

We already compared W and LR, LM is missing.

The Lagrange problem for $H_{0}:a(\theta)=0$ is now

$$\sqrt{n}\frac{\partial Q_{n}(\theta)}{\partial\theta}+A(\theta)^{\prime}\sqrt{n}\gamma_{n}=0$$

$$\sqrt{n}a(\theta)=0$$

The derivation follows exactly the same steps as for ML, now using $\frac{\partial Q_{n}(\theta)}{\partial\theta}$ instead of the score:

$$LM=n\gamma_{n}^{\prime}[A(\tilde{\theta})\tilde{\Sigma}^{-1}A(\tilde{\theta})^{\prime}]^{-1}\gamma_{n}=n\left(\frac{\partial Q_{n}(\tilde{\theta})}{\partial\theta}\right)^{\prime}\tilde{\Sigma}^{-1}\left(\frac{\partial Q_{n}(\tilde{\theta})}{\partial\theta}\right)\xrightarrow{d}\chi_{d}^{2} \quad \text{under } H_{0}$$

GMM versus ML

So far, we took orthogonality conditions given and tried to find $W$ that implies lowest variance.

An alternative question is: what is the optimal choice of orthogonality conditions?

It can be shown that the inverse of the information matrix $\mathbb{E}[s(w_{i},\theta)s(w_{i},\theta)^{\prime}]$ is the lower bound for the asymptotic variance of GMM estimators.

Asymptotic efficiency of ML over GMM follows because $\mathbb{E}[s(w_{i},\theta)s(w_{i},\theta)^{\prime}]^{-1}$ is precisely the asymptotic variance of ML.

Intuitively, ML exploits all likelihood information whereas GMM only exploits certain moments.

GMM achieves the variance lower bound when $g(w_{i};\theta)=\frac{\partial \log f(w_{i},\theta)}{\partial\theta}=s(w_{i},\theta)$.

Thus, the GMM estimator with optimal orthogonality conditions is (numerically) equivalent to ML.

The restriction implies that the number of moments equals the number of parameters.

Additional Readings

Hayashi Chapters 3, 4 and 7.