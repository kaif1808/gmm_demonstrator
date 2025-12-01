# GMM Demonstrator

A web-based demonstration of Generalized Method of Moments (GMM) estimation for linear models with endogenous variables.

## What This Project Is

This project provides an interactive Streamlit application that demonstrates GMM estimation techniques for linear regression models with endogeneity. It includes both synthetic data generation and real data analysis capabilities, making it ideal for learning and teaching econometric methods.

## Key Features

- **Interactive Data Generation**: Generate synthetic data with various data generating processes (DGP types)
- **1-Step GMM Estimation**: Perform Two-Stage Least Squares (2SLS) equivalent estimation
- **2-Step GMM Estimation**: Implement efficient GMM with optimal weighting matrix
- **Hansen J-Test**: Test overidentifying restrictions
- **Monte Carlo Simulations**: Compare estimator performance through repeated sampling
- **Real Data Analysis**: Upload and analyze your own datasets
- **Visualization**: Interactive plots and comprehensive output tables

## Usage

1. **Data Source**: Choose between generating synthetic data or pasting your own CSV data
2. **Parameters**: Set the number of instruments (K), endogenous variables (L), sample size (n), and simulation parameters
3. **DGP Type**: Select from various data generating processes:
   - Homoskedastic
   - Heteroskedastic (Linear/Quadratic/Exponential)
   - High/Low Endogeneity
   - Invalid Instruments (Not Exogenous)
4. **Analysis**: Explore the four main tabs:
   - Data & DGP: Summary statistics and data visualization
   - 1-Step GMM: Two-stage least squares equivalent estimation
   - 2-Step GMM: Efficient GMM with optimal weighting
   - Comparison & Hansen J-Test: Monte Carlo results and specification tests

## Project Structure

- `app.py`: Streamlit web interface
- `utils/gmm_calculations.py`: Core GMM computational functions
- `gmmdetail/GMMdetail.tex`: Theoretical documentation
- `gmmlecture.md`: Comprehensive lecture notes on GMM theory
- `requirements.txt`: Python dependencies

## Technical Requirements

- Python 3.7+
- Streamlit
- NumPy
- SciPy
- Matplotlib
- Pandas

Perfect for students and researchers learning econometric methods!
