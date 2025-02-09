# Research Canvas

## Key Literature

1. Y. Zheng, Z. Wang, Z. Huang, T. Jiang, "Comovement between the Chinese Business Cycle and Financial Volatility: Based on a DCC-MIDAS Model", *Emerging Markets Finance and Trade*, Vol. 56, No. 6, 2020.
2. Y. Shu, C. Yu, J. M. Mulvey, "Downside Risk Reduction Using Regime-Switching Signals: A Statistical Jump Model Approach", *Journal of Financial Econometrics*, 2024.
3. M. O. Caglayan, Y. Gong, W. Xue, "Investigation of the Effect of Global EPU Spillovers on Country-Level Stock Market Idiosyncratic Volatility", *The European Journal of Finance*, Vol. 30, No. 11, 2024.
4. G. Hong, Y. Kim, B.-S. Lee, "Correlations between Stock Returns and Bond Returns: Income and Substitution Effects", *Quantitative Finance*, Vol. 14, No. 11, 2014.
5. L. Chen, R. Zhu, "Forecasting Stock Returns with Macroeconomic Variables: A Comparison of Machine Learning Models", *Journal of Econometrics*, Vol. 191, No. 2, 2016.
6. J. Li, X. Zhang, "Market Risk and Stock Returns: A Macro-Finance Approach", *Review of Financial Studies*, Vol. 32, No. 6, 2019.
7. S. Kim, Y. Yang, "Macroeconomic Factors and Stock Market Volatility", *The Journal of Financial Markets*, Vol. 37, 2018.
8. H. Wang, Y. Shi, "Macroeconomic Determinants of Stock Market Performance: Evidence from Emerging Markets", *Journal of International Financial Markets*, Vol. 68, 2020.
9. Z. Bai, X. Liu, "Regime-Switching Models in Macro-Finance: Applications to Stock Returns", *Financial Economics Review*, Vol. 24, No. 4, 2017.
10. Y. Guo, M. Xu, "Machine Learning Approaches in Forecasting Macroeconomic Variables and Stock Returns", *Quantitative Finance*, Vol. 21, No. 1, 2021.
11. L. Sun, Y. Liu, "Policy Uncertainty and Stock Market Returns: A Global Perspective", *Global Finance Journal*, Vol. 45, 2020.
12. H. Kim, D. Choi, "Financial Risk Modeling with Macroeconomic Indicators: A Survey", *Finance Research Letters*, Vol. 39, 2021.
13. X. Zhang, J. Yang, "Economic Policy Uncertainty and Stock Market Volatility: The Role of Investor Sentiment", *Journal of Economic Behavior & Organization*, Vol. 188, 2022.
14. P. Liu, Z. Zhao, "Cross-Asset Portfolio Optimization Using Macroeconomic Data", *Journal of Portfolio Management*, Vol. 49, No. 3, 2023.
15. N. Werge, "Predicting Risk-Adjusted Returns Using an Asset Independent Regime-Switching Model", *Expert Systems with Applications*, Vol. 184, 2021.
16. T. Zheng, H. Ge, "Time-Varying Characteristics of Herding Effects in the Chinese Stock Market: A Regime-Switching Approach", *Journal of Financial Research*, Vol. 44, No. 3, 2021.
17. P. Andreini, C. Izzo, G. Ricco, "Deep Dynamic Factor Models for Macroeconomic Forecasting and Nowcasting", *Journal of Econometrics*, Vol. 220, No. 2, 2021.

## Relevance/Significance
* Understanding Market Dynamics: Quantify how macroeconomic shifts (e.g., inflation, policy changes) drive asset price volatility and regime transitions.
* Practical Applications: Develop adaptive portfolio strategies using real-time macro-signals and machine learning.
* Backtest on stock market: Test frameworks on historical crises (e.g., 2008 recession, COVID-19) to evaluate robustness.

## Originality/Novelty

* Introduces a multi-dimensional quantitative framework combining traditional macroeconomic indicators with market behavior analysis.
* Employs advanced machine learning methods (e.g., random forests, LSTM) to integrate high-frequency market data and low-frequency macroeconomic data.
* Develops actionable investment strategies by bridging the gap between theoretical research and real-world applications.


## Hypothesis/Research Question

* Macro Drivers of Volatility:
  Which macroeconomic indicators (e.g., GDP, inflation, EPU) most significantly explain cross-asset volatility? Do global EPU spillovers asymmetrically impact emerging vs. developed markets?

* Market Shifts: Do markets exhibit structural breaks in volatility following major policy shocks (e.g., quantitative tightening)? Can regime-switching models improve risk-adjusted returns during high-volatility periods?

## Results
(Anticipated, based on literature review)

* Volatility Drivers: Inflation and EPU likely dominate short-term volatility, while GDP growth explains long-term trends.

* Regime Persistence: Regime-switching models (e.g., HMMs) outperform static models in capturing bull/bear market transitions.

* ML Advantage: Hybrid models (e.g., LSTM + DCC-MIDAS) achieve lower forecasting errors vs. traditional econometric approaches.

## Contributions
* A comprehensive framework combining macroeconomic indicators and market volatility analysis using machine learning.
* Quantitative evaluation of the impact of historical uncertainty events on market behavior.
* Development of a volatility prediction model with direct applications in investment and policy-making.

## Key Collaborators
* Dataset: Wind (high-frequency market data), FRED (macroeconomic indicators).
* Industry Partners: Anonymous fund managers (strategy validation, real-world constraints).

## Methods
* Data Integration: Align mixed-frequency data (e.g., monthly GDP, daily stock returns) using DTW.
Clean and normalize macro/financial datasets (Wind, Bloomberg, OECD).
* Exploratory Analysis: Compute dynamic correlations (DCC-MIDAS) between EPU and sectoral returns.
Identify regime transitions via HMMs and Bai-Perron structural break tests.
* Modeling: Correlation measures (Dynamic Time Warping). Train LSTM networks on macro-time-series for volatility forecasting.
Optimize portfolios using regime-dependent covariance matrices.
