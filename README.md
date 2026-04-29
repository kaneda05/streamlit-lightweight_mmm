# Marketing Mix Modeling (MMM) Dashboard

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-FF4B4B.svg)](https://streamlit.io/)
[![LightweightMMM](https://img.shields.io/badge/LightweightMMM-0.1.9-green.svg)](https://github.com/google/lightweight_mmm)

## 📌 概要 (Overview)
本アプリケーションは、Googleが開発したベイズ統計モデリングライブラリ「**LightweightMMM**」を活用し、マーケティング投資の最適化を行うためのStreamlitベースのWEBダッシュボードです。

データアナリストやメディアプランナーが、過去のプロモーション実績（広告費やKPI）から各施策の真の貢献度やROI（投資対効果）を可視化し、次期の最適な予算アロケーション（配分）をシミュレーションするためのツールとして設計されています。

## ✨ 主な機能 (Key Features)

1. **探索的データ分析 (EDA)**
   - アップロードされたCSVデータの欠損値チェック機能
   - 各変数間の相関関係を可視化するヒートマップ（Seabornによる描画）
2. **ベイズ推定によるモデリング (Bayesian MCMC Modeling)**
   - JAX/NumPyroエンジンを利用した高速なマルコフ連鎖モンテカルロ法（MCMC）による学習
   - 広告の残存効果（Adstock）や季節性（Seasonality）を考慮した高精度な予測
3. **分析結果レポート (Business Dashboards)**
   - **Model Fit**: 実績値と予測値の乖離（精度）の確認
   - **Contribution & ROI**: メディアごとの売上貢献度と費用対効果の可視化
   - **Adstock Posteriors**: 広告効果の持続期間（減衰率）のパラメータ分布
4. **最適予算配分シミュレーション (Budget Optimization)**
   - 過去と同じ総予算の条件下で、KPI（売上等）を最大化する理想的な予算配分を自動計算
   - 極端な予算変動を防ぐためのビジネス制約（過去実績の20%〜200%以内）を考慮した最適化アルゴリズムの実装

## 🛠 技術スタック (Tech Stack)
- **Frontend / Backend**: Streamlit
- **MMM Engine**: LightweightMMM (by Google)
- **Computation**: JAX, jaxlib, SciPy
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, japanize-matplotlib

## 🚀 デプロイと環境構築の注意点 (Deployment Notes)
このアプリケーションは **Streamlit Community Cloud** へのデプロイを前提としています。
LightweightMMMと最新の計算エンジン（JAX等）のバージョン衝突を防ぐため、`requirements.txt` には極めて厳格なバージョン指定を行っています。

ローカルまたはクラウド環境で構築する際は、以下の構成を維持してください（Python 3.10環境を推奨）。

```text
streamlit
pandas
numpy<2.0.0
scipy<1.13.0
japanize-matplotlib
lightweight_mmm==0.1.9
jax==0.4.20
jaxlib==0.4.20