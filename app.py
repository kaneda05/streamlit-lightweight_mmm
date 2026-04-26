import os
# NumbaのコンパイルエラーをOSレベルで完全に回避
os.environ["NUMBA_DISABLE_JIT"] = "1"  

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import jax.numpy as jnp
import io

# LightweightMMM関連のインポート
from lightweight_mmm import lightweight_mmm
from lightweight_mmm import preprocessing
from lightweight_mmm import plot
from lightweight_mmm import optimize_media

# --- ユーティリティ関数 ---
def st_pyplot_with_download(fig, filename="chart.png", button_text="画像をダウンロード"):
    """グラフを描画し、その下にダウンロードボタンを配置する"""
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300, facecolor='white')
    buf.seek(0)
    st.download_button(label=f"📥 {button_text}", data=buf, file_name=filename, mime="image/png")

def display_manual_image(path, caption):
    """マニュアル用の画像を小さく中央寄せで表示する"""
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        # widthを指定、またはuse_container_width=Falseでサイズを抑制
        st.image(path, caption=f"【表示例】{caption}", use_container_width=True)

# --- ページ設定 ---
st.set_page_config(page_title="MMM 分析プラットフォーム", layout="wide")

# =====================================================================
# サイドバー：ページ遷移ナビゲーション
# =====================================================================
st.sidebar.title("操作メニュー")
page = st.sidebar.radio("移動先を選択してください", ["📊 分析ダッシュボード", "📘 操作マニュアル"])

# =====================================================================
# ページA：マニュアル
# =====================================================================
if page == "📘 操作マニュアル":
    st.title("MMM 分析ダッシュボード 操作マニュアル")
    st.info("本ページでは、各分析結果の解釈方法とビジネス上の活用ポイントを解説します。")
    
    st.markdown("---")
    st.header("① Model Fit（予測精度の検証）")
    st.markdown("""
    構築されたAIモデルが、過去の実際の売上推移をどれだけ正確に再現できているかを確認します。
    - **見方**: 黒線（実績値）に対し、青線（予測値）が連動しているかを確認してください。
    - **ビジネス判断**: 乖離が激しい場合は、突発的なキャンペーンや競合の動きなど、データに含まれていない要因が影響している可能性があります。
    """)
    display_manual_image("images/model_fit.png", "Model Fit")

    st.markdown("---")
    st.header("② Contribution & ROI（貢献度と投資対効果）")
    st.markdown("""
    どの施策が売上に寄与したか（貢献度）と、投下予算に対していくらのリターンがあったか（ROI）を分析します。
    - **Contribution**: 全売上における各メディアの寄与割合。baselineは広告なしでも発生する基礎売上を指します。
    - **ROI**: 1円あたりのリターン。効率の良い媒体の特定に活用します。
    """)
    display_manual_image("images/contribution_roi.png", "Contribution & ROI")

    st.markdown("---")
    st.header("③ Adstock（残存効果の特性）")
    st.markdown("""
    広告の「持続性」を可視化します。
    - **見方**: 分布が右側に寄っているメディアほど、一度の投下で効果が数週間にわたって持続する傾向があります。
    - **ビジネス判断**: 残存効果の高いメディアは、間隔を空けた投下でも効果を維持しやすい特性があります。
    """)
    display_manual_image("images/adstock.png", "Adstock Posteriors")

    st.markdown("---")
    st.header("④ Budget Optimization（予算配分の最適化）")
    st.markdown("""
    現在の総予算を維持したまま、売上を最大化するための理想的な配分をシミュレーションします。
    - **Previous**: 現状の配分比率。
    - **Optimal**: 統計的に導き出された最も効率的な配分案。
    - **ビジネス判断**: 次期の予算策定におけるエビデンス（論拠）として利用します。
    """)
    display_manual_image("images/budget_optimization.png", "Budget Optimization")

# =====================================================================
# ページB：分析ダッシュボード
# =====================================================================
elif page == "📊 分析ダッシュボード":
    st.title("MMM 分析ダッシュボード")

    # --- 1. データ設定 ---
    st.header("1. データ設定")
    uploaded_file = st.file_uploader("分析用CSVデータをアップロードしてください", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        date_col = [col for col in df.columns if 'date' in col.lower()][0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col).set_index(date_col)
        
        col_agg, col_target, col_media = st.columns(3)
        with col_agg:
            agg_level = st.radio("データの集計単位", ('週別', '日別'))
            if agg_level == '週別':
                df = df.resample('W-MON').sum()
        with col_target:
            target_col = st.selectbox("目的変数（KPI）", df.columns)
        with col_media:
            media_cols = st.multiselect("説明変数（メディア支出）", [col for col in df.columns if col != target_col])

        if target_col and media_cols:
            st.markdown("---")
            st.header("2. モデル構築の設定")
            
            min_date, max_date = df.index.min().date(), df.index.max().date()
            col_train, col_test = st.columns(2)
            with col_train:
                train_start = st.date_input("分析開始日", min_date, min_value=min_date, max_value=max_date)
                train_end = st.date_input("分析終了日", min_date + datetime.timedelta(days=int((max_date-min_date).days * 0.8)))
            with col_test:
                test_start = st.date_input("検証期間開始日", train_end + datetime.timedelta(days=1))
                test_end = st.date_input("検証期間終了日", max_date)

            if st.button("モデル構築を実行する", type="primary"):
                with st.spinner('ベイズ推定による計算を実行中です...'):
                    df_train = df.loc[train_start:train_end]
                    media_data_train = df_train[media_cols].values
                    target_train = df_train[target_col].values
                    costs_train = df_train[media_cols].sum().values
                    
                    media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
                    target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
                    cost_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)

                    media_data_scaled = media_scaler.fit_transform(media_data_train)
                    target_scaled = target_scaler.fit_transform(target_train)
                    costs_scaled = cost_scaler.fit_transform(costs_train)

                    mmm = lightweight_mmm.LightweightMMM(model_name="adstock")
                    mmm.fit(
                        media=media_data_scaled,
                        media_prior=costs_scaled,
                        target=target_scaled,
                        number_warmup=1000,
                        number_samples=1000,
                        seed=42
                    )
                    
                    st.session_state['mmm'] = mmm
                    st.session_state['target_scaler'] = target_scaler
                    st.session_state['cost_scaler'] = cost_scaler
                    st.session_state['media_scaler'] = media_scaler
                    st.session_state['media_cols'] = media_cols
                    st.session_state['costs_train'] = costs_train
                    st.session_state['df_train'] = df_train
                    if 'optim_result' in st.session_state:
                        del st.session_state['optim_result']
                    
                    st.success("計算が完了しました。")

            # --- 3. 分析結果表示 ---
            if 'mmm' in st.session_state:
                mmm = st.session_state['mmm']
                target_scaler = st.session_state['target_scaler']
                cost_scaler = st.session_state['cost_scaler']
                media_scaler = st.session_state['media_scaler']
                costs_train = st.session_state['costs_train']
                media_cols = st.session_state['media_cols']
                df_train = st.session_state['df_train']
                
                st.markdown("---")
                st.header("3. 分析結果レポート")
                
                media_effect_hat, roi_hat = mmm.get_posterior_metrics(target_scaler=target_scaler, cost_scaler=cost_scaler)
                
                tab1, tab2, tab3, tab4 = st.tabs([
                    "① Model Fit（精度）", 
                    "② Contribution & ROI", 
                    "③ Adstock（残存効果）", 
                    "④ Budget Optimization（最適配分）"
                ])
                
                with tab1:
                    c1, c2, c3 = st.columns([1, 4, 1])
                    with c2:
                        fig1 = plot.plot_model_fit(mmm, target_scaler=target_scaler)
                        st_pyplot_with_download(fig1, "model_fit.png", "グラフを保存")
                
                with tab2:
                    c1, c2, c3 = st.columns([1, 6, 1])
                    with c2:
                        fig2_area = plot.plot_media_baseline_contribution_area_plot(
                            media_mix_model=mmm, target_scaler=target_scaler, channel_names=media_cols
                        )
                        st_pyplot_with_download(fig2_area, "contribution_area.png", "時系列推移を保存")

                    col_bar1, col_bar2 = st.columns(2)
                    with col_bar1:
                        fig2_bar = plot.plot_bars_media_metrics(metric=media_effect_hat, metric_name="Contribution", channel_names=media_cols)
                        st_pyplot_with_download(fig2_bar, "contribution_bar.png", "貢献度を保存")
                    with col_bar2:
                        fig3 = plot.plot_bars_media_metrics(metric=roi_hat, metric_name="ROI", channel_names=media_cols)
                        st_pyplot_with_download(fig3, "roi_bar.png", "ROIを保存")
                        
                with tab3:
                    c1, c2, c3 = st.columns([1, 4, 1])
                    with c2:
                        fig_posteriors = plot.plot_media_channel_posteriors(media_mix_model=mmm, channel_names=media_cols)
                        st_pyplot_with_download(fig_posteriors, "adstock_posteriors.png", "残存パラメータを保存")

                with tab4:
                    # タブを開いた際に自動実行
                    if 'optim_result' not in st.session_state:
                        with st.spinner("予算配分の最適化を計算中..."):
                            opt_periods = len(df_train)
                            opt_budget = float(costs_train.sum())
                            current_allocation_ratio = costs_train / costs_train.sum()
                            # 配分計算を安定させるためのNumPyキャスト
                            previous_budget_allocation = np.array(opt_budget * current_allocation_ratio)

                            trace_backup = mmm.trace.copy()
                            for site in ["mu", "media_transformed", "prediction"]:
                                mmm.trace.pop(site, None)
                            
                            try:
                                # 探索範囲を適正化し、異常値（マイナス等）を抑制
                                solution, kpi_without_optim, kpi_with_optim = optimize_media.find_optimal_budgets(
                                    n_time_periods=opt_periods,
                                    media_mix_model=mmm,
                                    extra_features=None,
                                    budget=opt_budget,
                                    prices=jnp.ones(mmm.n_media_channels),
                                    media_scaler=media_scaler,
                                    target_scaler=target_scaler,
                                    bounds_lower_pct=0.2, # 過度な削減を抑える
                                    bounds_upper_pct=2.0, # 過度な増加を抑える
                                    seed=42,
                                )
                            finally:
                                mmm.trace = trace_backup
                            
                            st.session_state['optim_result'] = {
                                'solution': np.array(solution.x),
                                'kpi_before': float(jnp.mean(kpi_without_optim)),
                                'kpi_after': float(jnp.mean(kpi_with_optim)),
                                'prev_budget': previous_budget_allocation
                            }

                    res = st.session_state['optim_result']
                    c1, c2, c3 = st.columns([1, 6, 1])
                    with c2:
                        plot_kwargs = {
                            "media_mix_model": mmm,
                            "kpi_with_optim": res['kpi_after'],
                            "kpi_without_optim": res['kpi_before'],
                            "previous_budget_allocation": res['prev_budget'],
                            "figure_size": (10, 10),
                            "channel_names": media_cols
                        }
                        try:
                            # ライブラリのバージョン差異を吸収
                            fig_opt = plot.plot_pre_post_budget_allocation_comparison(
                                **plot_kwargs, optimal_buget_allocation=res['solution']
                            )
                        except TypeError:
                            fig_opt = plot.plot_pre_post_budget_allocation_comparison(
                                **plot_kwargs, optimal_budget_allocation=res['solution']
                            )

                        st_pyplot_with_download(fig_opt, "budget_optimization.png", "アロケーション案を保存")