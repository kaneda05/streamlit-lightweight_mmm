import os
# 計算エンジンのコンパイルエラーをOSレベルで回避
os.environ["NUMBA_DISABLE_JIT"] = "1"  
os.environ["PYTENSOR_FLAGS"] = "cxx=" # PyTensorのC++コンパイルエラー回避

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib # グラフの日本語化
import datetime
import jax.numpy as jnp
import io

# --- LightweightMMM ---
from lightweight_mmm import lightweight_mmm
from lightweight_mmm import preprocessing
from lightweight_mmm import plot as lmmm_plot
from lightweight_mmm import optimize_media

# --- ユーティリティ関数 ---
def st_pyplot_with_download(fig, filename="chart.png", button_text="画像をダウンロード"):
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300, facecolor='white')
    buf.seek(0)
    st.download_button(label=f"📥 {button_text}", data=buf, file_name=filename, mime="image/png")

def display_manual_image(path, caption):
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        if os.path.exists(path):
            st.image(path, caption=f"【表示例】{caption}", use_container_width=True)
        else:
            st.info(f"💡 **画像プレースホルダー: {caption}**\n\n分析後に画像を `{path}` に配置してください。")

# --- ページ設定 ---
st.set_page_config(page_title="MMM 分析プラットフォーム", layout="wide")

st.sidebar.title("操作メニュー")
page = st.sidebar.radio("移動先を選択してください", ["📊 分析ダッシュボード", "📘 操作マニュアル"])

# =====================================================================
# ページA：マニュアル
# =====================================================================
if page == "📘 操作マニュアル":
    st.title("MMM 分析ダッシュボード 操作マニュアル")
    st.info("本プラットフォームでは、Googleの「LightweightMMM」エンジンを用いて、マーケティング投資の最適化分析が可能です。")

    st.markdown("---")
    st.header("① Model Fit（予測精度の検証）")
    st.write("過去の実績（黒線）を、AIの予測モデル（色線）がどれだけトレースできているかを確認します。")
    display_manual_image("images/model_fit.png", "Model Fit")

    st.markdown("---")
    st.header("② Contribution & ROI（貢献度と投資対効果）")
    st.write("各プロモーションが全体の売上に何割寄与したか（貢献度）と、費用対効果（ROI）を分析します。")
    display_manual_image("images/contribution_roi.png", "Contribution & ROI")

    st.markdown("---")
    st.header("③ Adstock & Saturation（残存効果と収穫逓減）")
    st.write("広告効果の「長持ち度（Adstock）」を確認します。分布が右に寄っているメディアほど効果が長続きします。")
    display_manual_image("images/adstock.png", "Adstock Parameters")

    st.markdown("---")
    st.header("④ Budget Optimization（予算配分の最適化）")
    st.write("過去と同じ総予算を使って、売上を最大化するための理想の配分をシミュレーションします。")
    display_manual_image("images/budget_optimization.png", "Budget Optimization")

# =====================================================================
# ページB：分析ダッシュボード
# =====================================================================
elif page == "📊 分析ダッシュボード":
    st.title("MMM 分析ダッシュボード")

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
            # =========================================================
            # 追加実装部分：EDA（探索的データ分析）
            # =========================================================
            st.markdown("---")
            st.header("2. データの基礎分析 (EDA)")
            
            with st.expander("🔍 欠損値と相関関係を確認する（クリックで開閉）", expanded=False):
                col_eda1, col_eda2 = st.columns(2)
                
                # 左側：欠損値の確認テーブル
                with col_eda1:
                    st.subheader("欠損値の確認")
                    check_cols = [target_col] + media_cols
                    missing_data = df[check_cols].isnull().sum().reset_index()
                    missing_data.columns = ['変数名', '欠損値の数']
                    st.dataframe(missing_data, use_container_width=True)
                    if missing_data['欠損値の数'].sum() > 0:
                        st.warning("⚠️ 欠損値が含まれています。MMMを実行する前に前処理（0埋めや補間など）を行うことを推奨します。")
                    else:
                        st.success("✅ 欠損値はありません。")
                
                # 右側：相関関係のヒートマップ
                with col_eda2:
                    st.subheader("変数間の相関関係 (Heatmap)")
                    fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
                    corr_matrix = df[check_cols].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".2f", ax=ax_corr, 
                                vmin=-1, vmax=1, center=0, square=True)
                    st.pyplot(fig_corr)

            # =========================================================
            # モデル構築
            # =========================================================
            st.markdown("---")
            st.header("3. モデル構築の実行")
            
            min_date, max_date = df.index.min().date(), df.index.max().date()
            col_train, col_test = st.columns(2)
            with col_train:
                train_start = st.date_input("分析開始日", min_date, min_value=min_date, max_value=max_date)
                train_end = st.date_input("分析終了日", min_date + datetime.timedelta(days=int((max_date-min_date).days * 0.8)))
            with col_test:
                test_start = st.date_input("検証期間開始日", train_end + datetime.timedelta(days=1))
                test_end = st.date_input("検証期間終了日", max_date)

            if st.button("LightweightMMM で分析を開始する", type="primary"):
                with st.spinner("ベイズ最適化を実行中です...（数分かかります）"):
                    for key in ['mmm_model', 'optim_result']:
                        if key in st.session_state:
                            del st.session_state[key]
                            
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

                    model = lightweight_mmm.LightweightMMM(model_name="adstock")
                    model.fit(
                        media=media_data_scaled,
                        media_prior=costs_scaled,
                        target=target_scaled,
                        number_warmup=1000,
                        number_samples=1000,
                        seed=42
                    )
                    
                    st.session_state['mmm_model'] = model
                    st.session_state['target_scaler'] = target_scaler
                    st.session_state['cost_scaler'] = cost_scaler
                    st.session_state['media_scaler'] = media_scaler
                    st.session_state['costs_train'] = costs_train
                    st.session_state['media_cols'] = media_cols
                    st.session_state['df_train'] = df_train
                    
                    st.success("計算が完了しました。")

            # --- 4. 分析結果表示 ---
            if 'mmm_model' in st.session_state:
                model = st.session_state['mmm_model']
                media_cols = st.session_state['media_cols']
                df_train = st.session_state['df_train']
                
                st.markdown("---")
                st.header("4. 分析結果レポート")
                
                tab1, tab2, tab3, tab4 = st.tabs([
                    "① Model Fit (精度)", 
                    "② Contribution (貢献度)", 
                    "③ Adstock (残存効果)", 
                    "④ Optimization (最適配分)"
                ])
                
                target_scaler = st.session_state['target_scaler']
                cost_scaler = st.session_state['cost_scaler']
                media_scaler = st.session_state['media_scaler']
                costs_train = st.session_state['costs_train']
                media_effect_hat, roi_hat = model.get_posterior_metrics(target_scaler=target_scaler, cost_scaler=cost_scaler)
                
                with tab1:
                    c1, c2, c3 = st.columns([1, 4, 1])
                    with c2:
                        fig1 = lmmm_plot.plot_model_fit(model, target_scaler=target_scaler)
                        st_pyplot_with_download(fig1, "lmmm_fit.png", "グラフを保存")
                
                with tab2:
                    c1, c2, c3 = st.columns([1, 6, 1])
                    with c2:
                        fig2_area = lmmm_plot.plot_media_baseline_contribution_area_plot(
                            media_mix_model=model, target_scaler=target_scaler, channel_names=media_cols
                        )
                        st_pyplot_with_download(fig2_area, "lmmm_area.png", "時系列推移を保存")

                    col_bar1, col_bar2 = st.columns(2)
                    with col_bar1:
                        fig2_bar = lmmm_plot.plot_bars_media_metrics(metric=media_effect_hat, metric_name="Contribution", channel_names=media_cols)
                        st_pyplot_with_download(fig2_bar, "lmmm_contribution.png", "貢献度を保存")
                    with col_bar2:
                        fig3 = lmmm_plot.plot_bars_media_metrics(metric=roi_hat, metric_name="ROI", channel_names=media_cols)
                        st_pyplot_with_download(fig3, "lmmm_roi.png", "ROIを保存")
                        
                with tab3:
                    c1, c2, c3 = st.columns([1, 4, 1])
                    with c2:
                        fig_posteriors = lmmm_plot.plot_media_channel_posteriors(media_mix_model=model, channel_names=media_cols)
                        st_pyplot_with_download(fig_posteriors, "lmmm_adstock.png", "残存パラメータを保存")

                with tab4:
                    if 'optim_result' not in st.session_state:
                        with st.spinner("予算最適化シミュレーションを実行中..."):
                            opt_periods = len(df_train)
                            opt_budget = float(costs_train.sum())
                            current_allocation = costs_train / costs_train.sum()
                            prev_budget = np.array(opt_budget * current_allocation)

                            trace_backup = model.trace.copy()
                            for site in ["mu", "media_transformed", "prediction"]:
                                model.trace.pop(site, None)
                            
                            try:
                                solution, kpi_without_optim, kpi_with_optim = optimize_media.find_optimal_budgets(
                                    n_time_periods=opt_periods, media_mix_model=model, extra_features=None,
                                    budget=opt_budget, prices=jnp.ones(model.n_media_channels),
                                    media_scaler=media_scaler, target_scaler=target_scaler,
                                    bounds_lower_pct=0.2, bounds_upper_pct=2.0, seed=42,
                                )
                            finally:
                                model.trace = trace_backup
                            
                            st.session_state['optim_result'] = {
                                'solution': np.array(solution.x),
                                'kpi_before': float(jnp.mean(kpi_without_optim)),
                                'kpi_after': float(jnp.mean(kpi_with_optim)),
                                'prev_budget': prev_budget
                            }

                    res = st.session_state['optim_result']
                    c1, c2, c3 = st.columns([1, 6, 1])
                    with c2:
                        plot_kwargs = {
                            "media_mix_model": model, "kpi_with_optim": res['kpi_after'],
                            "kpi_without_optim": res['kpi_before'], "previous_budget_allocation": res['prev_budget'],
                            "figure_size": (10, 10), "channel_names": media_cols
                        }
                        try:
                            fig_opt = lmmm_plot.plot_pre_post_budget_allocation_comparison(**plot_kwargs, optimal_buget_allocation=res['solution'])
                        except TypeError:
                            fig_opt = lmmm_plot.plot_pre_post_budget_allocation_comparison(**plot_kwargs, optimal_budget_allocation=res['solution'])
                        st_pyplot_with_download(fig_opt, "lmmm_budget_opt.png", "アロケーション案を保存")