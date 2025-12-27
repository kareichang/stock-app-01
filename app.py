import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------------------
# 1. アプリ全体の設定
# -------------------------------------------
st.set_page_config(page_title="AI株価分析Pro", layout="wide", page_icon="📈")

st.title("📈 AI株価分析Pro")
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #FF4B4B;
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 24px;
}
</style>
""", unsafe_allow_html=True)

# サイドバー設定
st.sidebar.header("🔍 分析設定")

# 監視リストのプリセット
preset_list = st.sidebar.selectbox(
    "リスト選択",
    ("米国ハイテク (Mag7)", "日本株 (主力)", "暗号資産", "カスタム")
)

if preset_list == "米国ハイテク (Mag7)":
    default_tickers = "NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, AMD, AVGO"
elif preset_list == "日本株 (主力)":
    default_tickers = "7203.T, 9984.T, 8035.T, 6920.T, 6146.T, 6758.T, 8306.T, 9983.T"
elif preset_list == "暗号資産":
    default_tickers = "BTC-USD, ETH-USD, SOL-USD, XRP-USD"
else:
    default_tickers = "NVDA"

ticker_input = st.sidebar.text_area("銘柄リスト (カンマ区切り)", default_tickers)
tickers_list = [t.strip() for t in ticker_input.split(',')]

# パラメータ設定
with st.sidebar.expander("ロジック詳細設定", expanded=False):
    bb_period = st.number_input("BB期間", value=20)
    adx_threshold = st.number_input("ADX基準値", value=25)
    atr_period = st.number_input("ATR期間", value=22)
    atr_multiplier = st.number_input("ATR倍率", value=3.5)

# 個別チャート用の銘柄選択
selected_ticker = st.sidebar.selectbox("📊 チャートを表示する銘柄", tickers_list)

# -------------------------------------------
# 2. 計算＆データ処理関数
# -------------------------------------------
def get_stock_data(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty or len(df) < 50:
            return None

        # --- テクニカル計算 ---
        # BB
        df['SMA'] = df['Close'].rolling(window=bb_period).mean()
        df['STD'] = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['SMA'] + (2.0 * df['STD'])
        df['BB_Lower'] = df['SMA'] - (2.0 * df['STD'])

        # ADX
        adx_len = 14
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        
        df['UpMove'] = df['High'] - df['High'].shift(1)
        df['DownMove'] = df['Low'].shift(1) - df['Low']
        df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
        df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
        
        df['TR_s'] = df['TR'].rolling(window=adx_len).mean()
        df['+DM_s'] = df['+DM'].rolling(window=adx_len).mean()
        df['-DM_s'] = df['-DM'].rolling(window=adx_len).mean()
        
        df['+DI'] = 100 * (df['+DM_s'] / df['TR_s'])
        df['-DI'] = 100 * (df['-DM_s'] / df['TR_s'])
        df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
        df['ADX'] = df['DX'].rolling(window=adx_len).mean()

        # ATR Exit
        df['ATR'] = df['TR'].rolling(window=atr_period).mean()
        df['High_Roll'] = df['High'].rolling(window=atr_period).max()

        # シグナル判定ループ
        trend = np.zeros(len(df))
        stop_line = np.zeros(len(df))
        buy_sig = [np.nan] * len(df)
        sell_sig = [np.nan] * len(df)
        
        curr_trend = -1
        curr_stop = 0.0
        
        first_idx = max(bb_period, adx_len, atr_period)
        
        for i in range(first_idx, len(df)):
            close = df['Close'].iloc[i]
            high_roll = df['High_Roll'].iloc[i]
            atr = df['ATR'].iloc[i]
            adx = df['ADX'].iloc[i]
            bb_upper = df['BB_Upper'].iloc[i]
            
            long_stop = high_roll - (atr * atr_multiplier)
            
            if curr_trend == 1: # 保有中
                curr_stop = max(long_stop, curr_stop)
                if close < curr_stop:
                    curr_trend = -1 # 決済
                    sell_sig[i] = close
                else:
                    stop_line[i] = curr_stop
                    trend[i] = 1
            else: # 待機中
                curr_stop = long_stop
                if (close > bb_upper) and (adx > adx_threshold):
                    curr_trend = 1 # エントリー
                    buy_sig[i] = close
                    stop_line[i] = long_stop
                    trend[i] = 1
                else:
                    stop_line[i] = long_stop # 表示用に計算だけしておく

        df['StopLine'] = stop_line
        df['Trend'] = trend
        df['Buy'] = buy_sig
        df['Sell'] = sell_sig
        
        return df

    except Exception as e:
        return None

# -------------------------------------------
# 3. チャート描画関数 (TradingView風)
# -------------------------------------------
def plot_beautiful_chart(df, ticker):
    # レイアウト作成（上がローソク足、下がADX）
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.75, 0.25],
                        subplot_titles=(f"{ticker} Hybrid Strategy Chart", "ADX Trend Strength"))

    # --- 1. ローソク足 ---
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)

    # --- 2. ボリンジャーバンド (クラウド & ハイライト) ---
    # 下限バンド
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(128,128,128,0.5)', width=1),
                             mode='lines', name='BB Lower', showlegend=False), row=1, col=1)
    # 上限バンド (通常・グレー)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(128,128,128,0.5)', width=1),
                             mode='lines', fill='tonexty', fillcolor='rgba(0, 153, 255, 0.1)', # 青い雲
                             name='BB Cloud'), row=1, col=1)

    # ★ ADX連動ハイライト (ADX > 基準値 の時だけ、上限バンドをオレンジに)
    high_adx_mask = df['ADX'] > adx_threshold
    bb_upper_highlight = df['BB_Upper'].copy()
    bb_upper_highlight[~high_adx_mask] = None # 条件を満たさない場所を消す
    
    fig.add_trace(go.Scatter(x=df.index, y=bb_upper_highlight, line=dict(color='#FFAA00', width=3),
                             mode='lines', name='BB Strong (Entry Zone)'), row=1, col=1)

    # --- 3. ATR命綱 (保有中のみ表示) ---
    holding_mask = df['Trend'] == 1
    stop_line_plot = df['StopLine'].copy()
    stop_line_plot[~holding_mask] = None
    
    fig.add_trace(go.Scatter(x=df.index, y=stop_line_plot, mode='markers',
                             marker=dict(color='#00FF00', size=4), name='ATR Stop (Hold)'), row=1, col=1)

    # --- 4. 売買サイン ---
    fig.add_trace(go.Scatter(x=df.index, y=df['Buy'], mode='markers',
                             marker=dict(symbol='triangle-up', color='#FF0000', size=15, line=dict(width=1, color='black')),
                             name='BUY Signal'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Sell'], mode='markers',
                             marker=dict(symbol='triangle-down', color='#0000FF', size=15, line=dict(width=1, color='black')),
                             name='EXIT Signal'), row=1, col=1)

    # --- 5. ADX (サブプロット) ---
    fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], line=dict(color='purple', width=2), name='ADX'), row=2, col=1)
    # 基準線
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=adx_threshold, y1=adx_threshold,
                  line=dict(color="orange", width=1, dash="dash"), row=2, col=1)
    
    # スタイル調整
    fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_rangeslider_visible=False,
                      paper_bgcolor='rgba(0,0,0,0)', # 背景透明
                      plot_bgcolor='rgba(240,240,240,0.5)',
                      hovermode='x unified')
    
    # Y軸設定
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="ADX", row=2, col=1)

    return fig

# -------------------------------------------
# 4. メイン表示切り替え (タブ)
# -------------------------------------------
tab1, tab2 = st.tabs(["📊 チャート詳細分析", "🚀 全銘柄スクリーナー"])

# --- タブ1: 個別チャート ---
with tab1:
    st.subheader(f"{selected_ticker} の詳細分析")
    with st.spinner('チャートを描画中...'):
        df_chart = get_stock_data(selected_ticker)
        
        if df_chart is not None:
            # 最新ステータス表示
            last = df_chart.iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("現在株価", f"{last['Close']:.2f}")
            col2.metric("ADX (勢い)", f"{last['ADX']:.1f}", delta="強い" if last['ADX'] > adx_threshold else "弱い")
            
            if last['Trend'] == 1:
                col3.success("🟢 保有中 (HOLD)")
                col4.metric("決済ライン", f"{last['StopLine']:.2f}")
            else:
                col3.info("⚪ 様子見 (WAIT)")
                col4.write("エントリー待ち")

            # プロ級チャート表示
            fig = plot_beautiful_chart(df_chart, selected_ticker)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("データ取得エラー")

# --- タブ2: スクリーナー ---
with tab2:
    if st.button('全銘柄を一括スキャン開始 🔎', key='scan_btn'):
        results = []
        bar = st.progress(0)
        
        for i, t in enumerate(tickers_list):
            d = get_stock_data(t)
            if d is not None:
                last = d.iloc[-1]
                prev = d.iloc[-2]
                
                status = "⚪ WAIT"
                if last['Trend'] == 1 and prev['Trend'] == -1: status = "🚀 BUY NOW"
                elif last['Trend'] == 1: status = "🟢 HOLD"
                
                results.append({
                    "銘柄": t,
                    "株価": last['Close'],
                    "シグナル": status,
                    "ADX": f"{last['ADX']:.1f}",
                    "_sort": 2 if "BUY" in status else (1 if "HOLD" in status else 0)
                })
            bar.progress((i+1)/len(tickers_list))
        
        bar.empty()
        
        if results:
            res_df = pd.DataFrame(results).sort_values(by='_sort', ascending=False).drop(columns=['_sort'])
            
            # 色付け関数
            def color_signal(val):
                color = 'white'
                if 'BUY' in val: color = '#ffcccc'
                elif 'HOLD' in val: color = '#ccffcc'
                return f'background-color: {color}'

            st.dataframe(res_df.style.map(color_signal, subset=['シグナル']), use_container_width=True, height=500)
