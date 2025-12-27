import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# -------------------------------------------
# 1. アプリ設定 & CSSデザイン
# -------------------------------------------
st.set_page_config(page_title="Market Eagle Pro", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    /* 全体フォント */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', 'Hiragino Sans', sans-serif;
    }
    /* メトリックカード */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    /* ギャラリーモードのカード */
    .stock-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    /* バッジ */
    .badge-buy {
        background-color: #ff4b4b; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.9em;
    }
    .badge-hold {
        background-color: #00e396; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.9em;
    }
    .badge-wait {
        background-color: #e0e0e0; color: #555; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------
# 2. 銘柄辞書 & 設定
# -------------------------------------------
TICKER_DICT = {
    "NVDA | NVIDIA": "NVDA", "AAPL | Apple": "AAPL", "MSFT | Microsoft": "MSFT",
    "TSLA | Tesla": "TSLA", "AMZN | Amazon": "AMZN", "GOOGL | Google": "GOOGL",
    "META | Meta": "META", "AMD | AMD": "AMD", "PLTR | Palantir": "PLTR",
    "7203.T | トヨタ": "7203.T", "9984.T | SBG": "9984.T", "8035.T | 東エレ": "8035.T",
    "6146.T | ディスコ": "6146.T", "6920.T | レーザー": "6920.T", "6758.T | ソニー": "6758.T",
    "BTC-USD | Bitcoin": "BTC-USD", "ETH-USD | Ethereum": "ETH-USD"
}

# -------------------------------------------
# 3. サイドバー
# -------------------------------------------
with st.sidebar:
    st.title("🦅 Market Eagle Pro")
    
    # モード切替
    app_mode = st.radio("表示モード", ["🔍 個別詳細分析", "🖼️ チャート一覧 (ギャラリー)"])
    
    st.divider()

    # 個別分析用
    if app_mode == "🔍 個別詳細分析":
        st.subheader("銘柄選択")
        selected_label = st.selectbox("銘柄", list(TICKER_DICT.keys()), index=0)
        current_ticker = TICKER_DICT.get(selected_label, selected_label)
        
        st.subheader("期間設定")
        chart_period = st.select_slider("期間", options=["3mo", "6mo", "1y", "2y", "5y"], value="1y")

    # ギャラリー用
    else:
        st.subheader("リスト設定")
        preset_list = st.selectbox("監視リスト", ["米国ハイテク", "日本株", "暗号資産", "カスタム"])
        
        if preset_list == "米国ハイテク":
            default_tickers = "NVDA, AAPL, MSFT, TSLA, AMZN, GOOGL, META, AMD"
        elif preset_list == "日本株":
            default_tickers = "7203.T, 9984.T, 8035.T, 6146.T, 6920.T, 6758.T"
        elif preset_list == "暗号資産":
            default_tickers = "BTC-USD, ETH-USD, SOL-USD"
        else:
            default_tickers = "NVDA, 7203.T"
            
        ticker_input = st.text_area("コード (カンマ区切り)", default_tickers)
        gallery_tickers = [t.strip() for t in ticker_input.split(',')]
        
        gallery_period = st.select_slider("一覧の期間", options=["6mo", "1y"], value="1y")

    # ロジック設定 (共通)
    with st.expander("⚙️ ロジック詳細設定"):
        bb_period = st.number_input("BB期間", value=20)
        adx_th = st.number_input("ADX基準", value=25)
        atr_p = st.number_input("ATR期間", value=22)
        atr_m = st.number_input("ATR倍率", value=3.5)

# -------------------------------------------
# 4. 計算エンジン
# -------------------------------------------
@st.cache_data(ttl=3600)
def get_analyzed_data(ticker, period):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 30: return None

        # BB
        df['SMA'] = df['Close'].rolling(bb_period).mean()
        df['STD'] = df['Close'].rolling(bb_period).std()
        df['BB_Upper'] = df['SMA'] + (2.0 * df['STD'])
        df['BB_Lower'] = df['SMA'] - (2.0 * df['STD'])

        # ADX
        adx_len = 14
        df['TR'] = np.maximum((df['High'] - df['Low']), 
                   np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                              abs(df['Low'] - df['Close'].shift(1))))
        df['+DM'] = np.where((df['High']-df['High'].shift(1)) > (df['Low'].shift(1)-df['Low']), 
                             np.maximum(df['High']-df['High'].shift(1), 0), 0)
        df['-DM'] = np.where((df['Low'].shift(1)-df['Low']) > (df['High']-df['High'].shift(1)), 
                             np.maximum(df['Low'].shift(1)-df['Low'], 0), 0)
        
        # Smooth
        df['TR_s'] = df['TR'].rolling(adx_len).mean()
        df['+DM_s'] = df['+DM'].rolling(adx_len).mean()
        df['-DM_s'] = df['-DM'].rolling(adx_len).mean()
        df['DX'] = 100 * abs((df['+DM_s']/df['TR_s']) - (df['-DM_s']/df['TR_s'])) / \
                   ((df['+DM_s']/df['TR_s']) + (df['-DM_s']/df['TR_s']))
        df['ADX'] = df['DX'].rolling(adx_len).mean()

        # ATR Exit
        df['ATR'] = df['TR'].rolling(atr_p).mean()
        df['High_Roll'] = df['High'].rolling(atr_p).max()

        # Logic Loop
        trend = np.zeros(len(df))
        stop_line = np.zeros(len(df))
        buy_sig = [np.nan] * len(df)
        sell_sig = [np.nan] * len(df)
        
        curr_trend = -1
        curr_stop = 0.0
        first_idx = max(bb_period, adx_len, atr_p)

        for i in range(first_idx, len(df)):
            close = df['Close'].iloc[i]
            long_stop = df['High_Roll'].iloc[i] - (df['ATR'].iloc[i] * atr_m)
            
            if curr_trend == 1: # Hold
                curr_stop = max(long_stop, curr_stop)
                if close < curr_stop:
                    curr_trend = -1
                    sell_sig[i] = close
                else:
                    trend[i] = 1
                    stop_line[i] = curr_stop
            else: # Wait
                curr_stop = long_stop
                if (close > df['BB_Upper'].iloc[i]) and (df['ADX'].iloc[i] > adx_th):
                    curr_trend = 1
                    buy_sig[i] = close
                    trend[i] = 1
                    stop_line[i] = long_stop
                else:
                    stop_line[i] = long_stop

        df['Trend'] = trend
        df['StopLine'] = stop_line
        df['Buy'] = buy_sig
        df['Sell'] = sell_sig
        
        return df
    except:
        return None

# -------------------------------------------
# 5. チャート描画関数 (ハイライト強化版)
# -------------------------------------------
def plot_enhanced_chart(df, ticker, minimal=False):
    # レイアウト: ミニマルモードならADXなし
    rows = 1 if minimal else 2
    row_heights = [1.0] if minimal else [0.8, 0.2]
    
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=row_heights)

    # 1. ローソク足
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price', showlegend=not minimal
    ), row=1, col=1)

    # 2. BB Cloud & Highlight
    if not minimal:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], fill=None, line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], fill='tonexty', fillcolor='rgba(0,100,255,0.05)', 
                                 line=dict(width=0), showlegend=False), row=1, col=1)
        
        # ハイライト
        hi_adx = df[df['ADX'] > adx_th]
        fig.add_trace(go.Scatter(x=hi_adx.index, y=hi_adx['BB_Upper'], mode='markers', 
                                 marker=dict(color='#FFAA00', size=2), name='Energy Zone'), row=1, col=1)

    # 3. ATR Stop (保有ライン)
    holding = df[df['Trend'] == 1]
    fig.add_trace(go.Scatter(
        x=holding.index, y=holding['StopLine'], mode='markers',
        marker=dict(color='#00E396', size=3 if minimal else 5), name='Hold Line', showlegend=not minimal
    ), row=1, col=1)

    # 4. ★ シグナル強調 (巨大化 & 垂直ライン)
    buys = df.dropna(subset=['Buy'])
    sells = df.dropna(subset=['Sell'])

    # BUYサイン
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys['Buy'], mode='markers',
            marker=dict(symbol='triangle-up', color='#FF0000', size=20 if not minimal else 12, line=dict(width=2, color='white')),
            name='BUY'
        ), row=1, col=1)
        # 垂直ライン (詳細モードのみ)
        if not minimal:
            for d in buys.index:
                fig.add_vline(x=d, line_width=1, line_dash="dash", line_color="red", opacity=0.3)

    # SELLサイン
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells['Sell'], mode='markers',
            marker=dict(symbol='triangle-down', color='#0000FF', size=20 if not minimal else 12, line=dict(width=2, color='white')),
            name='SELL'
        ), row=1, col=1)

    # 5. ADX (詳細モードのみ)
    if not minimal:
        fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], line=dict(color='#775DD0', width=2), name='ADX'), row=2, col=1)
        fig.add_hline(y=adx_th, line_dash="dash", line_color="orange", row=2, col=1)

    # レイアウト調整
    fig.update_layout(
        height=300 if minimal else 600,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='white',
        xaxis_rangeslider_visible=False,
        showlegend=not minimal,
        hovermode='x unified'
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')

    return fig

# -------------------------------------------
# 6. メイン表示ロジック
# -------------------------------------------

# === モードA: 個別詳細分析 ===
if app_mode == "🔍 個別詳細分析":
    st.title(f"🦅 {selected_label.split('|')[0]} 詳細分析")
    
    with st.spinner('Calculating...'):
        df = get_analyzed_data(current_ticker, chart_period)
    
    if df is not None:
        last = df.iloc[-1]
        
        # --- シグナル発生日時の特定 ---
        # 最後に Buy または Sell が出た日を探す
        last_buy_date = df['Buy'].last_valid_index()
        days_since_buy = (df.index[-1] - last_buy_date).days if last_buy_date else None
        
        # 3カラム表示
        c1, c2, c3 = st.columns(3)
        
        with c1:
            # 株価表示
            chg = last['Close'] - df.iloc[-2]['Close']
            st.metric("現在株価", f"{last['Close']:,.2f}", f"{chg:+.2f}")
            
        with c2:
            # シグナル状態
            if last['Trend'] == 1:
                st.markdown('<div class="badge-hold" style="text-align:center; padding:10px; font-size:20px;">🟢 HOLD (保有中)</div>', unsafe_allow_html=True)
                st.metric("決済ライン (逆指値)", f"{last['StopLine']:,.2f}")
            else:
                st.markdown('<div class="badge-wait" style="text-align:center; padding:10px; font-size:20px;">⚪ WAIT (様子見)</div>', unsafe_allow_html=True)
                
        with c3:
            # ★ 発生日の明記
            if last_buy_date:
                date_str = last_buy_date.strftime('%Y/%m/%d')
                st.metric("直近のBUYサイン点灯", f"{date_str}", f"{days_since_buy}日前")
            else:
                st.metric("直近のBUYサイン", "なし (期間内)")

        # チャート表示
        st.plotly_chart(plot_enhanced_chart(df, current_ticker), use_container_width=True)
        
        with st.expander("データ詳細"):
            st.dataframe(df.tail(10))
    else:
        st.error("データが取得できませんでした")

# === モードB: チャート一覧 (ギャラリー) ===
else:
    st.title("🖼️ チャート・ギャラリー (一覧)")
    st.markdown("設定したリストのチャートを一括表示します。")
    
    if st.button("一覧を更新 🔄"):
        cols = st.columns(2) # 2列で表示
        
        for i, ticker in enumerate(gallery_tickers):
            with cols[i % 2]: # 列を交互に使う
                df = get_analyzed_data(ticker, gallery_period)
                
                if df is not None:
                    last = df.iloc[-1]
                    # バッジ決定
                    if last['Trend'] == 1:
                        badge = '<span class="badge-hold">🟢 HOLD</span>'
                    else:
                        badge = '<span class="badge-wait">⚪ WAIT</span>'
                    
                    # 発生日
                    last_buy = df['Buy'].last_valid_index()
                    buy_info = f"BUY点灯: {last_buy.strftime('%m/%d')}" if last_buy else "BUYなし"
                    
                    # カード表示
                    st.markdown(f"""
                    <div class="stock-card">
                        <h3>{ticker} {badge}</h3>
                        <p style="font-size:1.2em; font-weight:bold;">${last['Close']:.2f}</p>
                        <p style="color:gray;">{buy_info}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # ミニマルチャート
                    st.plotly_chart(plot_enhanced_chart(df, ticker, minimal=True), use_container_width=True)
                else:
                    st.warning(f"{ticker}: 取得エラー")
