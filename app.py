import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------------------
# 1. モダンUI設定 & CSSデザイン注入
# -------------------------------------------
st.set_page_config(page_title="Market Eagle 🦅", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', 'Hiragino Sans', sans-serif;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    div.stButton > button {
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------
# 2. 予測変換用の銘柄辞書
# -------------------------------------------
TICKER_DICT = {
    # --- 米国株 ---
    "NVDA | NVIDIA": "NVDA",
    "AAPL | Apple": "AAPL",
    "MSFT | Microsoft": "MSFT",
    "TSLA | Tesla": "TSLA",
    "AMZN | Amazon": "AMZN",
    "GOOGL | Google": "GOOGL",
    "META | Meta": "META",
    "AMD | AMD": "AMD",
    "PLTR | Palantir": "PLTR",
    "COIN | Coinbase": "COIN",
    # --- 日本株 ---
    "7203.T | トヨタ自動車": "7203.T",
    "9984.T | ソフトバンクG": "9984.T",
    "8035.T | 東京エレクトロン": "8035.T",
    "6146.T | ディスコ": "6146.T",
    "6920.T | レーザーテック": "6920.T",
    "6758.T | ソニーG": "6758.T",
    "8306.T | 三菱UFJ": "8306.T",
    "9983.T | ファーストリテイリング": "9983.T",
    "7974.T | 任天堂": "7974.T",
    "7011.T | 三菱重工": "7011.T",
    # --- 暗号資産 ---
    "BTC-USD | Bitcoin": "BTC-USD",
    "ETH-USD | Ethereum": "ETH-USD",
    "SOL-USD | Solana": "SOL-USD",
    "XRP-USD | XRP": "XRP-USD",
}

# -------------------------------------------
# 3. サイドバー設定
# -------------------------------------------
with st.sidebar:
    st.title("🦅 Market Eagle")
    st.caption("AI Hybrid Strategy: BB+ADX x ATR")
    
    st.divider()
    
    st.subheader("🔍 銘柄検索")
    selected_label = st.selectbox(
        "銘柄を選択または入力",
        options=list(TICKER_DICT.keys()),
        index=0
    )
    current_ticker = TICKER_DICT.get(selected_label, selected_label)

    st.subheader("📅 チャート期間")
    chart_period = st.select_slider(
        "表示期間",
        options=["3mo", "6mo", "1y", "2y", "5y"],
        value="1y"
    )

    with st.expander("⚙️ ロジック詳細設定"):
        bb_period = st.number_input("BB期間", value=20)
        adx_threshold = st.number_input("ADX基準値", value=25)
        atr_period = st.number_input("ATR期間", value=22)
        atr_multiplier = st.number_input("ATR倍率", value=3.5)

# -------------------------------------------
# 4. データ処理関数
# -------------------------------------------
@st.cache_data(ttl=3600)
def get_data(ticker, period):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty or len(df) < 20: return None
        
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
        
        # 簡易ADX計算
        df['+DI'] = 100 * (df['+DM'].rolling(adx_len).mean() / df['TR'].rolling(adx_len).mean())
        df['-DI'] = 100 * (df['-DM'].rolling(adx_len).mean() / df['TR'].rolling(adx_len).mean())
        df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
        df['ADX'] = df['DX'].rolling(adx_len).mean()

        # ATR Exit
        df['ATR'] = df['TR'].rolling(atr_period).mean()
        df['High_Roll'] = df['High'].rolling(atr_period).max()

        # シグナル判定
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
                    curr_trend = 1
                    buy_sig[i] = close
                    stop_line[i] = long_stop
                    trend[i] = 1
                else:
                    stop_line[i] = long_stop

        df['StopLine'] = stop_line
        df['Trend'] = trend
        df['Buy'] = buy_sig
        df['Sell'] = sell_sig
        
        return df

    except Exception:
        return None

# -------------------------------------------
# 5. モダンチャート描画 (エラー回避版)
# -------------------------------------------
def plot_modern_chart(df, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.8, 0.2],
                        subplot_titles=("", ""))

    # 1. ローソク足
    # ★ ここが修正箇所です。nameのみ指定し、余計なパラメータは削除しました。
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price'
    ), row=1, col=1)

    # 2. BB Cloud
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BB_Upper'], line=dict(width=0), showlegend=False, hoverinfo='skip'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BB_Lower'], fill='tonexty', fillcolor='rgba(0, 100, 255, 0.05)',
        line=dict(width=0), showlegend=False, name='BB Cloud', hoverinfo='skip'
    ), row=1, col=1)

    # 3. BB Highlight
    high_adx = df[df['ADX'] > adx_threshold]
    fig.add_trace(go.Scatter(
        x=high_adx.index, y=high_adx['BB_Upper'], mode='markers',
        marker=dict(size=3, color='#FFAA00'), name='High Energy Zone'
    ), row=1, col=1)

    # 4. ATR Stop
    holding = df[df['Trend'] == 1]
    fig.add_trace(go.Scatter(
        x=holding.index, y=holding['StopLine'], mode='markers',
        marker=dict(size=4, color='#00E396'), name='ATR Stop'
    ), row=1, col=1)

    # 5. Signals
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Buy'], mode='markers',
        marker=dict(symbol='triangle-up', color='#FF4560', size=12, line=dict(width=1, color='white')),
        name='BUY'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Sell'], mode='markers',
        marker=dict(symbol='triangle-down', color='#008FFB', size=12, line=dict(width=1, color='white')),
        name='SELL'
    ), row=1, col=1)

    # 6. ADX
    fig.add_trace(go.Scatter(
        x=df.index, y=df['ADX'], line=dict(color='#775DD0', width=2), name='ADX'
    ), row=2, col=1)
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=adx_threshold, y1=adx_threshold,
                  line=dict(color="#FFAA00", width=1, dash="dash"), row=2, col=1)

    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')

    return fig

# -------------------------------------------
# 6. メインコンテンツ
# -------------------------------------------
st.title(f"📊 Analysis: {selected_label.split('|')[0]}")

with st.spinner('Fetching data...'):
    df = get_data(current_ticker, chart_period)

if df is not None:
    last = df.iloc[-1]
    prev = df.iloc[-2]
    change = last['Close'] - prev['Close']
    pct_change = (change / prev['Close']) * 100
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric("株価", f"{last['Close']:,.2f}", f"{pct_change:+.2f}%")
    
    with c2:
        trend_status = "HOLD (保有中)" if last['Trend'] == 1 else "WAIT (様子見)"
        color = "#00E396" if last['Trend'] == 1 else "#FEB019"
        st.markdown(f"""
        <div style="background-color:{color}; padding:10px; border-radius:5px; text-align:center; color:white; font-weight:bold;">
            {trend_status}
        </div>
        """, unsafe_allow_html=True)

    with c3:
        if last['Trend'] == 1:
            st.metric("決済ライン (逆指値)", f"{last['StopLine']:,.2f}", delta_color="inverse")
        else:
            dist_to_bb = last['BB_Upper'] - last['Close']
            st.metric("ブレイクまであと", f"{dist_to_bb:+.2f}")

    st.plotly_chart(plot_modern_chart(df, current_ticker), use_container_width=True)

    with st.expander("📄 詳細データを見る"):
        st.dataframe(df[['Close', 'BB_Upper', 'ADX', 'Trend', 'StopLine']].tail(10).style.format("{:.2f}"))

else:
    st.error("データが見つかりませんでした。銘柄コードを確認してください。")
