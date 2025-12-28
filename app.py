import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# -------------------------------------------
# 1. 設定
# -------------------------------------------
st.set_page_config(page_title="Market Eagle Screener", layout="wide", page_icon="🦅")

st.title("🦅 Market Eagle: 全銘柄一括スクリーナー")
st.markdown("BB + ADX + ATR戦略で、チャンスのある銘柄をリスト化します。")

# サイドバー設定
with st.sidebar:
    st.header("⚙️ 検索設定")
    
    # プリセットリスト
    preset = st.selectbox("銘柄リスト選択", ["米国ハイテク", "日本株(主力)", "暗号資産", "カスタム"])
    
    if preset == "米国ハイテク":
        default_tickers = "NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, AMD, AVGO, TSM, PLTR, COIN, MSTR, SMCI, ARM"
    elif preset == "日本株(主力)":
        default_tickers = "7203.T, 9984.T, 8035.T, 6146.T, 6920.T, 6758.T, 8306.T, 9983.T, 6857.T, 6501.T, 7011.T, 7735.T, 4063.T, 4502.T, 9432.T"
    elif preset == "暗号資産":
        default_tickers = "BTC-USD, ETH-USD, SOL-USD, XRP-USD, DOGE-USD, ADA-USD, BNB-USD"
    else:
        default_tickers = "NVDA, 7203.T"
        
    tickers_input = st.text_area("銘柄コード (カンマ区切り)", default_tickers, height=150)
    tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]

    st.divider()
    
    # パラメータ
    with st.expander("ロジック詳細設定"):
        bb_period = st.number_input("BB期間", 20)
        adx_th = st.number_input("ADX基準", 25)
        atr_period = st.number_input("ATR期間", 22)
        atr_mult = st.number_input("ATR倍率", 3.5)

# -------------------------------------------
# 2. 分析エンジン
# -------------------------------------------
@st.cache_data(ttl=600) # 10分キャッシュ
def analyze_ticker(ticker):
    try:
        # データ取得 (期間は長めに)
        df = yf.download(ticker, period="1y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 50: return None

        # --- 計算 ---
        # BB
        df['SMA'] = df['Close'].rolling(bb_period).mean()
        df['STD'] = df['Close'].rolling(bb_period).std()
        df['BB_Upper'] = df['SMA'] + (2.0 * df['STD'])

        # ADX
        adx_len = 14
        df['TR'] = np.maximum((df['High'] - df['Low']), 
                   np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                              abs(df['Low'] - df['Close'].shift(1))))
        df['+DM'] = np.where((df['High']-df['High'].shift(1)) > (df['Low'].shift(1)-df['Low']), 
                             np.maximum(df['High']-df['High'].shift(1), 0), 0)
        df['-DM'] = np.where((df['Low'].shift(1)-df['Low']) > (df['High']-df['High'].shift(1)), 
                             np.maximum(df['Low'].shift(1)-df['Low'], 0), 0)
        
        df['TR_s'] = df['TR'].rolling(adx_len).mean()
        df['+DM_s'] = df['+DM'].rolling(adx_len).mean()
        df['-DM_s'] = df['-DM'].rolling(adx_len).mean()
        df['DX'] = 100 * abs((df['+DM_s']/df['TR_s']) - (df['-DM_s']/df['TR_s'])) / \
                   ((df['+DM_s']/df['TR_s']) + (df['-DM_s']/df['TR_s']))
        df['ADX'] = df['DX'].rolling(adx_len).mean()

        # ATR
        df['ATR'] = df['TR'].rolling(atr_period).mean()
        df['High_Roll'] = df['High'].rolling(atr_period).max()

        # --- トレンド判定ループ ---
        trend = 0 # 1:Hold, -1:Wait
        stop_line = 0.0
        curr_trend = -1
        curr_stop = 0.0
        
        first_idx = max(bb_period, adx_len, atr_period)
        
        # 最終的な状態を知るために全期間回す
        for i in range(first_idx, len(df)):
            close = df['Close'].iloc[i]
            long_stop = df['High_Roll'].iloc[i] - (df['ATR'].iloc[i] * atr_mult)
            
            if curr_trend == 1: # Hold
                curr_stop = max(long_stop, curr_stop)
                if close < curr_stop:
                    curr_trend = -1
                else:
                    pass # Keep Hold
            else: # Wait
                curr_stop = long_stop
                if (close > df['BB_Upper'].iloc[i]) and (df['ADX'].iloc[i] > adx_th):
                    curr_trend = 1

        # 結果整理
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        status = "WAIT"
        sort_rank = 0
        signal_msg = "-"
        
        # シグナル判定
        # 前日はWaitだったが、今日はHoldになった = 買いシグナル
        if curr_trend == 1 and (df['Close'].iloc[-2] < (df['High_Roll'].iloc[-2] - (df['ATR'].iloc[-2] * atr_mult))): 
            # 簡易判定: 本来は過去のTrend配列を持つべきだが、軽量化のため「今日BBブレイクしてるか」を見る
             if (last['Close'] > last['BB_Upper']) and (last['ADX'] > adx_th):
                status = "🚀 BUY NOW"
                sort_rank = 2
        elif curr_trend == 1:
            status = "🟢 HOLD"
            sort_rank = 1
        
        return {
            "銘柄": ticker,
            "現在株価": last['Close'],
            "シグナル": status,
            "逆指値(逃げ)": curr_stop if curr_trend == 1 else None,
            "ADX(勢い)": last['ADX'],
            "BB乖離": last['Close'] - last['BB_Upper'],
            "_rank": sort_rank
        }

    except Exception as e:
        return None

# -------------------------------------------
# 3. メイン画面
# -------------------------------------------
if st.button("銘柄スキャン開始 (実行)"):
    results = []
    bar = st.progress(0)
    
    for i, t in enumerate(tickers):
        data = analyze_ticker(t)
        if data:
            results.append(data)
        bar.progress((i+1)/len(tickers))
    
    bar.empty()
    
    if results:
        df_res = pd.DataFrame(results)
        
        # 並び替え: BUY > HOLD > WAIT
        df_res = df_res.sort_values(by="_rank", ascending=False).drop(columns=["_rank"])
        
        # 表示用フォーマット
        st.subheader(f"📊 診断結果 ({len(results)}銘柄)")
        
        # スタイル適用
        def style_df(val):
            if val == "🚀 BUY NOW":
                return 'background-color: #ff4b4b; color: white; font-weight: bold;'
            elif val == "🟢 HOLD":
                return 'background-color: #d4edda; color: green; font-weight: bold;'
            return ''

        # データフレーム表示
        st.dataframe(
            df_res.style.map(style_df, subset=['シグナル'])
                  .format({"現在株価": "{:.2f}", "逆指値(逃げ)": "{:.2f}", "ADX(勢い)": "{:.1f}", "BB乖離": "{:+.2f}"}),
            use_container_width=True,
            height=600
        )
    else:
        st.warning("データが取得できませんでした。")
