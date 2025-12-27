import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# -------------------------------------------
# 1. アプリ全体の設定
# -------------------------------------------
st.set_page_config(page_title="AI株価スクリーナー", layout="wide")

st.title("🚀 チャンス発見スクリーナー")
st.markdown("最強ロジック (Entry: **BB+ADX** / Exit: **ATR**) で全銘柄を一括診断します。")

# サイドバー設定
st.sidebar.header("⚙️ スクリーニング設定")

# 監視リストのプリセット
preset_list = st.sidebar.selectbox(
    "監視リストを選択",
    ("米国ハイテク (Mag7 + AI)", "日本株 (主力大型)", "暗号資産 (Major)")
)

if preset_list == "米国ハイテク (Mag7 + AI)":
    default_tickers = "NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, AMD, AVGO, TSM"
elif preset_list == "日本株 (主力大型)":
    default_tickers = "7203.T, 9984.T, 8035.T, 6920.T, 6146.T, 6758.T, 8306.T, 9983.T, 6857.T, 6501.T"
else:
    default_tickers = "BTC-USD, ETH-USD, SOL-USD, XRP-USD, DOGE-USD"

ticker_input = st.sidebar.text_area("銘柄コード (カンマ区切り)", default_tickers)
tickers = [t.strip() for t in ticker_input.split(',')]

# パラメータ設定
st.sidebar.subheader("ロジック調整")
bb_period = st.sidebar.number_input("BB期間", value=20)
adx_threshold = st.sidebar.number_input("ADX基準値", value=25)
atr_period = st.sidebar.number_input("ATR期間", value=22)
atr_multiplier = st.sidebar.number_input("ATR倍率", value=3.5)

# -------------------------------------------
# 2. 計算エンジン (1銘柄ごとの診断)
# -------------------------------------------
def analyze_stock(ticker):
    try:
        # 期間は長めに取る（ADX計算のため）
        df = yf.download(ticker, period="1y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty or len(df) < 50:
            return None

        # --- テクニカル計算 ---
        # BB
        df['SMA'] = df['Close'].rolling(window=bb_period).mean()
        df['STD'] = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['SMA'] + (2.0 * df['STD']) # Entryは2σ固定でOK

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
        trend = 0 # 1:Hold, -1:Wait
        stop_line = 0.0
        
        # 高速化のため直近データのみで最終判定
        # (本来は全期間ループが必要だが、現状の状態を知るため簡易シミュレーション)
        # 正確を期すため、全期間回します
        trends = []
        stops = []
        
        curr_trend = -1
        curr_stop = 0.0
        
        first_idx = max(bb_period, adx_len, atr_period)
        
        for i in range(len(df)):
            if i < first_idx:
                trends.append(0)
                stops.append(0)
                continue
                
            close = df['Close'].iloc[i]
            high_roll = df['High_Roll'].iloc[i]
            atr = df['ATR'].iloc[i]
            adx = df['ADX'].iloc[i]
            bb_upper = df['BB_Upper'].iloc[i]
            
            # ATR Stop calculation
            long_stop = high_roll - (atr * atr_multiplier)
            
            if curr_trend == 1: # 保有中
                curr_stop = max(long_stop, curr_stop)
                if close < curr_stop:
                    curr_trend = -1 # 売り転換
                # 維持
            else: # 待機中
                curr_stop = long_stop
                # 買い条件: BBブレイク AND ADX > 基準値
                if (close > bb_upper) and (adx > adx_threshold):
                    curr_trend = 1 # 買い転換
            
            trends.append(curr_trend)
            stops.append(curr_stop)

        # 最新の状態を取得
        last_close = df['Close'].iloc[-1]
        last_trend = trends[-1]
        prev_trend = trends[-2]
        last_stop = stops[-1]
        last_adx = df['ADX'].iloc[-1]

        # ステータス決定
        status = ""
        color = ""
        action = ""
        
        if last_trend == 1 and prev_trend == -1:
            status = "🚀 BUY SIGNAL" # 今日点灯
            color = "background-color: #ffcccc; color: red; font-weight: bold;" # 赤背景
            action = "今すぐエントリー"
        elif last_trend == 1:
            status = "🟢 HOLD"
            color = "background-color: #ccffcc; color: green;" # 緑背景
            action = f"逆指値: {last_stop:.2f}"
        else:
            status = "⚪ WAIT"
            color = ""
            action = "様子見"
            
        return {
            "銘柄": ticker,
            "株価": last_close,
            "シグナル": status,
            "ADX (勢い)": f"{last_adx:.1f}",
            "アクション": action,
            "_raw_signal": 2 if "BUY" in status else (1 if "HOLD" in status else 0), # ソート用
            "_style": color
        }
    except Exception as e:
        return None

# -------------------------------------------
# 3. メイン処理：一覧表示
# -------------------------------------------
if st.button('全銘柄を一括スキャン開始 🔎'):
    results = []
    progress_bar = st.progress(0)
    
    for i, t in enumerate(tickers):
        data = analyze_stock(t)
        if data:
            results.append(data)
        progress_bar.progress((i + 1) / len(tickers))
    
    progress_bar.empty()

    if results:
        # データフレーム化
        df_res = pd.DataFrame(results)
        
        # ソート: BUYシグナル(2) > HOLD(1) > WAIT(0) の順に並べる
        df_res = df_res.sort_values(by='_raw_signal', ascending=False)
        
        # 表示用カラム整理
        display_cols = ["銘柄", "株価", "シグナル", "ADX (勢い)", "アクション"]
        
        # スタイル適用関数
        def style_rows(row):
            return [row['_style']] * len(display_cols)

        # テーブル表示
        st.subheader(f"📊 診断結果 ({len(results)}銘柄)")
        st.write("一番上が最もチャンスのある銘柄です。")
        
        st_df = df_res[display_cols].style.apply(lambda x: df_res['_style'], axis=0, subset=display_cols)
        # シンプルに表示（Streamlitのdataframe機能で色付けは制限があるため、簡易表示）
        # 色付けを確実にするため、独自のHTML生成などはせず、Streamlit標準のdataframeで見やすくします
        
        # 簡易的な色付けロジック
        def highlight_signal(val):
            if "BUY" in val:
                return 'background-color: #ff4b4b; color: white; font-weight: bold;'
            elif "HOLD" in val:
                return 'background-color: #d4edda; color: black;'
            return ''

        st.dataframe(
            df_res[display_cols].style.map(highlight_signal, subset=['シグナル']),
            use_container_width=True,
            height=600
        )
    else:
        st.warning("データが取得できませんでした。")