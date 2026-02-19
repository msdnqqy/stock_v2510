import pandas as pd
import numpy as np
from utils import *

class StrongReversalConfig:
    # --- 1. 趋势与位置 ---
    # 改为：抓上升趋势中的回调
    MA_WINDOW = 5              # 均线周期 (对应图中的蓝线)
    PULLBACK_WINDOW = 3         # 至少连续 N 天并没有创新高 (定义回调)
    SUPPORT_TOLERANCE = 0.03    # 股价回踩到均线 %3 以内的范围
    
    # --- 2. 核心：力度过滤器 (针对黄框的特征) ---
    MIN_BODY_RATIO = 1.5        # 阳线实体必须是过去5天平均实体的 1.5 倍以上 (大阳线)
    MIN_VOLUME_RATIO = 2.0      # 成交量必须是过去10天均量的 2.0 倍以上 (放量)
    
    # --- 3. 形态微调 ---
    MORNING_STAR_GAP = False    # 不需要严格缺口 (A股等市场跳空少，设为False更实用)

def detect_strong_pullback_reversal(df: pd.DataFrame, config=StrongReversalConfig) -> pd.DataFrame:
    df = df.copy()
    O, H, L, C, V = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
    
    # === 1. 基础指标计算 ===
    # 均线 (Trend)
    ma_trend = C.rolling(window=config.MA_WINDOW).mean()
    # 均量 (Volume MA)
    vol_ma = V.rolling(window=10).mean()
    # 实体大小 (Body Size)
    body_len = np.abs(C - O)
    avg_body = body_len.rolling(window=5).mean()
    
    # === 2. 识别“回调”环境 (Context) ===
    # 逻辑：均线本身是向上的 (MA_slope > 0)，但股价最近在跌或横盘
    ma_slope = ma_trend > ma_trend.shift(1) # 长期趋势向上
    
    # 股价接近均线 (Support): 最低价没有跌破均线太远，且离均线很近
    # 判定：Low >= MA * 0.97  AND  Low <= MA * 1.05 (在均线附近)
    dist_to_ma = np.abs(L - ma_trend) / ma_trend
    is_at_support = (dist_to_ma <= config.SUPPORT_TOLERANCE)
    
    # === 3. 识别“强力反转信号” (Trigger) ===
    
    # A. 必须是大阳线 (Big Bullish Candle) -> 对应黄框第三根
    is_big_bull = (C > O) & (body_len > avg_body.shift(1) * config.MIN_BODY_RATIO)
    
    # B. 必须是放量 (High Volume) -> 对应黄框下方的高成交量
    # 逻辑：当天的量 > 均量 * 倍数
    is_explode_vol = V > (vol_ma.shift(1) * config.MIN_VOLUME_RATIO)
    
    # C. 早晨之星 / 强力吞没 逻辑修正
    # Day -2: 阴线 (回调)
    # Day -1: 小星线 (企稳，实体很小)
    # Day 0: 大阳线 + 放量
    
    # Day -1 是小实体 (Star)
    prev_body = body_len.shift(1)
    is_star = prev_body < avg_body.shift(1) * 0.6
    
    # Day -2 是阴线
    is_prev2_bear = C.shift(2) < O.shift(2)
    
    # 组合形态：早晨之星变体
    is_morning_star_strong = (
        is_prev2_bear &
        is_star &
        is_big_bull &     # 这一天必须够强
        is_explode_vol    # 这一天必须放量
    )
    
    # 组合形态：强力吞没 (Engulfing)
    # 昨天跌，今天涨，且今天实体巨大，且放量
    is_engulfing_strong = (
        (C.shift(1) < O.shift(1)) & # 昨天阴
        is_big_bull &               # 今天大阳
        (C > O.shift(1)) &          # 收盘价越过昨天开盘
        is_explode_vol              # 必须放量
    )

    # === 4. 最终信号汇总 ===
    # 只有在“均线向上”且“回踩均线支撑”的背景下，出现“强反转”，才算数
    
    valid_signal = (
        ma_slope &              # 大趋势向上
        is_at_support &         # 回踩到了均线
        (is_morning_star_strong | is_engulfing_strong) # 出现了强力K线组合
    )
    
    results = pd.DataFrame(index=df.index)
    results['Close'] = C
    results['MA20'] = ma_trend
    results['Volume_Ratio'] = V / vol_ma.shift(1) # 方便查看倍数
    results['Is_Support_Touch'] = is_at_support
    results['Signal_Strong_Reversal'] = valid_signal

    return results

# --- 为了演示，你可以直接看 Volume_Ratio 和 Signal_Strong_Reversal 列 ---

# --- 使用示例 ---
if __name__ == "__main__":
    # 模拟数据生成 (包含 Volume)
    dates = pd.date_range(start='2024-01-01', periods=100)
    data = {
        'Open': np.random.uniform(10, 20, 100),
        'High': np.random.uniform(10, 20, 100),
        'Low': np.random.uniform(10, 20, 100),
        'Close': np.random.uniform(10, 20, 100),
        'Volume': np.random.randint(1000, 50000, 100)
    }
    # 简单修正 High/Low 逻辑
    df = get_data()
    # df = pd.DataFrame(data, index=dates)
    df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
    df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
    
    # 运行分析

    # 运行策略
    results = detect_strong_pullback_reversal(df)
    print(results)
    
    # 筛选出所有出现信号的日期
    signals = results[results['Signal_Strong_Reversal'] == True]
    
    if not signals.empty:
        print(f"✅ {ticker_symbol} 发现信号！日期如下：")
        # 打印日期、收盘价、量比
        print(signals[['Close', 'Volume_Ratio', 'MA20']])
    else:
        print(f"❌ {ticker_symbol} 在过去一年未发现符合该严苛条件的信号。")