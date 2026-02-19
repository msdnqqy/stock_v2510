import pandas as pd
import numpy as np
from utils import *
class PatternConfig:
    """策略参数配置，方便后续进行参数调优"""
    # 趋势定义
    TREND_MA_WINDOW = 5       # 使用20日均线判断趋势
    TREND_LOOKBACK = 3          # 要求过去N天主要处于均线下方
    
    # 形态阈值
    HAMMER_SHADOW_RATIO = 2.0   # 锤子线：下影线是实体的多少倍
    HAMMER_UPPER_LIMIT = 0.5    # 锤子线：上影线最大长度限制
    
    ENGULFING_RATIO = 1.0       # 吞没形态：阳线实体至少要包裹阴线实体的比例
    
    # 辅助确认
    VOL_MA_WINDOW = 10          # 量能均线周期
    VOL_MULTIPLIER = 1.2        # 放量倍数：当前成交量 > 1.2 * 10日均量

def analyze_stock_patterns(df: pd.DataFrame, config=PatternConfig) -> pd.DataFrame:
    """
    输入: 包含 Open, High, Low, Close, Volume 的 DataFrame
    输出: 标记了形态信号的 DataFrame
    """
    # 1. 预计算基础指标 (利用向量化计算，速度极快)
    # -------------------------------------------------------
    df = df.copy() # 避免修改原始数据
    O, H, L, C, V = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
    
    # 实体与影线
    body_len = np.abs(C - O)
    body_top = np.maximum(C, O)
    body_bottom = np.minimum(C, O)
    
    upper_shadow = H - body_top
    lower_shadow = body_bottom - L
    
    # 均线与均量
    ma_trend = C.rolling(window=config.TREND_MA_WINDOW).mean()
    vol_ma = V.rolling(window=config.VOL_MA_WINDOW).mean()
    
    # 平均实体大小 (用于判断星星或十字星)
    avg_body = body_len.rolling(window=10).mean()

    # 2. 核心：趋势过滤器 (Trend Filter)
    # -------------------------------------------------------
    # 逻辑：不仅仅看今天，要求过去 TREND_LOOKBACK 天里，收盘价主要在均线下方
    # 这里使用 rolling sum 来判断连续性
    is_below_ma = (C < ma_trend).astype(int)
    # 比如：过去3天里至少有3天在均线下方 (严格下降趋势)
    trend_check = is_below_ma.rolling(window=config.TREND_LOOKBACK).sum() >= config.TREND_LOOKBACK
    
    # 或者：判断均线本身的斜率是向下的 (MA今天 < MA昨天)
    ma_slope_down = ma_trend < ma_trend.shift(1)
    
    # === 最终趋势条件：股价在均线下方 AND 均线向下 ===
    is_downtrend = trend_check & ma_slope_down

    # 3. 核心：成交量确认 (Volume Filter)
    # -------------------------------------------------------
    # 逻辑：反转日或者是反转形态的关键日，必须放量
    is_high_volume = V > (vol_ma.shift(1) * config.VOL_MULTIPLIER)

    # 4. 形态识别 (Pattern Recognition)
    # -------------------------------------------------------
    
    # --- A. 锤子线 (Hammer) ---
    # 定义：跌势中 + 下影线长 + 实体小 + 上影线短 + (可选: 放量)
    is_hammer = (
        is_downtrend &
        (lower_shadow >= config.HAMMER_SHADOW_RATIO * body_len) &
        (upper_shadow <= config.HAMMER_UPPER_LIMIT * body_len) &
        (body_len < avg_body * 1.5) # 实体不能太大
        # & is_high_volume # 锤子线是否严格要求放量可选，此处暂不强制，但在结果中返回量能信号
    )

    # --- B. 看涨吞没 (Bullish Engulfing) ---
    # 定义：跌势中 + 前阴后阳 + 阳包阴
    prev_O, prev_C = O.shift(1), C.shift(1)
    prev_body = np.abs(prev_C - prev_O)
    
    is_prev_bear = prev_C < prev_O # 昨天跌
    is_curr_bull = C > O         # 今天涨
    
    is_engulfing = (
        is_downtrend &
        is_prev_bear &
        is_curr_bull &
        (C > prev_O) & # 今天收盘 > 昨天开盘 (顶部覆盖)
        (O < prev_C) & # 今天开盘 < 昨天收盘 (底部覆盖)
        (body_len > prev_body * config.ENGULFING_RATIO) # 实体大小压制
    )

    # --- C. 早晨之星 (Morning Star) ---
    # 定义：Day-2大阴 -> Day-1小星(跳空低开) -> Day-0大阳(回补)
    
    # Day -2: 大阴线
    d2_O, d2_C = O.shift(2), C.shift(2)
    d2_body = np.abs(d2_C - d2_O)
    is_d2_bear = (d2_C < d2_O) & (d2_body > avg_body.shift(2))
    
    # Day -1: 星线 (实体很小)
    d1_body = body_len.shift(1)
    is_d1_star = d1_body < (avg_body.shift(1) * 0.6)
    
    # Day 0: 大阳线 + 刺入 Day-2 实体一半以上
    mid_point_d2 = (d2_O + d2_C) / 2
    is_d0_bull = (C > O) & (C > mid_point_d2)
    
    # 趋势判断：看 Day-2 之前是否处于跌势
    is_morning_star = (
        is_downtrend.shift(2) & 
        is_d2_bear &
        is_d1_star &
        is_d0_bull
    )

    # 5. 结果组装
    # -------------------------------------------------------
    results = pd.DataFrame(index=df.index)
    results['Close'] = C
    results['MA_Trend'] = ma_trend
    
    # 信号列 (Boolean)
    results['Signal_Hammer'] = is_hammer
    results['Signal_Engulfing'] = is_engulfing
    results['Signal_MorningStar'] = is_morning_star
    
    # 辅助列：是否有量能配合 (供后续筛选优先级使用)
    results['Is_High_Volume'] = is_high_volume
    
    # 综合信号：只要满足任一形态，标记为 True
    results['Buy_Signal'] = (is_hammer | is_engulfing | is_morning_star)

    return results



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
    signals = analyze_stock_patterns(df)
    
    # 筛选出有信号的日期
    valid_signals = signals[signals['Buy_Signal']]
    
    print(f"发现 {len(valid_signals)} 个反转信号")
    if not valid_signals.empty:
        # 打印出具体的信号类型和是否放量
        print(valid_signals[['Close', 'Signal_Hammer', 'Signal_Engulfing', 'Signal_MorningStar', 'Is_High_Volume']].tail())