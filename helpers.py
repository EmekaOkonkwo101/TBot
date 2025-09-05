import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decouple import config


def analyze_multiple_markets(market_data_dict):
    """
    market_data_dict: dict of {market_name: DataFrame}
    Returns: dict of {market_name: {'signals': Series, 'atr': Series}}
    """
    results = {}

    for market, df in market_data_dict.items():
        try:
            signals, atr = atr_breakout_signals(df)
            results[market] = {
                'signals': signals,
                'atr': atr
            }
        except Exception as e:
            print(f"Error processing {market}: {e}")
            results[market] = {
                'signals': None,
                'atr': None
            }

    return results


# def atr_breakout_signals(df, atr_multiplier=1):
#     import numpy as np
#     import pandas as pd

#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = [col[0] for col in df.columns]

#     close = df['Close']
#     high  = df['High']
#     low   = df['Low']

#     ema20 = close.ewm(span=8, adjust=False).mean()
#     ema50 = close.ewm(span=22, adjust=False).mean()

#     prev_close = close.shift(1)
#     tr = pd.concat([
#         high - low,
#         (high - prev_close).abs(),
#         (low - prev_close).abs()
#     ], axis=1).max(axis=1)
#     atr = tr.rolling(3, min_periods=1).mean()

#     # Breakout levels
#     prev_hi = high.shift(1).rolling(20, min_periods=1).max()
#     prev_lo = low.shift(1).rolling(20, min_periods=1).min()

#     trend_up = ema20 > ema50
#     trend_dn = ema20 < ema50

#     candle_range = high - low
#     min_range = atr * atr_multiplier

#     # Breakout Detection
#     long_breakout = (close > prev_hi)
#     short_breakout = (close < prev_lo)

#     breakout_high = np.where(long_breakout, prev_hi, np.nan)
#     breakout_low  = np.where(short_breakout, prev_lo, np.nan)

#     breakout_high = pd.Series(breakout_high, index=df.index).ffill()
#     breakout_low  = pd.Series(breakout_low, index=df.index).ffill()

#     # Retest conditions
#     retest_long = (low <= breakout_high) & (close > breakout_high)
#     retest_short = (high >= breakout_low) & (close < breakout_low)

#     # Final signal logic
#     long_entry = (
#         trend_up &
#         retest_long &
#         (atr > atr.shift(1)) &
#         (candle_range >= min_range)
#     )

#     short_entry = (
#         trend_dn &
#         retest_short &
#         (atr > atr.shift(1)) &
#         (candle_range >= min_range)
#     )

#     signals = pd.Series(index=df.index, dtype=object)
#     signals.loc[long_entry] = "LONG"
#     signals.loc[short_entry] = "SHORT"

#     # TP and SL levels with 6:1 reward-to-risk ratio
#     tp = pd.Series(index=df.index, dtype=float)
#     sl = pd.Series(index=df.index, dtype=float)

#     rr_ratio = 3  # reward-to-risk
#     risk = atr  # define risk as ATR

#     # LONG setup
#     sl.loc[long_entry] = close.loc[long_entry] - risk.loc[long_entry]
#     tp.loc[long_entry] = close.loc[long_entry] + risk.loc[long_entry] * rr_ratio

#     # SHORT setup
#     sl.loc[short_entry] = close.loc[short_entry] + risk.loc[short_entry]
#     tp.loc[short_entry] = close.loc[short_entry] - risk.loc[short_entry] * rr_ratio

    # return signals, atr, tp, sl

def atr_breakout_signals(df, atr_multiplier=1, atr_filter=True, use_volume=False, allow_pure_breakout=True, cooldown_period=5):
    import numpy as np
    import pandas as pd

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    ema_fast = close.ewm(span=5, adjust=False).mean()
    ema_slow = close.ewm(span=13, adjust=False).mean()
    ema_slope = ema_fast.diff()

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(5, min_periods=1).mean()

    prev_hi = high.shift(1).rolling(20, min_periods=1).max()
    prev_lo = low.shift(1).rolling(20, min_periods=1).min()
    range_width = prev_hi - prev_lo
    avg_range = range_width.rolling(20).mean()
    tight_range = range_width < avg_range * 0.5
    low_volatility = atr < atr.rolling(20).mean()
    consolidation_zone = tight_range & low_volatility

    avg_volume = volume.rolling(20).mean()
    volume_ok = volume > avg_volume if use_volume else pd.Series(True, index=df.index)

    trend_up = ema_fast > ema_slow
    trend_dn = ema_fast < ema_slow
    strong_trend_up = ema_slope > 0.1
    strong_trend_dn = ema_slope < -0.1

    candle_range = high - low
    min_range = atr * atr_multiplier

    long_breakout = close > prev_hi
    short_breakout = close < prev_lo

    breakout_high = pd.Series(np.where(long_breakout, prev_hi, np.nan), index=df.index).ffill()
    breakout_low = pd.Series(np.where(short_breakout, prev_lo, np.nan), index=df.index).ffill()

    retest_long = (low <= breakout_high) & (close > breakout_high)
    retest_short = (high >= breakout_low) & (close < breakout_low)

    long_entry = trend_up & strong_trend_up & volume_ok & (candle_range >= min_range)
    short_entry = trend_dn & strong_trend_dn & volume_ok & (candle_range >= min_range)

    if allow_pure_breakout:
        long_entry &= (long_breakout | retest_long)
        short_entry &= (short_breakout | retest_short)
    else:
        long_entry &= retest_long
        short_entry &= retest_short

    if atr_filter:
        long_entry &= ~consolidation_zone
        short_entry &= ~consolidation_zone

    signals = pd.Series(index=df.index, dtype=object)
    signals.loc[long_entry] = "LONG"
    signals.loc[short_entry] = "SHORT"

    tp = pd.Series(index=df.index, dtype=float)
    sl = pd.Series(index=df.index, dtype=float)

    rr_ratio = 3.0
    risk = atr

    sl.loc[long_entry] = close.loc[long_entry] - risk.loc[long_entry]
    tp.loc[long_entry] = close.loc[long_entry] + risk.loc[long_entry] * rr_ratio
    sl.loc[short_entry] = close.loc[short_entry] + risk.loc[short_entry]
    tp.loc[short_entry] = close.loc[short_entry] - risk.loc[short_entry] * rr_ratio

    reversal = pd.Series(index=df.index, dtype=object)
    rev_tp = pd.Series(index=df.index, dtype=float)
    rev_sl = pd.Series(index=df.index, dtype=float)

    last_trade_index = -cooldown_period
    last_trade_direction = None

    for i in range(1, len(df)):
        if signals.iloc[i] in ["LONG", "SHORT"]:
            if i - last_trade_index < cooldown_period and signals.iloc[i] == last_trade_direction:
                signals.iloc[i] = None
                continue
            last_trade_direction = signals.iloc[i]
            last_trade_index = i
            last_sl = sl.iloc[i]

        if last_trade_direction == "LONG" and close.iloc[i] < last_sl * 0.995 and not consolidation_zone.iloc[i]:
            reversal.iloc[i] = "SHORT"
            signals.iloc[i] = "SHORT"
            rev_sl.iloc[i] = close.iloc[i] + risk.iloc[i]
            rev_tp.iloc[i] = close.iloc[i] - risk.iloc[i] * rr_ratio
            last_trade_direction = "SHORT"
            last_trade_index = i
            last_sl = rev_sl.iloc[i]

        elif last_trade_direction == "SHORT" and close.iloc[i] > last_sl * 1.005 and not consolidation_zone.iloc[i]:
            reversal.iloc[i] = "LONG"
            signals.iloc[i] = "LONG"
            rev_sl.iloc[i] = close.iloc[i] - risk.iloc[i]
            rev_tp.iloc[i] = close.iloc[i] + risk.iloc[i] * rr_ratio
            last_trade_direction = "LONG"
            last_trade_index = i
            last_sl = rev_sl.iloc[i]

    tp.update(rev_tp)
    sl.update(rev_sl)

    return signals, atr, tp, sl, reversal


# def atr_breakout_signals(df, atr_multiplier=1.5, atr_filter=True, use_volume=False, allow_pure_breakout=True, cooldown_period=10):
#     import numpy as np
#     import pandas as pd

#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = [col[0] for col in df.columns]

#     close = df['Close']
#     high = df['High']
#     low = df['Low']
#     volume = df['Volume']

#     ema_fast = close.ewm(span=20, adjust=False).mean()
#     ema_slow = close.ewm(span=50, adjust=False).mean()
#     ema_slope = ema_fast.diff()

#     prev_close = close.shift(1)
#     tr = pd.concat([
#         high - low,
#         (high - prev_close).abs(),
#         (low - prev_close).abs()
#     ], axis=1).max(axis=1)
#     atr = tr.rolling(3, min_periods=2).mean()

#     prev_hi = high.shift(1).rolling(20, min_periods=1).max()
#     prev_lo = low.shift(1).rolling(20, min_periods=1).min()
#     range_width = prev_hi - prev_lo
#     avg_range = range_width.rolling(20).mean()
#     tight_range = range_width < avg_range * 0.5
#     low_volatility = atr < atr.rolling(20).mean()
#     consolidation_zone = tight_range & low_volatility

#     avg_volume = volume.rolling(20).mean()
#     volume_ok = volume > avg_volume if use_volume else pd.Series(True, index=df.index)

#     trend_up = ema_fast > ema_slow
#     trend_dn = ema_fast < ema_slow
#     strong_trend_up = ema_slope > 0.5
#     strong_trend_dn = ema_slope < -0.5

#     candle_range = high - low
#     min_range = atr * atr_multiplier

#     long_breakout = close > prev_hi
#     short_breakout = close < prev_lo

#     breakout_high = pd.Series(np.where(long_breakout, prev_hi, np.nan), index=df.index).ffill()
#     breakout_low = pd.Series(np.where(short_breakout, prev_lo, np.nan), index=df.index).ffill()

#     retest_long = (low <= breakout_high) & (close > breakout_high)
#     retest_short = (high >= breakout_low) & (close < breakout_low)

#     long_entry = trend_up & strong_trend_up & volume_ok & (candle_range >= min_range)
#     short_entry = trend_dn & strong_trend_dn & volume_ok & (candle_range >= min_range)

#     if allow_pure_breakout:
#         long_entry &= (long_breakout | retest_long)
#         short_entry &= (short_breakout | retest_short)
#     else:
#         long_entry &= retest_long
#         short_entry &= retest_short

#     if atr_filter:
#         long_entry &= ~consolidation_zone
#         short_entry &= ~consolidation_zone

#     signals = pd.Series(index=df.index, dtype=object)
#     signals.loc[long_entry] = "LONG"
#     signals.loc[short_entry] = "SHORT"

#     tp = pd.Series(index=df.index, dtype=float)
#     sl = pd.Series(index=df.index, dtype=float)

#     rr_ratio = 3.0
#     risk = atr

#     sl.loc[long_entry] = close.loc[long_entry] - risk.loc[long_entry]
#     tp.loc[long_entry] = close.loc[long_entry] + risk.loc[long_entry] * rr_ratio
#     sl.loc[short_entry] = close.loc[short_entry] + risk.loc[short_entry]
#     tp.loc[short_entry] = close.loc[short_entry] - risk.loc[short_entry] * rr_ratio

#     reversal = pd.Series(index=df.index, dtype=object)
#     rev_tp = pd.Series(index=df.index, dtype=float)
#     rev_sl = pd.Series(index=df.index, dtype=float)

#     last_trade_index = -cooldown_period
#     last_trade_direction = None

#     for i in range(1, len(df)):
#         if signals.iloc[i] in ["LONG", "SHORT"]:
#             if i - last_trade_index < cooldown_period and signals.iloc[i] == last_trade_direction:
#                 signals.iloc[i] = None
#                 continue
#             last_trade_direction = signals.iloc[i]
#             last_trade_index = i
#             last_sl = sl.iloc[i]

#         if last_trade_direction == "LONG" and close.iloc[i] < last_sl * 0.995 and not consolidation_zone.iloc[i]:
#             reversal.iloc[i] = "SHORT"
#             signals.iloc[i] = "SHORT"
#             rev_sl.iloc[i] = close.iloc[i] + risk.iloc[i]
#             rev_tp.iloc[i] = close.iloc[i] - risk.iloc[i] * rr_ratio
#             last_trade_direction = "SHORT"
#             last_trade_index = i
#             last_sl = rev_sl.iloc[i]

#         elif last_trade_direction == "SHORT" and close.iloc[i] > last_sl * 1.005 and not consolidation_zone.iloc[i]:
#             reversal.iloc[i] = "LONG"
#             signals.iloc[i] = "LONG"
#             rev_sl.iloc[i] = close.iloc[i] - risk.iloc[i]
#             rev_tp.iloc[i] = close.iloc[i] + risk.iloc[i] * rr_ratio
#             last_trade_direction = "LONG"
#             last_trade_index = i
#             last_sl = rev_sl.iloc[i]

#     tp.update(rev_tp)
#     sl.update(rev_sl)

#     return signals, atr, tp, sl, reversal
    
def atr_breakout_signals_intraday(df, atr_multiplier=1, atr_filter=True, use_volume=False, allow_pure_breakout=True, cooldown_period=3):
    import numpy as np
    import pandas as pd

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # EMAs and slope
    ema_fast = close.ewm(span=5, adjust=False).mean()
    ema_slow = close.ewm(span=13, adjust=False).mean()
    ema_slope = ema_fast.diff()

    # ATR calculation
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(3, min_periods=1).mean()

    # Consolidation zone detection
    prev_hi = high.shift(1).rolling(20, min_periods=1).max()
    prev_lo = low.shift(1).rolling(20, min_periods=1).min()
    range_width = prev_hi - prev_lo
    avg_range = range_width.rolling(20).mean()
    tight_range = range_width < avg_range * 0.5
    low_volatility = atr < atr.rolling(20).mean()
    consolidation_zone = tight_range & low_volatility

    # Volume filter
    avg_volume = volume.rolling(20).mean()
    volume_ok = volume > avg_volume if use_volume else pd.Series(True, index=df.index)

    # Trend filters
    trend_up = ema_fast > ema_slow
    trend_dn = ema_fast < ema_slow
    strong_trend_up = ema_slope > 0.03
    strong_trend_dn = ema_slope < -0.03

    # Candle range and breakout logic
    candle_range = high - low
    min_range = atr * atr_multiplier
    long_breakout = close > prev_hi
    short_breakout = close < prev_lo

    breakout_high = pd.Series(np.where(long_breakout, prev_hi, np.nan), index=df.index).ffill()
    breakout_low = pd.Series(np.where(short_breakout, prev_lo, np.nan), index=df.index).ffill()

    retest_long = (low <= breakout_high) & (close > breakout_high)
    retest_short = (high >= breakout_low) & (close < breakout_low)

    # Big move filters
    price_change = close.pct_change()
    volatility_threshold = price_change.rolling(12).std() * 2
    big_move = price_change.abs() > volatility_threshold

    long_term_hi = close.shift(1).rolling(24).max()
    long_term_lo = close.shift(1).rolling(24).min()
    significant_long_breakout = close > long_term_hi
    significant_short_breakout = close < long_term_lo

    atr_spike = atr > atr.rolling(12).mean() * 1.5
    volume_surge = volume > volume.rolling(12).mean() * 1.5

    # Entry conditions
    long_entry = trend_up & strong_trend_up & volume_ok & (candle_range >= min_range)
    short_entry = trend_dn & strong_trend_dn & volume_ok & (candle_range >= min_range)

    if allow_pure_breakout:
        long_entry &= (long_breakout | retest_long)
        short_entry &= (short_breakout | retest_short)
    else:
        long_entry &= retest_long
        short_entry &= retest_short

    if atr_filter:
        long_entry &= ~consolidation_zone
        short_entry &= ~consolidation_zone

    # Relaxed big move filters for intraday
    long_entry &= (big_move | significant_long_breakout | atr_spike | volume_surge)
    short_entry &= (big_move | significant_short_breakout | atr_spike | volume_surge)

    # Signal generation
    signals = pd.Series(index=df.index, dtype=object)
    signals.loc[long_entry] = "LONG"
    signals.loc[short_entry] = "SHORT"

    tp = pd.Series(index=df.index, dtype=float)
    sl = pd.Series(index=df.index, dtype=float)

    # Fixed dollar risk and reward
    risk_dollars = 2.0
    reward_dollars = 6.0

    sl.loc[long_entry] = close.loc[long_entry] - risk_dollars
    tp.loc[long_entry] = close.loc[long_entry] + reward_dollars

    sl.loc[short_entry] = close.loc[short_entry] + risk_dollars
    tp.loc[short_entry] = close.loc[short_entry] - reward_dollars



    # Reversal logic
    reversal = pd.Series(index=df.index, dtype=object)
    rev_tp = pd.Series(index=df.index, dtype=float)
    rev_sl = pd.Series(index=df.index, dtype=float)

    last_trade_index = -cooldown_period
    last_trade_direction = None

    for i in range(1, len(df)):
        if signals.iloc[i] in ["LONG", "SHORT"]:
            if i - last_trade_index < cooldown_period and signals.iloc[i] == last_trade_direction:
                signals.iloc[i] = None
                continue
            last_trade_direction = signals.iloc[i]
            last_trade_index = i
            last_sl = sl.iloc[i]

        if last_trade_direction == "LONG" and close.iloc[i] < last_sl * 0.995 and not consolidation_zone.iloc[i]:
            reversal.iloc[i] = "SHORT"
            signals.iloc[i] = "SHORT"
            rev_sl.iloc[i] = close.iloc[i] + risk_dollars
            rev_tp.iloc[i] = close.iloc[i] - reward_dollars
            last_trade_direction = "SHORT"
            last_trade_index = i
            last_sl = rev_sl.iloc[i]

        elif last_trade_direction == "SHORT" and close.iloc[i] > last_sl * 1.005 and not consolidation_zone.iloc[i]:
            reversal.iloc[i] = "LONG"
            signals.iloc[i] = "LONG"
            rev_sl.iloc[i] = close.iloc[i] - risk_dollars
            rev_tp.iloc[i] = close.iloc[i] + reward_dollars
            last_trade_direction = "LONG"
            last_trade_index = i
            last_sl = rev_sl.iloc[i]

    tp.update(rev_tp)
    sl.update(rev_sl)

    return signals, atr, tp, sl, reversal


def back_testing_atr():
    import yfinance as yf
    import matplotlib.pyplot as plt

    # Download data
    df = yf.download("BTC-USD", period="3d", interval="30m", auto_adjust=True)

    # Get signals with reversal
    # signals, _atr, tp, sl, reversal = atr_breakout_signals(df)
    signals, _atr, tp, sl, reversal = atr_breakout_signals_intraday(df)

    # print(reversal[reversal.notna()])


    # Plot price
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')

    # LONG entries
    long_idx = df.index[signals == "LONG"]
    long_price = df['Close'][signals == "LONG"]
    plt.scatter(long_idx, long_price, marker='^', color='green', label='LONG Signal', s=100)

    # SHORT entries
    short_idx = df.index[signals == "SHORT"]
    short_price = df['Close'][signals == "SHORT"]
    plt.scatter(short_idx, short_price, marker='v', color='red', label='SHORT Signal', s=100)

    # TP and SL for LONG
    plt.scatter(long_idx, tp[long_idx], marker='o', color='lime', label='TP (LONG)', s=60)
    plt.scatter(long_idx, sl[long_idx], marker='o', color='orange', label='SL (LONG)', s=60)

    # TP and SL for SHORT
    plt.scatter(short_idx, tp[short_idx], marker='o', color='magenta', label='TP (SHORT)', s=60)
    plt.scatter(short_idx, sl[short_idx], marker='o', color='cyan', label='SL (SHORT)', s=60)

    # Reversal entries
    rev_long_idx = df.index[reversal == "LONG"]
    rev_short_idx = df.index[reversal == "SHORT"]
    rev_long_price = df['Close'][rev_long_idx]
    rev_short_price = df['Close'][rev_short_idx]

    # Plot reversal signals
    plt.scatter(rev_long_idx, rev_long_price, marker='P', color='darkgreen', label='Reversal to LONG', s=100)
    plt.scatter(rev_short_idx, rev_short_price, marker='X', color='darkred', label='Reversal to SHORT', s=100)

    # Final touches
    plt.title("ATR Breakout Signals with TP/SL and Reversal Trades")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# back_testing_atr()

