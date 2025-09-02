import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
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

#     ema20 = close.ewm(span=4, adjust=False).mean()
#     ema50 = close.ewm(span=11, adjust=False).mean()

#     prev_close = close.shift(1)
#     tr = pd.concat([
#         high - low,
#         (high - prev_close).abs(),
#         (low - prev_close).abs()
#     ], axis=1).max(axis=1)
#     atr = tr.rolling(7, min_periods=1).mean()

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

#     rr_ratio = 6  # reward-to-risk
#     risk = atr  # define risk as ATR

#     # LONG setup
#     sl.loc[long_entry] = close.loc[long_entry] - risk.loc[long_entry]
#     tp.loc[long_entry] = close.loc[long_entry] + risk.loc[long_entry] * rr_ratio

#     # SHORT setup
#     sl.loc[short_entry] = close.loc[short_entry] + risk.loc[short_entry]
#     tp.loc[short_entry] = close.loc[short_entry] - risk.loc[short_entry] * rr_ratio

#     return signals, atr, tp, sl



def atr_breakout_signals(df, atr_multiplier=1):
    import numpy as np
    import pandas as pd

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    close = df['Close']
    high  = df['High']
    low   = df['Low']

    ema20 = close.ewm(span=4, adjust=False).mean()
    ema50 = close.ewm(span=11, adjust=False).mean()

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(7, min_periods=1).mean()

    # Breakout levels
    prev_hi = high.shift(1).rolling(20, min_periods=1).max()
    prev_lo = low.shift(1).rolling(20, min_periods=1).min()

    trend_up = ema20 > ema50
    trend_dn = ema20 < ema50

    candle_range = high - low
    min_range = atr * atr_multiplier

    # Breakout Detection
    long_breakout = (close > prev_hi)
    short_breakout = (close < prev_lo)

    breakout_high = np.where(long_breakout, prev_hi, np.nan)
    breakout_low  = np.where(short_breakout, prev_lo, np.nan)

    breakout_high = pd.Series(breakout_high, index=df.index).ffill()
    breakout_low  = pd.Series(breakout_low, index=df.index).ffill()

    # Retest conditions
    retest_long = (low <= breakout_high) & (close > breakout_high)
    retest_short = (high >= breakout_low) & (close < breakout_low)

    # Final signal logic
    long_entry = (
        trend_up &
        retest_long &
        (atr > atr.shift(1)) &
        (candle_range >= min_range)
    )

    short_entry = (
        trend_dn &
        retest_short &
        (atr > atr.shift(1)) &
        (candle_range >= min_range)
    )

    signals = pd.Series(index=df.index, dtype=object)
    signals.loc[long_entry] = "LONG"
    signals.loc[short_entry] = "SHORT"

    # TP and SL levels with 3:1 reward-to-risk ratio
    tp = pd.Series(index=df.index, dtype=float)
    sl = pd.Series(index=df.index, dtype=float)

    rr_ratio = 6  # reward-to-risk
    risk = atr  # define risk as ATR

    # LONG setup
    sl.loc[long_entry] = close.loc[long_entry] - risk.loc[long_entry]
    tp.loc[long_entry] = close.loc[long_entry] + risk.loc[long_entry] * rr_ratio

    # SHORT setup
    sl.loc[short_entry] = close.loc[short_entry] + risk.loc[short_entry]
    tp.loc[short_entry] = close.loc[short_entry] - risk.loc[short_entry] * rr_ratio

    return signals, atr, tp, sl




def back_testing_atr():


    df = yf.download("SOL-USD", period="2d", interval="5m", auto_adjust=True)

    # Get signals
    signals, _atr, tp, sl = atr_breakout_signals(df)

    # Plot price
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')

    # LONG entries
    long_idx = df.index[signals == "LONG"]
    long_price = df['Close'][signals == "LONG"]
    plt.scatter(long_idx, long_price, marker='^', color='green', label='LONG Signal', s=100)

    # SHORT entries
    short_idx = df.index[signals == "SHORT"]
    short_price = df['Close'][signals == "SHORT"]
    plt.scatter(short_idx, short_price, marker='v', color='red', label='SHORT Signal', s=100)

    # TP and SL as circles for LONG
    plt.scatter(long_idx, tp[long_idx], marker='o', color='lime', label='TP (LONG)', s=60)
    plt.scatter(long_idx, sl[long_idx], marker='o', color='orange', label='SL (LONG)', s=60)

    # TP and SL as circles for SHORT
    plt.scatter(short_idx, tp[short_idx], marker='o', color='magenta', label='TP (SHORT)', s=60)
    plt.scatter(short_idx, sl[short_idx], marker='o', color='cyan', label='SL (SHORT)', s=60)

    plt.title("ATR Breakout Signals with TP/SL Circles")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

# back_testing_atr()

