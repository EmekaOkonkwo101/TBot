import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
from decouple import config
import logging
import time


from helpers import atr_breakout_signals


try:
    last_signal = None
    while True:
        tickers = ['ETH-USD']
        meta_ticker = 'ETHUSD'

        live_data = yf.download(tickers, period="2d", interval="5m", progress=False)
        print(live_data)

        # Get signal & ATR value
        signal, atr_value, TP, SL = atr_breakout_signals(live_data)
        
        if not mt5.initialize():
            print("MT5 initialization failed")
            mt5.shutdown()
        
        account = 61392904
        authorized = mt5.login(account, password=config("PASSWORD"), server="Pepperstone-Demo")
            
        if authorized:
            symbol = meta_ticker
            positions = mt5.positions_get(symbol=symbol)
            sl_update_tracker = {}  # key: position.ticket, value: last_update_time
            cooldown_seconds = 300  # 5 minutes
            atr_multiplier = 2

            for pos in positions:
                tp_threshold_met = False
                target_price = pos.tp

                if pos.symbol == symbol:
                    tick = mt5.symbol_info_tick(symbol)
                    current_price = tick.bid if pos.type == mt5.ORDER_TYPE_SELL else tick.ask
                    atr = atr_value.iloc[-1]
                    trail_distance = atr * atr_multiplier
                    entry_price = pos.price_open

                    now = time.time()
                    last_update = sl_update_tracker.get(pos.ticket, 0)

                    # ✅ Check if trade is in profit before updating SL
                    if pos.type == mt5.ORDER_TYPE_BUY:
                        # 25% progress toward TP
                        tp_trigger_price = entry_price + 0.25 * (target_price - entry_price)
                        tp_threshold_met = current_price >= tp_trigger_price

                    elif pos.type == mt5.ORDER_TYPE_SELL:
                        tp_trigger_price = entry_price - 0.25 * (entry_price - target_price)
                        tp_threshold_met = current_price <= tp_trigger_price

                    # In-profit check based on ATR trail distance
                    in_profit = False
                    if pos.type == mt5.ORDER_TYPE_BUY:
                        in_profit = current_price > entry_price + trail_distance
                    elif pos.type == mt5.ORDER_TYPE_SELL:
                        in_profit = current_price < entry_price - trail_distance

                    # Trailing stop update logic
                    if in_profit and tp_threshold_met and now - last_update >= cooldown_seconds:
                        if pos.type == mt5.ORDER_TYPE_BUY:
                            new_sl = current_price - trail_distance
                            if new_sl > pos.sl:
                                modify_request = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "position": pos.ticket,
                                    "sl": new_sl,
                                    "tp": pos.tp,
                                }
                                result = mt5.order_send(modify_request)
                                logging.info(f"Updated SL for BUY position #{pos.ticket} to {new_sl:.2f}")
                                sl_update_tracker[pos.ticket] = now

                        elif pos.type == mt5.ORDER_TYPE_SELL:
                            new_sl = current_price + trail_distance
                            if new_sl < pos.sl:
                                modify_request = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "position": pos.ticket,
                                    "sl": new_sl,
                                    "tp": pos.tp,
                                }
                                result = mt5.order_send(modify_request)
                                logging.info(f"Updated SL for SELL position #{pos.ticket} to {new_sl:.2f}")
                                sl_update_tracker[pos.ticket] = now
                    else:
                        logging.info(f"Position #{pos.ticket} not in profit or cooldown active. SL not updated.")


            lot = 0.05
            price = mt5.symbol_info_tick(symbol).ask
            deviation = 20
            has_open_position = any(pos.symbol == symbol for pos in positions)
            if not signal.empty and not signal.equals(last_signal):
                
                last_signal = signal
                print(f"{signal} signal for {tickers} at ATR={atr_value.iloc[-1]:.2f}")
                latest_index = signal.index[-1]
                tp_price = TP.loc[latest_index]
                sl_price = SL.loc[latest_index]
                if signal.iloc[-1] == "LONG" and not has_open_position:
                    logging.info(f"LONG signal triggered for {symbol} at price {price}")
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": lot,
                        "type": mt5.ORDER_TYPE_BUY,
                        "price": price,
                        "sl": sl_price,
                        "tp": tp_price,
                        "deviation": deviation,
                        "magic": 234000,
                        "comment": "PairsTradeLong",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    result = mt5.order_send(request)
                    logging.info(f"Order result: {result}")
                if signal.iloc[-1] == "SHORT" and not has_open_position:
                    logging.info(f"SHORT signal triggered for {symbol} at price {price}")
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": lot,
                        "type": mt5.ORDER_TYPE_SELL,
                        "price": mt5.symbol_info_tick(symbol).bid,
                        "sl": sl_price,
                        "tp": tp_price,
                        "deviation": deviation,
                        "magic": 234001,
                        "comment": "PairsTradeShort",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    result = mt5.order_send(request)
                    logging.info(f"Order result: {result}")
            else:
                logging.info("No new signal.")
        else:
            print("MT5 login failed")
        
        time.sleep(120)  # Wait 60 seconds before next check
            
except KeyboardInterrupt:
    print("Bot stopped manually.")
finally:
    mt5.shutdown()




