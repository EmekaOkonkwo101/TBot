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
    last_trade_direction = None

    while True:
        tickers = ['BTC-USD']
        meta_ticker = 'ETHUSD'

        live_data = yf.download(tickers, period="2d", interval="5m", progress=False)
        signal, atr_value, TP, SL, reversal = atr_breakout_signals(live_data)

        if not mt5.initialize():
            print("MT5 initialization failed")
            mt5.shutdown()

        account = 61392904
        authorized = mt5.login(account, password=config("PASSWORD"), server="Pepperstone-Demo")

        if authorized:
            symbol = meta_ticker
            positions = mt5.positions_get(symbol=symbol)
            sl_update_tracker = {}
            cooldown_seconds = 150
            atr_multiplier = 2

            for pos in positions:
                tick = mt5.symbol_info_tick(symbol)
                current_price = tick.bid if pos.type == mt5.ORDER_TYPE_SELL else tick.ask
                atr = atr_value.iloc[-1]
                trail_distance = atr * atr_multiplier
                entry_price = pos.price_open
                now = time.time()
                last_update = sl_update_tracker.get(pos.ticket, 0)

                tp_trigger_price = entry_price + 0.25 * (pos.tp - entry_price) if pos.type == mt5.ORDER_TYPE_BUY else entry_price - 0.25 * (entry_price - pos.tp)
                tp_threshold_met = current_price >= tp_trigger_price if pos.type == mt5.ORDER_TYPE_BUY else current_price <= tp_trigger_price

                in_profit = current_price > entry_price + trail_distance if pos.type == mt5.ORDER_TYPE_BUY else current_price < entry_price - trail_distance

                if in_profit and tp_threshold_met and now - last_update >= cooldown_seconds:
                    new_sl = current_price - trail_distance if pos.type == mt5.ORDER_TYPE_BUY else current_price + trail_distance
                    if (pos.type == mt5.ORDER_TYPE_BUY and new_sl > pos.sl) or (pos.type == mt5.ORDER_TYPE_SELL and new_sl < pos.sl):
                        modify_request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": pos.ticket,
                            "sl": new_sl,
                            "tp": pos.tp,
                        }
                        result = mt5.order_send(modify_request)
                        logging.info(f"Updated SL for position #{pos.ticket} to {new_sl:.2f}")
                        sl_update_tracker[pos.ticket] = now
                else:
                    logging.info(f"Position #{pos.ticket} not in profit or cooldown active. SL not updated.")

            lot = 0.02
            price = mt5.symbol_info_tick(symbol).ask
            deviation = 20
            has_open_position = any(pos.symbol == symbol for pos in positions)

            latest_signal = signal.iloc[-1] if not signal.empty else None
            latest_index = signal.index[-1] if not signal.empty else None
            tp_price = TP.loc[latest_index] if latest_index in TP else None
            sl_price = SL.loc[latest_index] if latest_index in SL else None

            if latest_signal in ["LONG", "SHORT"] and latest_signal != last_trade_direction and not has_open_position:
                order_type = mt5.ORDER_TYPE_BUY if latest_signal == "LONG" else mt5.ORDER_TYPE_SELL
                order_price = price if latest_signal == "LONG" else mt5.symbol_info_tick(symbol).bid
                magic_number = 234000 if latest_signal == "LONG" else 234001
                comment = "PairsTradeLong" if latest_signal == "LONG" else "PairsTradeShort"

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot,
                    "type": order_type,
                    "price": order_price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "deviation": deviation,
                    "magic": magic_number,
                    "comment": comment,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                result = mt5.order_send(request)
                logging.info(f"{latest_signal} signal triggered for {symbol} at price {order_price}")
                logging.info(f"Order result: {result}")
                last_trade_direction = latest_signal
            else:
                logging.info("No new actionable signal or position already open.")
        else:
            print("MT5 login failed")

        time.sleep(120)

            
except KeyboardInterrupt:
    print("Bot stopped manually.")
finally:
    mt5.shutdown()




