import requests
import yfinance as yf
import time
import logging
from decouple import config
from helpers import atr_breakout_signals

DERIV_API_TOKEN = "AkiNgAu1tDXHzcM"
API_URL = "https://api.deriv.com/v2"
symbol = "frxBTCUSD"
amount = 7  # USD stake
headers = {
    "Authorization": f"Bearer {DERIV_API_TOKEN}",
    "Content-Type": "application/json"
}

def place_multiplier_trade(symbol, direction, amount, tp, sl):
    contract_type = "MULTUP" if direction == "LONG" else "MULTDOWN"
    payload = {
        "buy": 1,
        "parameters": {
            "contract_type": contract_type,
            "symbol": symbol,
            "amount": amount,
            "basis": "stake",
            "currency": "USD",
            "take_profit": tp,
            "stop_loss": sl
        }
    }
    response = requests.post(f"{API_URL}/buy", json=payload, headers=headers)
    print(response.json())

# === Live Bot Loop ===
last_trade_direction = None
tickers = ['ETH-USD']

try:
    while True:
        live_data = yf.download(tickers, period="3d", interval="15m", progress=False)
        signal, atr_value, TP, SL, reversal = atr_breakout_signals(live_data)

        latest_signal = signal.iloc[-1] if not signal.empty else None
        latest_index = signal.index[-1] if not signal.empty else None
        tp_price = TP.loc[latest_index] if latest_index in TP else None
        sl_price = SL.loc[latest_index] if latest_index in SL else None

        if latest_signal in ["LONG", "SHORT"] and latest_signal != last_trade_direction:
            place_multiplier_trade(symbol, latest_signal, amount, tp_price, sl_price)
            logging.info(f"Trade placed: {latest_signal} | TP: {tp_price} | SL: {sl_price}")
            last_trade_direction = latest_signal
        else:
            logging.info("No new signal or same direction as last trade.")

        time.sleep(120)  # Wait 2 minutes before checking again

except KeyboardInterrupt:
    print("Bot stopped manually.")