import os
import time
import hmac
import hashlib
import urllib.parse
import threading
import json
import re
import sqlite3
import pandas as pd
import numpy as np
import requests
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ----------------- GROQ AI API CONFIGURATION -----------------
GROQ_API_KEY = "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"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

last_groq_call_time = 0
GROQ_COOLDOWN_SECONDS = 45  

DB_FILE = "bot_memory.db"

bot_state = {
    "is_running": False,
    "api_token": "",      
    "api_secret": "",     
    "mode": "paper",       
    "symbol": "ETHUSDT",   
    "leverage": 1,         
    "risk_pct": 15.0,       
    "virtual_balance": 1000.0,
    "current_balance": 1000.0,
    
    "current_price": 0.0,
    "status_message": "Fast Scalp Spot DCA Engine සූදානම්ව පවතී...",
    "wins": 0,
    "losses": 0,
    "total_trades": 0,
    "accuracy": 0.0,
    "current_profit": 0.0,
    "candles": [],
    "ema_20_series": [],
    "ema_50_series": [],
    "ema_200_series": [],
    "active_trades": [],
    
    "current_rsi": 0.0,
    "current_ema_20": 0.0,
    "current_ema_50": 0.0,
    "current_ema_200": 0.0,
    "current_macd": 0.0,
    "current_macd_signal": 0.0,
    "current_atr": 0.0,

    "ml_signal": "NEUTRAL",
    "ml_confidence": 0.0,
    "ai_decision": "WAITING",
    "ai_reasoning": "Fast Scalp DCA Engine සූදානම්ව පවතී...",
    "db_total_trades": 0,
    "db_win_rate": 0.0,
    "auto_tuned_ml_filter": 50.0
}

active_position = None  
last_trade_time = 0
state_lock = threading.Lock()
worker_thread_started = False

def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                avg_price REAL,
                layers_count INTEGER,
                tp REAL,
                sl REAL,
                rsi REAL,
                macd REAL,
                ml_conf REAL,
                pnl REAL,
                outcome TEXT
            )
        ''')
        conn.commit()
        conn.close()
        print("[DATABASE] SQLite Memory Engine initialized successfully!")
    except Exception as e:
        print(f"[DATABASE ERROR] Could not initialize DB: {e}")

def load_db_history():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, symbol, side, avg_price, layers_count, tp, sl, pnl, outcome FROM trade_history ORDER BY id DESC LIMIT 50")
        rows = cursor.fetchall()
        
        cursor.execute("SELECT COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), SUM(pnl) FROM trade_history")
        total_count, total_wins, total_pnl = cursor.fetchone()
        conn.close()

        history_trades = []
        for row in rows:
            timestamp, symbol, side, avg_price, layers_count, tp, sl, pnl, outcome = row
            history_trades.append({
                "time": timestamp,
                "type": f"{side} ({layers_count} Layers)",
                "price": avg_price,
                "tp": tp,
                "sl": sl,
                "pnl": pnl,
                "status": outcome
            })

        tot_trades = total_count if total_count else 0
        wins_count = total_wins if total_wins else 0
        losses_count = (tot_trades - wins_count) if tot_trades >= wins_count else 0
        acc = round((wins_count / tot_trades * 100), 1) if tot_trades > 0 else 0.0
        tot_pnl = round(total_pnl, 2) if total_pnl else 0.0

        with state_lock:
            bot_state["active_trades"] = history_trades
            bot_state["total_trades"] = tot_trades
            bot_state["wins"] = wins_count
            bot_state["losses"] = losses_count
            bot_state["accuracy"] = acc
            bot_state["current_profit"] = tot_pnl
            bot_state["db_total_trades"] = tot_trades
            bot_state["db_win_rate"] = acc

        print(f"[DATABASE] Auto-Loaded {len(history_trades)} historical trades into memory!")
    except Exception as e:
        print(f"[DATABASE ERROR] Could not load DB history: {e}")

init_db()
load_db_history()

def save_trade_to_db(timestamp, symbol, side, avg_price, layers_count, tp, sl, rsi, macd, ml_conf, pnl, outcome):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trade_history (timestamp, symbol, side, avg_price, layers_count, tp, sl, rsi, macd, ml_conf, pnl, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, symbol, side, avg_price, layers_count, tp, sl, rsi, macd, ml_conf, pnl, outcome))
        conn.commit()
        conn.close()
        load_db_history()
    except Exception as e:
        print(f"[DATABASE ERROR] Could not save trade: {e}")

def get_db_stats_and_dynamic_filter():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) FROM trade_history")
        total_count, total_wins = cursor.fetchone()
        conn.close()

        total_trades = total_count if total_count else 0
        historical_win_rate = round((total_wins / total_trades * 100), 1) if total_trades > 0 else 0.0

        min_ml_threshold = 50.0  
        return total_trades, historical_win_rate, min_ml_threshold
    except Exception as e:
        return 0, 0.0, 50.0

PUBLIC_BINANCE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com"
]

def spot_public_request(endpoint, params=None):
    for base_url in PUBLIC_BINANCE_URLS:
        try:
            url = f"{base_url}{endpoint}"
            res = requests.get(url, params=params, timeout=3)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list) and len(data) > 0:
                    return data
                elif isinstance(data, dict) and "code" not in data:
                    return data
        except Exception:
            continue
    return None

def spot_signed_request(endpoint, method="GET", params=None):
    if not params:
        params = {}
    
    with state_lock:
        api_key = bot_state["api_token"]
        api_secret = bot_state["api_secret"]
    
    if not api_key or not api_secret:
        return {"error": "API Key/Secret සපයා නැත."}
        
    params["timestamp"] = int(time.time() * 1000)
    query_string = urllib.parse.urlencode(params)
    signature = hmac.new(api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    params["signature"] = signature
    
    headers = {"X-MBX-APIKEY": api_key}
    
    for base_url in ["https://api.binance.com", "https://api1.binance.com", "https://api2.binance.com", "https://api3.binance.com"]:
        try:
            url = f"{base_url}{endpoint}"
            if method == "GET":
                res = requests.get(url, params=params, headers=headers, timeout=3)
            elif method == "POST":
                res = requests.post(url, data=params, headers=headers, timeout=3)
            
            if res.status_code == 200:
                return res.json()
        except Exception:
            continue
            
    return {"error": "Connection error or blocked endpoint"}

def analyze_trade_with_groq_ai(signal, price, rsi, macd, ema20, ema200, ml_conf, hist_win_rate):
    global last_groq_call_time
    current_time = time.time()
    
    if (current_time - last_groq_call_time) < GROQ_COOLDOWN_SECONDS:
        return True, "Groq Limit Protection: Fast Scalp Base Layer approved."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
    You are an expert Crypto Spot DCA Risk Manager. Validate base entry:
    Symbol: {signal}, Price: {price}, RSI: {rsi}, MACD: {macd}, ML Conf: {ml_conf}%
    Respond ONLY in JSON: {{"decision": "CONFIRM" or "REJECT", "reason": "Short summary"}}
    """

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 120
    }

    try:
        last_groq_call_time = time.time()
        res = requests.post(GROQ_URL, json=payload, headers=headers, timeout=4)
        if res.status_code == 200:
            result = res.json()
            content = result['choices'][0]['message']['content']
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                decision = parsed.get("decision", "REJECT") == "CONFIRM"
                reason = parsed.get("reason", "AI Approved Fast Scalp Base Entry")
                return decision, reason
        return True, "Groq AI bypass due to API limit/timeout."
    except Exception as e:
        return True, "Groq AI bypass due to connection timeout."

def train_and_predict_ml(df):
    try:
        df['ema_diff_20_200'] = (df['ema_20'] - df['ema_200']) / df['close']
        df['price_ema20_diff'] = (df['close'] - df['ema_20']) / df['close']
        df['macd_diff'] = df['macd'] - df['macd_signal']
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        features = ['rsi', 'ema_diff_20_200', 'price_ema20_diff', 'macd_diff', 'atr']
        clean_df = df.dropna().copy()
        
        if len(clean_df) < 100:
            return "NEUTRAL", 50.0

        X = clean_df[features][:-1]
        y = clean_df['target'][:-1]
        
        model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        model.fit(X, y)
        
        latest_features = clean_df[features].iloc[-1:].values
        probs = model.predict_proba(latest_features)[0]
        
        prob_up = probs[1] * 100
        
        if prob_up >= 50.0:
            return "LONG", round(prob_up, 1)
        else:
            return "NEUTRAL", round(prob_up, 1)

    except Exception as e:
        return "NEUTRAL", 50.0

def get_klines_and_indicators(symbol):
    params = {"symbol": symbol, "interval": "3m", "limit": 250}
    data = spot_public_request("/api/v3/klines", params)
    if not data or len(data) < 200:
        return None, None, None
    
    df = pd.DataFrame(data, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    df['time'] = (df['time'].astype(int) / 1000).astype(int)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    df['ema_20'] = close.ewm(span=20, adjust=False).mean()
    df['ema_50'] = close.ewm(span=50, adjust=False).mean()
    df['ema_200'] = close.ewm(span=200, adjust=False).mean()
    
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    ema_gain = gain.ewm(com=13, adjust=False).mean()
    ema_loss = loss.ewm(com=13, adjust=False).mean()
    rs = ema_gain / (ema_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()

    chart_candles = []
    ema_20_list = []
    ema_50_list = []
    ema_200_list = []

    for i in range(len(df)):
        t = int(df['time'].iloc[i])
        chart_candles.append({
            "time": t, "open": float(df['open'].iloc[i]),
            "high": float(df['high'].iloc[i]), "low": float(df['low'].iloc[i]),
            "close": float(df['close'].iloc[i])
        })
        if not np.isnan(df['ema_20'].iloc[i]):
            ema_20_list.append({"time": t, "value": float(df['ema_20'].iloc[i])})
        if not np.isnan(df['ema_50'].iloc[i]):
            ema_50_list.append({"time": t, "value": float(df['ema_50'].iloc[i])})
        if not np.isnan(df['ema_200'].iloc[i]):
            ema_200_list.append({"time": t, "value": float(df['ema_200'].iloc[i])})

    indicators = {
        "rsi": round(float(df['rsi'].iloc[-1]), 2),
        "ema_20": round(float(df['ema_20'].iloc[-1]), 4),
        "ema_50": round(float(df['ema_50'].iloc[-1]), 4),
        "ema_200": round(float(df['ema_200'].iloc[-1]), 4),
        "macd": round(float(df['macd'].iloc[-1]), 4),
        "macd_signal": round(float(df['macd_signal'].iloc[-1]), 4),
        "atr": round(float(df['atr'].iloc[-1]), 4),
        "current_price": float(close.iloc[-1]),
        "ema_20_series": ema_20_list,
        "ema_50_series": ema_50_list,
        "ema_200_series": ema_200_list
    }
    
    return chart_candles, indicators, df

def recalculate_spot_dca_levels(layers):
    total_qty = sum(l["qty"] for l in layers)
    total_cost = sum(l["cost"] for l in layers)
    avg_price = total_cost / total_qty if total_qty > 0 else 0.0

    # Fast Scalp Take Profit Target: 0.85% above avg entry (Quick execution)
    tp_price = round(avg_price * 1.0085, 4)
    lowest_price = min(l["price"] for l in layers)
    sl_price = round(lowest_price * 0.980, 4) 

    return round(avg_price, 4), round(total_qty, 4), round(total_cost, 2), tp_price, sl_price

def process_bot_logic(symbol, mode, risk_pct):
    global active_position, last_trade_time

    candles, ind, df_klines = get_klines_and_indicators(symbol)
    if not ind or df_klines is None:
        return

    current_time = time.time()
    current_price = ind["current_price"]
    
    ml_signal, ml_conf = train_and_predict_ml(df_klines)
    db_total, db_win_rate, min_ml_filter = get_db_stats_and_dynamic_filter()

    with state_lock:
        bot_state["current_price"] = current_price
        bot_state["candles"] = candles
        bot_state["ema_20_series"] = ind["ema_20_series"]
        bot_state["ema_50_series"] = ind["ema_50_series"]
        bot_state["ema_200_series"] = ind["ema_200_series"]
        bot_state["current_rsi"] = ind["rsi"]
        bot_state["current_ema_20"] = ind["ema_20"]
        bot_state["current_ema_50"] = ind["ema_50"]
        bot_state["current_ema_200"] = ind["ema_200"]
        bot_state["current_macd"] = ind["macd"]
        bot_state["current_macd_signal"] = ind["macd_signal"]
        bot_state["current_atr"] = ind["atr"]
        bot_state["ml_signal"] = ml_signal
        bot_state["ml_confidence"] = ml_conf
        bot_state["db_total_trades"] = db_total
        bot_state["db_win_rate"] = db_win_rate
        bot_state["auto_tuned_ml_filter"] = min_ml_filter

    if not bot_state["is_running"]:
        return

    atr_val = ind["atr"] if ind["atr"] > 0 else (current_price * 0.005)

    if mode == "real":
        acc_info = spot_signed_request("/api/v3/account")
        if isinstance(acc_info, dict) and "balances" in acc_info:
            usdt_asset = next((a for a in acc_info["balances"] if a["asset"] == "USDT"), None)
            if usdt_asset:
                with state_lock:
                    bot_state["current_balance"] = float(usdt_asset["free"])

    # 1. FAST SCALP SPOT DCA + TRAILING PROFIT GUARD
    if active_position:
        layers = active_position["layers"]
        avg_price = active_position["avg_price"]
        total_qty = active_position["total_qty"]
        tp_price = active_position["tp_price"]
        sl_price = active_position["sl_price"]
        last_layer_price = layers[-1]["price"]

        # Fast Trailing Profit Guard (+0.3% activation)
        if current_price >= avg_price * 1.0030:
            min_profit_sl = round(avg_price * 1.0018, 4) # Lock in net profit above fee
            potential_trailing_sl = round(current_price - (0.6 * atr_val), 4)
            new_sl = max(min_profit_sl, potential_trailing_sl)

            if new_sl > sl_price:
                active_position["sl_price"] = new_sl
                active_position["trailing_tp_active"] = True
                sl_price = new_sl

        layer_step_pct = max(0.007, (1.0 * atr_val) / current_price) 

        can_add_layer = False
        if len(layers) < 3 and current_price <= last_layer_price * (1 - layer_step_pct) and not active_position.get("trailing_tp_active", False):
            can_add_layer = True

        if can_add_layer:
            with state_lock:
                balance = bot_state["current_balance"]
            
            next_layer_usd = max(6.0, balance * (risk_pct / 100)) 
            next_layer_qty = round(next_layer_usd / current_price, 4)

            order_ok = True
            if mode == "real":
                res = spot_signed_request("/api/v3/order", "POST", {
                    "symbol": symbol, "side": "BUY", "type": "MARKET", "quoteOrderQty": round(next_layer_usd, 2)
                })
                if "orderId" not in res:
                    order_ok = False

            if order_ok and next_layer_qty > 0:
                layer_index = len(layers) + 1
                entry_time_str = time.strftime('%H:%M:%S', time.localtime())
                layers.append({
                    "layer": layer_index,
                    "price": current_price,
                    "qty": next_layer_qty,
                    "cost": round(next_layer_usd, 2),
                    "time": entry_time_str
                })
                avg_price, total_qty, total_cost, tp_price, sl_price = recalculate_spot_dca_levels(layers)
                
                active_position["layers"] = layers
                active_position["avg_price"] = avg_price
                active_position["total_qty"] = total_qty
                active_position["total_cost"] = total_cost
                active_position["tp_price"] = tp_price
                active_position["sl_price"] = sl_price

        status_trail = " (Trailing Active 🔥)" if active_position.get("trailing_tp_active", False) else ""
        status_msg = f"SPOT DCA (LONG {len(layers)}/3 Layers){status_trail} | Avg මිල: ${avg_price} | TP: ${tp_price} | Trailing/SL: ${sl_price}"
        with state_lock:
            bot_state["status_message"] = status_msg

        trade_closed = False
        gross_pnl = 0.0

        if current_price >= tp_price:
            gross_pnl = (tp_price - avg_price) * total_qty
            trade_closed = True
        elif current_price <= sl_price:
            gross_pnl = (sl_price - avg_price) * total_qty
            trade_closed = True

        if trade_closed:
            notional_val = total_qty * avg_price
            est_binance_spot_fee = (notional_val * 2) * 0.0010 
            net_pnl = round(gross_pnl - est_binance_spot_fee, 2)

            if active_position.get("trailing_tp_active", False) and net_pnl >= 0:
                outcome = f"ජයග්‍රහණය (Trailing Profit Hit 🔥 - {len(layers)} Layers)"
            elif net_pnl > 0:
                outcome = f"ජයග්‍රහණය (Spot DCA Target Hit - {len(layers)} Layers)"
            else:
                outcome = f"පරාජය (Emergency SL Hit)"

            if mode == "real":
                spot_signed_request("/api/v3/order", "POST", {
                    "symbol": symbol, "side": "SELL", "type": "MARKET", "quantity": round(total_qty, 4)
                })

            save_trade_to_db(active_position["entry_time"], symbol, "SPOT LONG", avg_price, len(layers), tp_price, sl_price, active_position["rsi"], active_position["macd"], active_position["ml_conf"], net_pnl, outcome)

            active_position = None
            last_trade_time = current_time

    # 2. FAST FLEXIBLE SPOT BASE ENTRY SIGNAL SEARCH
    else:
        cooldown_period = 10  # Fast 10-second cooldown for maximum daily frequency
        if (current_time - last_trade_time) < cooldown_period:
            rem_sec = int(cooldown_period - (current_time - last_trade_time))
            with state_lock:
                bot_state["status_message"] = f"විරාමය (Cooldown): තව තත්පර {rem_sec}..."
        else:
            with state_lock:
                bot_state["status_message"] = f"Fast Scalp Base Entry Signal (ML Filter: {min_ml_filter}%) නිරීක්ෂණය වේ..."
            
            tech_signal = None
            macd_diff = ind["macd"] - ind["macd_signal"]

            # Flexible High-Frequency Entry Trigger
            if ((ind["ema_20"] > ind["ema_50"] or macd_diff > 0) and 
                (30 <= ind["rsi"] <= 72)):
                tech_signal = "LONG"

            if tech_signal and tech_signal == ml_signal and ml_conf >= min_ml_filter:
                with state_lock:
                    bot_state["status_message"] = f"Groq AI හරහා Fast Scalp Base Entry එක තහවුරු කරමින්..."
                
                ai_approved, ai_reason = analyze_trade_with_groq_ai(
                    tech_signal, current_price, ind["rsi"], 
                    ind["macd"], ind["ema_20"], ind["ema_200"], ml_conf, db_win_rate
                )

                with state_lock:
                    bot_state["ai_decision"] = "CONFIRMED" if ai_approved else "REJECTED"
                    bot_state["ai_reasoning"] = ai_reason

                if ai_approved:
                    with state_lock:
                        balance = bot_state["current_balance"]
                    
                    base_layer_usd = max(6.0, balance * (risk_pct / 100)) 
                    base_qty = round(base_layer_usd / current_price, 4)

                    if base_qty > 0 and balance >= 6:
                        order_success = True
                        if mode == "real":
                            res = spot_signed_request("/api/v3/order", "POST", {
                                "symbol": symbol, "side": "BUY", "type": "MARKET", "quoteOrderQty": round(base_layer_usd, 2)
                            })
                            if "orderId" not in res:
                                order_success = False

                        if order_success:
                            entry_time_str = time.strftime('%H:%M:%S', time.localtime())
                            initial_layers = [{
                                "layer": 1,
                                "price": current_price,
                                "qty": base_qty,
                                "cost": round(base_layer_usd, 2),
                                "time": entry_time_str
                            }]
                            avg_price, total_qty, total_cost, tp_price, sl_price = recalculate_spot_dca_levels(initial_layers)

                            active_position = {
                                "side": "SPOT LONG",
                                "layers": initial_layers,
                                "avg_price": avg_price,
                                "total_qty": total_qty,
                                "total_cost": total_cost,
                                "tp_price": tp_price,
                                "sl_price": sl_price,
                                "entry_time": entry_time_str,
                                "rsi": ind["rsi"], "macd": ind["macd"], "ml_conf": ml_conf,
                                "trailing_tp_active": False
                            }

                            with state_lock:
                                bot_state["status_message"] = f"Binance Spot Base Layer 1 (ETH) ඇතුළත් විය!"

def bot_worker():
    while True:
        try:
            with state_lock:
                symbol = bot_state["symbol"]
                mode = bot_state["mode"]
                risk_pct = bot_state["risk_pct"]
            
            process_bot_logic(symbol, mode, risk_pct)
        except Exception as e:
            print(f"Bot cycle exception: {e}")
            
        time.sleep(2)

def start_worker_safely():
    global worker_thread_started
    if not worker_thread_started:
        with state_lock:
            if not worker_thread_started:
                t = threading.Thread(target=bot_worker, daemon=True)
                t.start()
                worker_thread_started = True

# ----------------- FLASK ROUTES -----------------
@app.route("/")
def index():
    start_worker_safely()
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    except FileNotFoundError:
        return "Error: index.html ගොනුව හමුවූයේ නැත!", 404

@app.route("/api/start", methods=["POST"])
def start_bot():
    start_worker_safely()
    data = request.json
    with state_lock:
        bot_state["api_token"] = data.get("api_token", "")
        bot_state["api_secret"] = data.get("api_secret", "")
        bot_state["mode"] = data.get("mode", "paper")
        bot_state["symbol"] = data.get("symbol", "ETHUSDT")
        bot_state["leverage"] = 1 
        bot_state["risk_pct"] = float(data.get("risk_pct", 15.0))
        
        if bot_state["mode"] == "paper" and not bot_state["is_running"]:
            init_bal = float(data.get("start_balance", 1000.0))
            bot_state["virtual_balance"] = init_bal
            bot_state["current_balance"] = init_bal
            
        bot_state["is_running"] = True
        bot_state["status_message"] = f"Binance Spot Fast Scalp AI Engine ({bot_state['mode'].upper()} Mode) ආරම්භ විය!"
        
    return jsonify({"status": "success", "message": "Fast Scalp Spot Bot සාර්ථකව ආරම්භ විය!"})

@app.route("/api/reset_demo", methods=["POST"])
def reset_demo():
    data = request.json
    init_bal = float(data.get("start_balance", 1000.0))
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM trade_history")
        conn.commit()
        conn.close()
        print("[DATABASE] Trade history cleared on explicit Reset Demo!")
    except Exception as e:
        print(f"[DATABASE ERROR] Could not clear DB on reset: {e}")

    with state_lock:
        bot_state["virtual_balance"] = init_bal
        bot_state["current_balance"] = init_bal
        bot_state["wins"] = 0
        bot_state["losses"] = 0
        bot_state["total_trades"] = 0
        bot_state["accuracy"] = 0.0
        bot_state["current_profit"] = 0.0
        bot_state["active_trades"] = []
        bot_state["db_total_trades"] = 0
        bot_state["db_win_rate"] = 0.0
        bot_state["status_message"] = f"Demo Balance එක සාර්ථකව ${init_bal} ට Reset විය."

    return jsonify({"status": "success", "message": f"Demo Balance සහ Trade History එක සාර්ථකව ${init_bal} ට Reset විය!"})

@app.route("/api/stop", methods=["POST"])
def stop_bot():
    with state_lock:
        bot_state["is_running"] = False
        bot_state["status_message"] = "රොබෝ නවතා ඇත (Stopped)"
    return jsonify({"status": "success", "message": "Bot සාර්ථකව නවතා දමන ලදී."})

@app.route("/api/status")
def get_status():
    start_worker_safely()
    with state_lock:
        symbol = bot_state["symbol"]
        mode = bot_state["mode"]
        risk_pct = bot_state["risk_pct"]

    try:
        process_bot_logic(symbol, mode, risk_pct)
    except Exception:
        pass

    with state_lock:
        active_pos_data = None
        available_free_balance = bot_state["current_balance"]

        if active_position:
            unrealized_pnl = round((bot_state["current_price"] - active_position["avg_price"]) * active_position["total_qty"], 2)
            active_pos_data = {
                "side": active_position["side"],
                "layers": active_position["layers"],
                "layers_count": len(active_position["layers"]),
                "avg_price": active_position["avg_price"],
                "total_qty": active_position["total_qty"],
                "total_cost": active_position["total_cost"],
                "tp_price": active_position["tp_price"],
                "sl_price": active_position["sl_price"],
                "unrealized_pnl": unrealized_pnl,
                "entry_time": active_position["entry_time"],
                "trailing_tp_active": active_position.get("trailing_tp_active", False)
            }

            if bot_state["mode"] == "paper":
                available_free_balance = max(0.0, bot_state["current_balance"] - active_position["total_cost"])

        return jsonify({
            "is_running": bot_state["is_running"],
            "mode": bot_state["mode"],
            "mode_verified": bot_state["mode"].upper(),
            "symbol": bot_state["symbol"],
            "leverage": 1,
            "risk_pct": bot_state["risk_pct"],
            "balance": available_free_balance,
            "invested_cost": active_position["total_cost"] if active_position else 0.0,
            "current_profit": bot_state["current_profit"],
            "total_trades": bot_state["total_trades"],
            "accuracy": bot_state["accuracy"],
            "wins": bot_state["wins"],
            "losses": bot_state["losses"],
            "rsi": bot_state["current_rsi"],
            "ema_20": bot_state["current_ema_20"],
            "ema_50": bot_state["current_ema_50"],
            "ema_200": bot_state["current_ema_200"],
            "macd": bot_state["current_macd"],
            "macd_signal": bot_state["current_macd_signal"],
            "atr": bot_state["current_atr"],
            "current_price": bot_state["current_price"],
            "status_message": bot_state["status_message"],
            "candles": bot_state["candles"],
            "ema_20_series": bot_state["ema_20_series"],
            "ema_50_series": bot_state["ema_50_series"],
            "ema_200_series": bot_state["ema_200_series"],
            "active_trades": bot_state["active_trades"],
            "active_position": active_pos_data,
            "ml_signal": bot_state["ml_signal"],
            "ml_confidence": bot_state["ml_confidence"],
            "ai_decision": bot_state["ai_decision"],
            "ai_reasoning": bot_state["ai_reasoning"],
            "db_total_trades": bot_state["db_total_trades"],
            "db_win_rate": bot_state["db_win_rate"],
            "auto_tuned_ml_filter": bot_state["auto_tuned_ml_filter"]
        })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
