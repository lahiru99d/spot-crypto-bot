import os
import gc
import time
import hmac
import hashlib
import urllib.parse
import threading
import sqlite3
import logging
import traceback
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from flask import Flask, jsonify, request

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================
# 🔒 CONFIGURATION & LOGGING
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

DB_FILE = "bot_memory.db"
TRADE_COOLDOWN_SECONDS = 15
TIMEFRAME = "3m"

app = Flask(__name__)

# ============================================
# 📊 BOT STATE
# ============================================
bot_state = {
    "is_running": False,
    "mode": "PAPER",
    "trading_mode": "PAPER",
    "trade_mode": "PAPER",
    "tradingMode": "PAPER",
    "tradeMode": "PAPER",
    "mode_type": "PAPER",
    "mode_name": "PAPER",
    "symbol": "ETHUSDT",
    "leverage": 1,
    "risk_pct": 10.0,
    "max_layers": 2,  # Strictly limited to 2 Layers
    "total_layers": 2,
    "virtual_balance": 1000.0,
    "current_balance": 1000.0,
    "current_price": 0.0,
    "status_message": "Safe Scalper Bot සූදානම්ව පවතී...",
    
    # Trade metrics
    "wins": 0,
    "losses": 0,
    "total_trades": 0,
    "accuracy": 0.0,
    "current_profit": 0.0,
    "consecutive_losses": 0,
    
    # Technical indicators
    "current_rsi": 50.0,
    "current_ema_20": 0.0,
    "current_ema_50": 0.0,
    "current_ema_200": 0.0,
    "current_macd": 0.0,
    "current_macd_signal": 0.0,
    "current_atr": 0.0,
    "volatility": 0.0,
    
    # Signals
    "ml_signal": "ACTIVE",
    "ml_confidence": 75.0,
    "technical_signal": "NEUTRAL",
    "final_signal": "NEUTRAL",
    
    # Chart data
    "candles": [],
    "ema_20_series": [],
    "ema_50_series": [],
    "ema_200_series": [],
    "active_trades": []
}

active_position = None
last_trade_time = 0
state_lock = threading.Lock()
worker_thread = None

# ============================================
# 🗄️ SAFE DATABASE HANDLERS
# ============================================
def get_db_connection():
    return sqlite3.connect(DB_FILE, timeout=20.0)

def init_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                layers INTEGER,
                tp REAL,
                sl REAL,
                rsi REAL,
                macd REAL,
                ml_confidence REAL,
                pnl REAL,
                outcome TEXT,
                exit_time TEXT
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized successfully!")
    except Exception as e:
        logger.error(f"❌ DB init error: {e}")

def load_db_history():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, symbol, side, entry_price, exit_price, 
                   layers, tp, sl, pnl, outcome FROM trade_history 
            ORDER BY id DESC LIMIT 50
        """)
        rows = cursor.fetchall()
        
        cursor.execute("""
            SELECT COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 
                   SUM(pnl)
            FROM trade_history
        """)
        total_count, total_wins, total_pnl = cursor.fetchone()
        conn.close()

        history_trades = []
        for row in rows:
            timestamp, symbol, side, entry_price, exit_price, layers, tp, sl, pnl, outcome = row
            # Provide ALL key aliases so frontend table never shows $undefined!
            history_trades.append({
                "time": timestamp,
                "timestamp": timestamp,
                "type": f"{side}",
                "side": f"{side}",
                "layers": layers,
                "entry": round(entry_price, 2) if entry_price else 0.0,
                "entry_price": round(entry_price, 2) if entry_price else 0.0,
                "entryPrice": round(entry_price, 2) if entry_price else 0.0,
                "cost": round(entry_price, 2) if entry_price else 0.0,
                "exit": round(exit_price, 2) if exit_price else 0.0,
                "exit_price": round(exit_price, 2) if exit_price else 0.0,
                "tp": round(tp, 2) if tp else 0.0,
                "take_profit": round(tp, 2) if tp else 0.0,
                "tp_price": round(tp, 2) if tp else 0.0,
                "sl": round(sl, 2) if sl else 0.0,
                "emergency_sl": round(sl, 2) if sl else 0.0,
                "stop_loss": round(sl, 2) if sl else 0.0,
                "sl_price": round(sl, 2) if sl else 0.0,
                "pnl": round(pnl, 2) if pnl else 0.0,
                "status": outcome,
                "outcome": outcome
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

    except Exception as e:
        logger.error(f"❌ Load history error: {e}")

def save_trade_to_db(entry_time, exit_time, symbol, side, entry_price, exit_price, 
                     layers, tp, sl, rsi, macd, conf, pnl, outcome):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trade_history 
            (timestamp, exit_time, symbol, side, entry_price, exit_price, layers, 
             tp, sl, rsi, macd, ml_confidence, pnl, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (entry_time, exit_time, symbol, side, entry_price, exit_price, layers, 
              tp, sl, rsi, macd, conf, pnl, outcome))
        conn.commit()
        conn.close()
        load_db_history()
        logger.info(f"💾 Trade Saved: {outcome} | PnL: ${pnl:.2f}")
    except Exception as e:
        logger.error(f"❌ Save trade error: {e}")

# ============================================
# 🎯 REFINED ENTRY FILTER (PREVENT FALLING KNIFE)
# ============================================
def get_technical_signal(ind):
    price = ind["current_price"]
    ema20 = ind["ema_20"]
    ema50 = ind["ema_50"]
    macd = ind["macd"]
    macd_signal = ind["macd_signal"]
    rsi = ind["rsi"]

    macd_bullish = macd > macd_signal
    macd_hist = macd - macd_signal

    # 1. Genuine Dip Reversal (RSI oversold curl with MACD turning green)
    if (30 <= rsi <= 45) and macd_bullish and macd_hist > 0:
        return "LONG", 80.0

    # 2. Trend Momentum Scalp (Above EMA 20 with good support)
    if (price >= ema20 * 0.999) and (ema20 >= ema50) and macd_bullish and (45 <= rsi <= 65):
        return "LONG", 75.0

    # 3. Deep Oversold Rebound
    if rsi < 30 and macd_hist > -0.2:
        return "LONG", 70.0

    # Strong bearish or dump -> Don't enter!
    if rsi > 68 or (not macd_bullish and rsi < 35):
        return "SHORT", 60.0

    return "NEUTRAL", 50.0

# ============================================
# 📊 SAFE BINANCE FETCHER
# ============================================
def get_klines_and_indicators(symbol):
    endpoints = [
        "https://api.binance.us/api/v3/klines",
        "https://api.binance.com/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://data-api.binance.vision/api/v3/klines"
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    data = None
    params = {"symbol": symbol, "interval": TIMEFRAME, "limit": 150}
    
    for url in endpoints:
        try:
            res = requests.get(url, params=params, headers=headers, timeout=4)
            if res.status_code == 200:
                raw = res.json()
                if isinstance(raw, list) and len(raw) >= 50:
                    data = raw
                    break
        except Exception:
            continue

    if not data or len(data) < 50:
        return None, None

    try:
        df = pd.DataFrame(data, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
        ])
        
        df['time'] = (df['time'].astype(np.int64) / 1000).astype(int)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        df['ema_20'] = close.ewm(span=20, adjust=False).mean()
        df['ema_50'] = close.ewm(span=50, adjust=False).mean()
        df['ema_200'] = close.ewm(span=200, adjust=False).mean()
        
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss = (-1 * delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        df['volatility'] = close.pct_change().rolling(window=15).std() * 100

        chart_candles = []
        ema_20_list, ema_50_list, ema_200_list = [], [], []
        tail_df = df.tail(80)

        for _, row in tail_df.iterrows():
            t = int(row['time'])
            chart_candles.append({
                "time": t,
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close'])
            })
            if not np.isnan(row['ema_20']):
                ema_20_list.append({"time": t, "value": float(row['ema_20'])})
            if not np.isnan(row['ema_50']):
                ema_50_list.append({"time": t, "value": float(row['ema_50'])})
            if not np.isnan(row['ema_200']):
                ema_200_list.append({"time": t, "value": float(row['ema_200'])})
        
        indicators = {
            "rsi": round(float(df['rsi'].iloc[-1]), 2),
            "ema_20": round(float(df['ema_20'].iloc[-1]), 4),
            "ema_50": round(float(df['ema_50'].iloc[-1]), 4),
            "ema_200": round(float(df['ema_200'].iloc[-1]), 4),
            "macd": round(float(df['macd'].iloc[-1]), 4),
            "macd_signal": round(float(df['macd_signal'].iloc[-1]), 4),
            "atr": round(float(df['atr'].iloc[-1]), 4),
            "volatility": round(float(df['volatility'].iloc[-1]), 2),
            "current_price": float(close.iloc[-1]),
            "ema_20_series": ema_20_list,
            "ema_50_series": ema_50_list,
            "ema_200_series": ema_200_list
        }
        return chart_candles, indicators
    except Exception as e:
        logger.error(f"❌ Parse indicator error: {e}")
        return None, None

# ============================================
# 💹 2-LAYER DCA & PROFITABLE TAKE PROFIT
# ============================================
def calculate_dca_levels(layers):
    if not layers:
        return None
    total_qty = sum(l["qty"] for l in layers)
    total_cost = sum(l["cost"] for l in layers)
    avg_price = total_cost / total_qty if total_qty > 0 else 0.0

    # REAL PROFIT TARGETS (Exceeds 0.2% fee easily!)
    if len(layers) == 1:
        tp_ratio = 1.008  # +0.8% Take Profit (Clean net profit after fee)
        sl_ratio = 0.980  # -2.0% Stop Loss
    else:
        # Layer 2 (Final Layer)
        tp_ratio = 1.005  # +0.5% Average-out profit
        sl_ratio = 0.975  # -2.5% Stop Loss

    return (round(avg_price, 4), round(total_qty, 4), round(total_cost, 2), 
            round(avg_price * tp_ratio, 4), round(avg_price * sl_ratio, 4))

def place_market_order(symbol, side, quote_qty, mode):
    if mode.lower() == "paper":
        return {"orderId": "PAPER_" + str(int(time.time())), "status": "FILLED"}
    
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        return None
    
    try:
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quoteOrderQty": round(quote_qty, 2),
            "timestamp": int(time.time() * 1000)
        }
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            BINANCE_API_SECRET.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
        
        res = requests.post(
            "https://api.binance.com/api/v3/order",
            params=params,
            headers=headers,
            timeout=5
        )
        return res.json() if res.status_code == 200 else None
    except Exception as e:
        logger.error(f"❌ Order error: {e}")
        return None

# ============================================
# 🎯 CORE TRADING LOGIC
# ============================================
def process_bot_logic(symbol, mode, risk_pct):
    global active_position, last_trade_time
    
    candles, ind = get_klines_and_indicators(symbol)
    if not ind or candles is None:
        with state_lock:
            bot_state["status_message"] = "⚠️ Binance Data සම්බන්ධතාව පරීක්ෂා කරමින්..."
        return
    
    current_time = time.time()
    current_price = ind["current_price"]
    
    tech_signal, tech_conf = get_technical_signal(ind)
    final_signal = tech_signal
    
    # State update
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
        bot_state["volatility"] = ind["volatility"]
        bot_state["ml_signal"] = final_signal
        bot_state["ml_confidence"] = tech_conf
        bot_state["technical_signal"] = tech_signal
        bot_state["final_signal"] = final_signal

    if not bot_state["is_running"]:
        with state_lock:
            bot_state["status_message"] = f"Live Market සම්බන්ධයි (${current_price}) | Bot Start කිරීමට සූදානම්..."
        return

    # ===== MANAGE ACTIVE TRADE =====
    if active_position:
        layers = active_position["layers"]
        avg_price = active_position["avg_price"]
        total_qty = active_position["total_qty"]
        tp_price = active_position["tp_price"]
        sl_price = active_position["sl_price"]
        
        unrealized_pnl = (current_price - avg_price) * total_qty
        unrealized_pnl_pct = (unrealized_pnl / active_position["total_cost"]) * 100

        # DCA Layer Add: STRICTLY MAX 2 LAYERS (Only 1 DCA layer allowed!)
        if len(layers) < 2 and current_price <= active_position["last_layer_price"] * 0.985:
            if ind["rsi"] < 40:  # Only average if genuinely oversold
                with state_lock:
                    balance = bot_state["current_balance"]
                
                next_layer_usd = balance * (risk_pct / 100)
                next_layer_qty = round(next_layer_usd / current_price, 4)
                
                order = place_market_order(symbol, "BUY", next_layer_usd, mode)
                if order and ("orderId" in order or mode.lower() == "paper"):
                    layers.append({
                        "layer": len(layers) + 1,
                        "price": current_price,
                        "qty": next_layer_qty,
                        "cost": next_layer_usd,
                        "time": datetime.now().strftime("%H:%M:%S")
                    })
                    avg_p, tot_q, tot_c, tp_p, sl_p = calculate_dca_levels(layers)
                    active_position.update({
                        "layers": layers, "avg_price": avg_p,
                        "total_qty": tot_q, "total_cost": tot_c,
                        "tp_price": tp_p, "sl_price": sl_p,
                        "last_layer_price": current_price
                    })
                    logger.info(f"📈 Final DCA Layer 2 Executed @ ${current_price}")

        # PROFITABLE EXIT CHECKS
        exit_reason = None
        
        # 1. Main TP Hit (+0.8% or +0.5%)
        if current_price >= tp_price:
            exit_reason = "TAKE_PROFIT_HIT"
        # 2. Hard Stop Loss Hit
        elif current_price <= sl_price:
            exit_reason = "SL_HIT"
        # 3. Dynamic Profit Exit: In solid profit (+0.5%+) and RSI is overbought (Reversal)
        elif unrealized_pnl_pct >= 0.5 and ind["rsi"] >= 68:
            exit_reason = "OVERBOUGHT_PROFIT_EXIT"

        if exit_reason:
            exit_price = current_price
            gross_pnl = (exit_price - avg_price) * total_qty
            trading_fee = (active_position["total_cost"] + (total_qty * exit_price)) * 0.001
            net_pnl = gross_pnl - trading_fee
            outcome = f"{exit_reason} | {'WIN ✅' if net_pnl > 0 else 'LOSS ❌'}"
            
            order = place_market_order(symbol, "SELL", total_qty * exit_price, mode)
            if order or mode.lower() == "paper":
                save_trade_to_db(
                    active_position["entry_time"],
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    symbol, "LONG", avg_price, exit_price, len(layers),
                    tp_price, sl_price, ind["rsi"], ind["macd"], tech_conf,
                    net_pnl, outcome
                )
                
                with state_lock:
                    if net_pnl > 0:
                        bot_state["wins"] += 1
                        bot_state["consecutive_losses"] = 0
                    else:
                        bot_state["losses"] += 1
                        bot_state["consecutive_losses"] += 1
                    
                    bot_state["total_trades"] += 1
                    bot_state["current_profit"] += net_pnl
                    if mode.lower() == "paper":
                        bot_state["current_balance"] += net_pnl
                
                active_position = None
                last_trade_time = current_time
                logger.info(f"🎯 Position Closed: {outcome} | Net PnL: ${net_pnl:.2f}")
        else:
            with state_lock:
                bot_state["status_message"] = (
                    f"SCALP ({len(layers)}/2) | Avg: ${avg_price:.2f} | "
                    f"TP: ${tp_price:.2f} | SL: ${sl_price:.2f} | PnL: ${unrealized_pnl:.2f}"
                )

    # ===== LOOK FOR ENTRY =====
    else:
        cooldown = TRADE_COOLDOWN_SECONDS
        if (current_time - last_trade_time) < cooldown:
            with state_lock:
                bot_state["status_message"] = f"Cooldown: {int(cooldown - (current_time - last_trade_time))}s"
        else:
            with state_lock:
                bot_state["status_message"] = f"Scanning... | Signal: {tech_signal} ({tech_conf:.0f}%) | RSI: {ind['rsi']}"

            # Only enter on strong setup
            if final_signal == "LONG" and tech_conf >= 70:
                with state_lock:
                    balance = bot_state["current_balance"]
                
                entry_usd = max(10.0, balance * (risk_pct / 100))
                entry_qty = round(entry_usd / current_price, 4)
                
                if entry_qty > 0 and balance >= 10:
                    order = place_market_order(symbol, "BUY", entry_usd, mode)
                    if order and ("orderId" in order or mode.lower() == "paper"):
                        entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        layers = [{
                            "layer": 1,
                            "price": current_price,
                            "qty": entry_qty,
                            "cost": entry_usd,
                            "time": datetime.now().strftime("%H:%M:%S")
                        }]
                        avg_p, tot_q, tot_c, tp_p, sl_p = calculate_dca_levels(layers)
                        active_position = {
                            "entry_time": entry_time,
                            "layers": layers,
                            "max_layers": 2,
                            "avg_price": avg_p,
                            "total_qty": tot_q,
                            "total_cost": tot_c,
                            "tp_price": tp_p,
                            "sl_price": sl_p,
                            "last_layer_price": current_price,
                            "ml_signal": final_signal,
                            "ml_conf": tech_conf
                        }
                        logger.info(f"🚀 SCALP 1/2 OPENED @ ${current_price} | TP: ${tp_p}")

# ============================================
# 🔄 WORKER THREAD
# ============================================
def bot_worker():
    logger.info("🚀 Scalper Market Worker Started!")
    loop_count = 0
    while True:
        try:
            with state_lock:
                symbol = bot_state["symbol"]
                mode = bot_state["mode"]
                risk_pct = bot_state["risk_pct"]
            
            process_bot_logic(symbol, mode, risk_pct)
            
            loop_count += 1
            if loop_count % 30 == 0:
                gc.collect()

        except Exception as e:
            logger.error(f"⚠️ Worker loop caught error: {e}")
            logger.error(traceback.format_exc())
            time.sleep(5)
        
        time.sleep(4)

def start_worker_safely():
    global worker_thread
    with state_lock:
        if worker_thread is None or not worker_thread.is_alive():
            worker_thread = threading.Thread(target=bot_worker, daemon=True)
            worker_thread.start()
            logger.info("🚀 Worker Thread (Re)started Successfully!")

init_db()
load_db_history()

# ============================================
# 🌐 FLASK API ENDPOINTS
# ============================================
@app.route("/")
def index():
    start_worker_safely()
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    except FileNotFoundError:
        return "index.html not found", 404

@app.route("/api/start", methods=["POST"])
def start_bot():
    start_worker_safely()
    data = request.json or {}
    with state_lock:
        mode_val = data.get("mode", "paper").lower()
        bot_state["mode"] = mode_val.upper()
        bot_state["trading_mode"] = mode_val.upper()
        bot_state["trade_mode"] = mode_val.upper()
        bot_state["tradingMode"] = mode_val.upper()
        bot_state["tradeMode"] = mode_val.upper()
        bot_state["mode_type"] = mode_val.upper()
        bot_state["mode_name"] = mode_val.upper()
        bot_state["symbol"] = data.get("symbol", "ETHUSDT")
        bot_state["risk_pct"] = float(data.get("risk_pct", 10.0))
        if mode_val == "paper":
            init_bal = float(data.get("start_balance", 1000.0))
            bot_state["virtual_balance"] = init_bal
            bot_state["current_balance"] = init_bal
        bot_state["is_running"] = True
    logger.info(f"▶️ Bot Started in {bot_state['mode']} mode")
    return jsonify({"status": "success", "message": f"Bot started in {bot_state['mode']} mode!"})

@app.route("/api/stop", methods=["POST"])
def stop_bot():
    with state_lock:
        bot_state["is_running"] = False
    logger.info("⏹️ Bot Stopped")
    return jsonify({"status": "success", "message": "Bot stopped"})

@app.route("/api/reset_demo", methods=["POST"])
def reset_demo():
    global active_position
    data = request.json or {}
    init_bal = float(data.get("start_balance", 1000.0))
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM trade_history")
        conn.commit()
        conn.close()
        logger.info("🗑️ Demo trade history cleared")
    except Exception as e:
        logger.error(f"Reset error: {e}")
    
    with state_lock:
        active_position = None
        bot_state["current_balance"] = init_bal
        bot_state["wins"] = 0
        bot_state["losses"] = 0
        bot_state["total_trades"] = 0
        bot_state["accuracy"] = 0.0
        bot_state["current_profit"] = 0.0
        bot_state["consecutive_losses"] = 0
        bot_state["active_trades"] = []
    
    return jsonify({"status": "success", "message": f"Demo reset to ${init_bal}"})

@app.route("/api/status")
def get_status():
    start_worker_safely()
    with state_lock:
        active_pos_data = None
        available_balance = bot_state["current_balance"]
        
        if active_position:
            unrealized_pnl = (bot_state["current_price"] - active_position["avg_price"]) * active_position["total_qty"]
            active_pos_data = {
                "entry_time": active_position["entry_time"],
                "layers": active_position["layers"],
                "layers_count": len(active_position["layers"]),
                "max_layers": 2,
                "total_layers": 2,
                "layers_max": 2,
                "avg_price": round(active_position["avg_price"], 2),
                "entry_price": round(active_position["avg_price"], 2),
                "entryPrice": round(active_position["avg_price"], 2),
                "total_qty": round(active_position["total_qty"], 4),
                "tp_price": round(active_position["tp_price"], 2),
                "sl_price": round(active_position["sl_price"], 2),
                "unrealized_pnl": round(unrealized_pnl, 2)
            }
            if bot_state["mode"].lower() == "paper":
                available_balance = max(0, bot_state["current_balance"] - active_position["total_cost"])
        
        win_rate = 0
        if bot_state["total_trades"] > 0:
            win_rate = round((bot_state["wins"] / bot_state["total_trades"] * 100), 1)
        
        return jsonify({
            "is_running": bot_state["is_running"],
            "mode": bot_state["mode"],
            "trading_mode": bot_state["trading_mode"],
            "trade_mode": bot_state["trade_mode"],
            "tradingMode": bot_state["tradingMode"],
            "tradeMode": bot_state["tradeMode"],
            "mode_type": bot_state["mode_type"],
            "mode_name": bot_state["mode_name"],
            "symbol": bot_state["symbol"],
            "risk_pct": bot_state["risk_pct"],
            "max_layers": 2,
            "total_layers": 2,
            "layers_max": 2,
            "balance": round(available_balance, 2),
            "current_profit": round(bot_state["current_profit"], 2),
            "total_trades": bot_state["total_trades"],
            "wins": bot_state["wins"],
            "losses": bot_state["losses"],
            "win_rate": win_rate,
            "current_price": round(bot_state["current_price"], 2),
            "rsi": bot_state["current_rsi"],
            "ema_20": bot_state["current_ema_20"],
            "ema_50": bot_state["current_ema_50"],
            "ema_200": bot_state["current_ema_200"],
            "macd": bot_state["current_macd"],
            "macd_signal": bot_state["current_macd_signal"],
            "atr": bot_state["current_atr"],
            "volatility": bot_state["volatility"],
            "status_message": bot_state["status_message"],
            "ml_signal": bot_state["ml_signal"],
            "ml_confidence": bot_state["ml_confidence"],
            "technical_signal": bot_state["technical_signal"],
            "final_signal": bot_state["final_signal"],
            "candles": bot_state["candles"][-80:],
            "ema_20_series": bot_state["ema_20_series"][-80:],
            "ema_50_series": bot_state["ema_50_series"][-80:],
            "ema_200_series": bot_state["ema_200_series"][-80:],
            "active_trades": bot_state["active_trades"],
            "active_position": active_pos_data
        })

@app.route("/api/export_csv")
def export_csv():
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("""
            SELECT timestamp, symbol, side, entry_price, exit_price, layers, 
                   tp, sl, pnl, outcome FROM trade_history ORDER BY id DESC
        """, conn)
        conn.close()
        csv_data = df.to_csv(index=False)
        return csv_data, 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': 'attachment; filename=trade_history.csv'
        }
    except Exception as e:
        return str(e), 500

# ============================================
# ▶️ ENTRY POINT
# ============================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
