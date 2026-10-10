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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# ============================================
# 🔒 CONFIGURATION & LOGGING
# ============================================
load_dotenv()

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
    "mode": "paper",
    "symbol": "ETHUSDT",
    "leverage": 1,
    "risk_pct": 10.0,
    "virtual_balance": 1000.0,
    "current_balance": 1000.0,
    "current_price": 0.0,
    "status_message": "Bot සූදානම්ව පවතී...",
    
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
    
    # AI signals
    "ml_signal": "NEUTRAL",
    "ml_confidence": 50.0,
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
worker_thread_started = False

# AI Model Cache in Memory (No repeated fits)
ml_model = None
ml_scaler = None
last_model_train_time = 0

# ============================================
# 🗄️ SAFE DATABASE HANDLERS
# ============================================
def get_db_connection():
    """Returns sqlite connection with safety timeout"""
    conn = sqlite3.connect(DB_FILE, timeout=20.0)
    return conn

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
        logger.info("✅ Database initialized safely!")
    except Exception as e:
        logger.error(f"❌ Database init error: {e}")

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
            history_trades.append({
                "time": timestamp,
                "type": f"{side}",
                "entry": entry_price,
                "exit": exit_price,
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

    except Exception as e:
        logger.error(f"❌ Load history error: {e}")

def save_trade_to_db(entry_time, exit_time, symbol, side, entry_price, exit_price, 
                     layers, tp, sl, rsi, macd, ml_conf, pnl, outcome):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trade_history 
            (timestamp, exit_time, symbol, side, entry_price, exit_price, layers, 
             tp, sl, rsi, macd, ml_confidence, pnl, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (entry_time, exit_time, symbol, side, entry_price, exit_price, layers, 
              tp, sl, rsi, macd, ml_conf, pnl, outcome))
        conn.commit()
        conn.close()
        load_db_history()
        logger.info(f"💾 Trade Saved: {outcome} | PnL: ${pnl:.2f}")
    except Exception as e:
        logger.error(f"❌ Save trade error: {e}")

# ============================================
# 🧠 LOW-RAM ML MODEL (NO CONTINUOUS FITTING)
# ============================================
FEATURE_COLS = ['rsi', 'price_momentum', 'ema_diff', 'macd_hist', 'volatility', 'atr']

def train_or_update_model(df):
    """Trains a ultra-lightweight model only once every 4 hours"""
    global ml_model, ml_scaler, last_model_train_time
    now = time.time()
    
    # Only train if model doesn't exist or 4 hours elapsed
    if ml_model is not None and (now - last_model_train_time < 14400):
        return

    try:
        clean_df = df[FEATURE_COLS + ['target']].dropna().copy()
        if len(clean_df) < 80:
            return

        X = clean_df[FEATURE_COLS]
        y = clean_df['target']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Ultra-light model: 30 trees, max depth 4 (takes < 15MB RAM)
        model = RandomForestClassifier(
            n_estimators=30,
            max_depth=4,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=1
        )
        model.fit(X_scaled, y)

        ml_model = model
        ml_scaler = scaler
        last_model_train_time = now
        logger.info("🧠 AI Model loaded & cached safely in memory.")
        
        # Free memory
        del clean_df, X, y, X_scaled
        gc.collect()
    except Exception as e:
        logger.error(f"❌ Model training error: {e}")

def predict_ml(df):
    """Fast prediction using cached model without memory leak"""
    global ml_model, ml_scaler
    if ml_model is None or ml_scaler is None:
        train_or_update_model(df)
        if ml_model is None:
            return "NEUTRAL", 50.0

    try:
        latest = df[FEATURE_COLS].iloc[-1:].values
        if np.isnan(latest).any():
            return "NEUTRAL", 50.0

        latest_scaled = ml_scaler.transform(latest)
        prob_up = ml_model.predict_proba(latest_scaled)[0][1] * 100

        if prob_up >= 52.0:
            return "LONG", round(prob_up, 1)
        elif prob_up <= 45.0:
            return "SHORT", round(100 - prob_up, 1)
        else:
            return "NEUTRAL", 50.0
    except Exception:
        return "NEUTRAL", 50.0

# ============================================
# 📊 TECHNICAL STRATEGY
# ============================================
def get_technical_signal(ind):
    price = ind["current_price"]
    ema20 = ind["ema_20"]
    ema50 = ind["ema_50"]
    macd_bullish = ind["macd"] > ind["macd_signal"]
    rsi = ind["rsi"]

    # Short-term momentum check
    trend_up = (price >= ema20) and (ema20 >= ema50)
    pullback_bounce = (rsi < 48) and macd_bullish and (price >= ema20 * 0.997)

    if trend_up and macd_bullish and (38 <= rsi <= 70):
        return "LONG", 70.0

    if pullback_bounce and (35 <= rsi <= 55):
        return "LONG", 60.0

    if not macd_bullish and rsi < 38:
        return "SHORT", 60.0

    return "NEUTRAL", 50.0

# ============================================
# 📊 SAFE BINANCE FETCHER (ANTI-BLOCKING)
# ============================================
def get_klines_and_indicators(symbol):
    try:
        params = {"symbol": symbol, "interval": TIMEFRAME, "limit": 200}
        # Connection timeout: 3.5s, Read timeout: 5s
        response = requests.get(
            "https://api.binance.com/api/v3/klines",
            params=params,
            timeout=(3.5, 5.0)
        )
        
        if response.status_code == 429:
            logger.warning("⚠️ Binance Rate limit warning (429)! Cooling down...")
            time.sleep(10)
            return None, None, None

        if response.status_code != 200:
            return None, None, None
        
        data = response.json()
        if not data or len(data) < 100:
            return None, None, None
        
        df = pd.DataFrame(data, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
        ])
        
        df['time'] = (df['time'].astype(int) / 1000).astype(int)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Fast indicator computation
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
        
        # Features for AI
        df['price_momentum'] = (close - close.shift(4)) / close.shift(4) * 100
        df['ema_diff'] = (df['ema_20'] - df['ema_50']) / close * 100
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['target'] = (close.shift(-1) > close).astype(int)

        # Build lightweight candle slice for chart
        chart_candles = []
        ema_20_list, ema_50_list, ema_200_list = [], [], []
        tail_df = df.tail(80)  # Keep payload small

        for idx, row in tail_df.iterrows():
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
        
        return chart_candles, indicators, df
    except requests.exceptions.RequestException as e:
        logger.warning(f"⚠️ Network timeout/error: {e}")
        return None, None, None
    except Exception as e:
        logger.error(f"❌ Indicator parse error: {e}")
        return None, None, None

# ============================================
# 💹 POSITION MANAGEMENT
# ============================================
def calculate_dca_levels(layers):
    if not layers:
        return None
    total_qty = sum(l["qty"] for l in layers)
    total_cost = sum(l["cost"] for l in layers)
    avg_price = total_cost / total_qty if total_qty > 0 else 0.0

    if len(layers) == 1:
        tp_ratio = 1.020  # +2.0%
        sl_ratio = 0.980  # -2.0%
    elif len(layers) == 2:
        tp_ratio = 1.015  # +1.5%
        sl_ratio = 0.975  # -2.5%
    else:
        tp_ratio = 1.012
        sl_ratio = 0.970

    return (round(avg_price, 4), round(total_qty, 4), round(total_cost, 2), 
            round(avg_price * tp_ratio, 4), round(avg_price * sl_ratio, 4))

def place_market_order(symbol, side, quote_qty, mode):
    if mode == "paper":
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
    
    candles, ind, df = get_klines_and_indicators(symbol)
    if not ind or df is None:
        return
    
    current_time = time.time()
    current_price = ind["current_price"]
    
    # Train model safely in background if needed
    train_or_update_model(df)
    
    ml_signal, ml_conf = predict_ml(df)
    tech_signal, tech_conf = get_technical_signal(ind)
    
    final_signal = "NEUTRAL"
    if tech_signal == "LONG" and (ml_signal == "LONG" or ml_conf >= 50.0):
        final_signal = "LONG"
    
    # Update state for dashboard
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
        bot_state["ml_signal"] = ml_signal
        bot_state["ml_confidence"] = ml_conf
        bot_state["technical_signal"] = tech_signal
        bot_state["final_signal"] = final_signal

    if not bot_state["is_running"]:
        return

    # ===== MANAGE ACTIVE TRADE =====
    if active_position:
        layers = active_position["layers"]
        avg_price = active_position["avg_price"]
        total_qty = active_position["total_qty"]
        tp_price = active_position["tp_price"]
        sl_price = active_position["sl_price"]
        
        unrealized_pnl = (current_price - avg_price) * total_qty
        
        # Check DCA Layer buy
        if len(layers) < 3 and current_price <= active_position["last_layer_price"] * 0.985:
            if ind["rsi"] < 45:
                with state_lock:
                    balance = bot_state["current_balance"]
                
                next_layer_usd = balance * (risk_pct / 100)
                next_layer_qty = round(next_layer_usd / current_price, 4)
                
                order = place_market_order(symbol, "BUY", next_layer_usd, mode)
                if order and ("orderId" in order or mode == "paper"):
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
                    logger.info(f"📈 DCA Layer {len(layers)} Executed @ ${current_price}")
        
        # Check Exit conditions
        exit_reason = None
        if current_price >= tp_price:
            exit_reason = "TP_HIT"
        elif current_price <= sl_price:
            exit_reason = "SL_HIT"
        
        if exit_reason:
            exit_price = current_price
            gross_pnl = (exit_price - avg_price) * total_qty
            trading_fee = (active_position["total_cost"] + (total_qty * exit_price)) * 0.001
            net_pnl = gross_pnl - trading_fee
            outcome = f"{exit_reason} | {'WIN ✅' if net_pnl > 0 else 'LOSS ❌'}"
            
            order = place_market_order(symbol, "SELL", total_qty * exit_price, mode)
            if order or mode == "paper":
                save_trade_to_db(
                    active_position["entry_time"],
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    symbol, "LONG", avg_price, exit_price, len(layers),
                    tp_price, sl_price, ind["rsi"], ind["macd"], ml_conf,
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
                    if mode == "paper":
                        bot_state["current_balance"] += net_pnl
                
                active_position = None
                last_trade_time = current_time
                logger.info(f"🎯 Position Closed: {outcome} | PnL: ${net_pnl:.2f}")
        else:
            with state_lock:
                bot_state["status_message"] = (
                    f"SPOT DCA (LONG {len(layers)}/3) | Avg: ${avg_price} | "
                    f"TP: ${tp_price} | SL: ${sl_price} | PnL: ${unrealized_pnl:.2f}"
                )

    # ===== LOOK FOR ENTRY =====
    else:
        cooldown = TRADE_COOLDOWN_SECONDS
        if (current_time - last_trade_time) < cooldown:
            with state_lock:
                bot_state["status_message"] = f"Cooldown: {int(cooldown - (current_time - last_trade_time))}s"
        else:
            with state_lock:
                bot_state["status_message"] = f"Scanning... | Tech: {tech_signal} ({tech_conf:.0f}%) | AI: {ml_signal} ({ml_conf:.0f}%)"
            
            if final_signal == "LONG" and tech_conf >= 60 and ml_conf >= 50:
                with state_lock:
                    balance = bot_state["current_balance"]
                
                entry_usd = max(10.0, balance * (risk_pct / 100))
                entry_qty = round(entry_usd / current_price, 4)
                
                if entry_qty > 0 and balance >= 10:
                    order = place_market_order(symbol, "BUY", entry_usd, mode)
                    if order and ("orderId" in order or mode == "paper"):
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
                            "avg_price": avg_p,
                            "total_qty": tot_q,
                            "total_cost": tot_c,
                            "tp_price": tp_p,
                            "sl_price": sl_p,
                            "last_layer_price": current_price,
                            "ml_signal": ml_signal,
                            "ml_conf": ml_conf
                        }
                        logger.info(f"🚀 New Position opened @ ${current_price}")

# ============================================
# 🔄 BULLETPROOF WORKER LOOP (NO CRASHES)
# ============================================
def bot_worker():
    """Persistent, self-healing background worker"""
    logger.info("🚀 Background Trading Worker Started Successfully!")
    loop_count = 0
    while True:
        try:
            with state_lock:
                symbol = bot_state["symbol"]
                mode = bot_state["mode"]
                risk_pct = bot_state["risk_pct"]
                is_running = bot_state["is_running"]
            
            if is_running:
                process_bot_logic(symbol, mode, risk_pct)
            
            loop_count += 1
            # Clean RAM every 20 loops (prevents Render 512MB RAM leaks)
            if loop_count % 20 == 0:
                gc.collect()

        except Exception as e:
            logger.error(f"⚠️ Recovered from worker loop error: {e}")
            logger.error(traceback.format_exc())
            time.sleep(5)
        
        time.sleep(6)  # Safe 6s delay (Prevents Binance 429 block)

def start_worker_safely():
    global worker_thread_started
    if not worker_thread_started:
        with state_lock:
            if not worker_thread_started:
                t = threading.Thread(target=bot_worker, daemon=True)
                t.start()
                worker_thread_started = True

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
        bot_state["mode"] = data.get("mode", "paper")
        bot_state["symbol"] = data.get("symbol", "ETHUSDT")
        bot_state["risk_pct"] = float(data.get("risk_pct", 10.0))
        if bot_state["mode"] == "paper":
            init_bal = float(data.get("start_balance", 1000.0))
            bot_state["virtual_balance"] = init_bal
            bot_state["current_balance"] = init_bal
        bot_state["is_running"] = True
    logger.info(f"▶️ Bot started: {bot_state['mode']}")
    return jsonify({"status": "success", "message": f"Bot started in {bot_state['mode']} mode!"})

@app.route("/api/stop", methods=["POST"])
def stop_bot():
    with state_lock:
        bot_state["is_running"] = False
    logger.info("⏹️ Bot stopped")
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
        logger.info("🗑️ Trade history cleared")
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
                "avg_price": active_position["avg_price"],
                "total_qty": active_position["total_qty"],
                "tp_price": active_position["tp_price"],
                "sl_price": active_position["sl_price"],
                "unrealized_pnl": round(unrealized_pnl, 2)
            }
            if bot_state["mode"] == "paper":
                available_balance = max(0, bot_state["current_balance"] - active_position["total_cost"])
        
        win_rate = 0
        if bot_state["total_trades"] > 0:
            win_rate = round((bot_state["wins"] / bot_state["total_trades"] * 100), 1)
        
        return jsonify({
            "is_running": bot_state["is_running"],
            "mode": bot_state["mode"],
            "symbol": bot_state["symbol"],
            "risk_pct": bot_state["risk_pct"],
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
    init_db()
    load_db_history()
    logger.info("=" * 60)
    logger.info("🤖 BULLETPROOF ANTI-FREEZE DCA BOT READY")
    logger.info("=" * 60)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
