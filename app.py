import os
import time
import hmac
import hashlib
import urllib.parse
import threading
import json
import sqlite3
import logging
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# ============================================
# 🔒 CONFIGURATION & SECURITY
# ============================================
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Binance API Credentials
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Bot configuration
DB_FILE = "bot_memory.db"
TRADE_COOLDOWN_SECONDS = 15

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
    "current_rsi": 0.0,
    "current_ema_20": 0.0,
    "current_ema_50": 0.0,
    "current_ema_200": 0.0,
    "current_macd": 0.0,
    "current_macd_signal": 0.0,
    "current_atr": 0.0,
    "volatility": 0.0,
    
    # AI signals
    "ml_signal": "NEUTRAL",
    "ml_confidence": 0.0,
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

# ============================================
# 🗄️ DATABASE
# ============================================
def init_db():
    """Initialize SQLite database"""
    try:
        conn = sqlite3.connect(DB_FILE)
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
        logger.info("✅ Database initialized!")
    except Exception as e:
        logger.error(f"❌ Database error: {e}")

def load_db_history():
    """Load trade history"""
    try:
        conn = sqlite3.connect(DB_FILE)
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

        logger.info(f"📊 Loaded {len(history_trades)} trades | Win Rate: {acc}%")
        
    except Exception as e:
        logger.error(f"❌ Load history error: {e}")

def save_trade_to_db(entry_time, exit_time, symbol, side, entry_price, exit_price, 
                     layers, tp, sl, rsi, macd, ml_conf, pnl, outcome):
    """Save trade"""
    try:
        conn = sqlite3.connect(DB_FILE)
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
        logger.info(f"💾 Trade: {outcome} | PnL: ${pnl:.2f}")
    except Exception as e:
        logger.error(f"❌ Save trade error: {e}")

# ============================================
# 🧠 IMPROVED ML MODEL (RELAXED THRESHOLD)
# ============================================
def train_and_predict_ml(df):
    """Pure ML prediction without external API"""
    try:
        if len(df) < 150:
            return "NEUTRAL", 50.0

        # Features
        df['price_momentum'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100
        df['rsi_trend'] = df['rsi'].diff()
        df['ema_alignment'] = (df['ema_20'] - df['ema_50']) / df['close'] * 100
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['volatility'] = df['close'].rolling(10).std() / df['close'] * 100
        
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        features = ['rsi', 'price_momentum', 'rsi_trend', 'ema_alignment', 
                   'macd_histogram', 'volatility', 'atr']
        
        clean_df = df[features + ['target']].dropna().copy()
        
        if len(clean_df) < 100:
            return "NEUTRAL", 50.0

        # Train/test split
        train_size = int(len(clean_df) * 0.8)
        X_train = clean_df[features][:train_size]
        y_train = clean_df['target'][:train_size]
        X_test = clean_df[features][train_size:]
        y_test = clean_df['target'][train_size:]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Predict
        latest_features = clean_df[features].iloc[-1:].values
        latest_scaled = scaler.transform(latest_features)
        probs = model.predict_proba(latest_scaled)[0]
        
        prob_up = probs[1] * 100
        
        # 🟢 RELAXED: 55% -> 51%
        if prob_up >= 51.0:
            return "LONG", round(prob_up, 1)
        elif prob_up <= 49.0:
            return "SHORT", round(100 - prob_up, 1)
        else:
            return "NEUTRAL", 50.0

    except Exception as e:
        logger.error(f"❌ ML error: {e}")
        return "NEUTRAL", 50.0

# ============================================
# 📊 TECHNICAL SIGNALS (RELAXED CONDITIONS)
# ============================================
def get_technical_signal(ind, df):
    """Generate signal purely from technical indicators (Relaxed for faster trades)"""
    
    # Short-term trends
    price_above_ema20 = ind["current_price"] >= ind["ema_20"]
    ema_short_bullish = ind["ema_20"] >= ind["ema_50"]
    
    # Momentum
    macd_bullish = ind["macd"] > ind["macd_signal"]
    
    # RSI filter
    rsi_ok = 35 <= ind["rsi"] <= 68
    rsi_bounce = (ind["rsi"] < 45) and macd_bullish
    
    # ===== LONG SIGNALS =====
    # 1. Price above EMA 20 with Bullish MACD & Safe RSI
    if price_above_ema20 and macd_bullish and rsi_ok:
        return "LONG", 70.0
    
    # 2. Short-term EMA Bullish Alignment
    if ema_short_bullish and macd_bullish and rsi_ok:
        return "LONG", 60.0
    
    # 3. Pullback / Oversold bounce signal
    if rsi_bounce:
        return "LONG", 55.0
    
    # Bearish check
    if not macd_bullish and ind["rsi"] < 40:
        return "SHORT", 60.0
    
    return "NEUTRAL", 50.0

# ============================================
# 📊 TECHNICAL ANALYSIS
# ============================================
def get_klines_and_indicators(symbol):
    """Fetch and calculate indicators"""
    try:
        params = {"symbol": symbol, "interval": "3m", "limit": 300}
        response = requests.get(
            "https://api.binance.com/api/v3/klines",
            params=params,
            timeout=5
        )
        
        if response.status_code != 200:
            logger.error(f"❌ Klines error: {response.status_code}")
            return None, None, None
        
        data = response.json()
        if not data or len(data) < 200:
            logger.warning("⚠️ Insufficient data")
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
        
        # EMAs
        df['ema_20'] = close.ewm(span=20, adjust=False).mean()
        df['ema_50'] = close.ewm(span=50, adjust=False).mean()
        df['ema_200'] = close.ewm(span=200, adjust=False).mean()
        
        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        ema_gain = gain.ewm(com=13, adjust=False).mean()
        ema_loss = loss.ewm(com=13, adjust=False).mean()
        rs = ema_gain / (ema_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # ATR
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        
        # Volatility
        df['volatility'] = close.pct_change().rolling(window=20).std() * 100
        
        # Chart data
        chart_candles = []
        ema_20_list = []
        ema_50_list = []
        ema_200_list = []
        
        for i in range(len(df)):
            t = int(df['time'].iloc[i])
            chart_candles.append({
                "time": t,
                "open": float(df['open'].iloc[i]),
                "high": float(df['high'].iloc[i]),
                "low": float(df['low'].iloc[i]),
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
            "volatility": round(float(df['volatility'].iloc[-1]), 2),
            "current_price": float(close.iloc[-1]),
            "ema_20_series": ema_20_list[-100:],
            "ema_50_series": ema_50_list[-100:],
            "ema_200_series": ema_200_list[-100:]
        }
        
        return chart_candles[-100:], indicators, df
        
    except Exception as e:
        logger.error(f"❌ Klines error: {e}")
        return None, None, None

# ============================================
# 💹 POSITION MANAGEMENT
# ============================================
def calculate_dca_levels(layers):
    """Calculate DCA levels"""
    if not layers:
        return None
    
    total_qty = sum(l["qty"] for l in layers)
    total_cost = sum(l["cost"] for l in layers)
    avg_price = total_cost / total_qty if total_qty > 0 else 0.0
    
    if len(layers) == 1:
        tp_ratio = 1.025
        sl_ratio = 0.985
    elif len(layers) == 2:
        tp_ratio = 1.020
        sl_ratio = 0.980
    else:
        tp_ratio = 1.015
        sl_ratio = 0.975
    
    tp_price = round(avg_price * tp_ratio, 4)
    sl_price = round(avg_price * sl_ratio, 4)
    
    return round(avg_price, 4), round(total_qty, 4), round(total_cost, 2), tp_price, sl_price

def place_market_order(symbol, side, quote_qty, mode):
    """Place order"""
    if mode == "paper":
        return {"orderId": "PAPER_" + str(int(time.time())), "status": "FILLED"}
    
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        logger.error("❌ API credentials missing")
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
        
        response = requests.post(
            "https://api.binance.com/api/v3/order",
            params=params,
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 200:
            logger.info(f"✅ Order: {side} {quote_qty} USDT")
            return response.json()
        else:
            logger.error(f"❌ Order failed: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Order error: {e}")
        return None

# ============================================
# 🎯 MAIN TRADING LOGIC
# ============================================
def process_bot_logic(symbol, mode, risk_pct):
    """Main bot logic"""
    global active_position, last_trade_time
    
    candles, ind, df = get_klines_and_indicators(symbol)
    if not ind or df is None:
        return
    
    current_time = time.time()
    current_price = ind["current_price"]
    
    # Signals
    ml_signal, ml_conf = train_and_predict_ml(df)
    tech_signal, tech_conf = get_technical_signal(ind, df)
    
    # Final signal - both agree
    final_signal = "NEUTRAL"
    if ml_signal == "LONG" and tech_signal == "LONG":
        final_signal = "LONG"
    
    # Update state
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
    
    # ===== MANAGE EXISTING POSITION =====
    if active_position:
        layers = active_position["layers"]
        avg_price = active_position["avg_price"]
        total_qty = active_position["total_qty"]
        tp_price = active_position["tp_price"]
        sl_price = active_position["sl_price"]
        
        unrealized_pnl = (current_price - avg_price) * total_qty
        unrealized_pnl_pct = (unrealized_pnl / active_position["total_cost"]) * 100
        
        # DCA Layer add
        if len(layers) < 3 and current_price <= active_position["last_layer_price"] * 0.985:
            if ind["rsi"] < 45:
                with state_lock:
                    balance = bot_state["current_balance"]
                
                next_layer_usd = balance * (risk_pct / 100)
                next_layer_qty = round(next_layer_usd / current_price, 4)
                
                order = place_market_order(symbol, "BUY", next_layer_usd, mode)
                if order and "orderId" in order:
                    layers.append({
                        "layer": len(layers) + 1,
                        "price": current_price,
                        "qty": next_layer_qty,
                        "cost": next_layer_usd,
                        "time": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    avg_price, total_qty, total_cost, tp_price, sl_price = calculate_dca_levels(layers)
                    active_position["layers"] = layers
                    active_position["avg_price"] = avg_price
                    active_position["total_qty"] = total_qty
                    active_position["total_cost"] = total_cost
                    active_position["tp_price"] = tp_price
                    active_position["sl_price"] = sl_price
                    active_position["last_layer_price"] = current_price
                    
                    logger.info(f"📈 Layer {len(layers)} @ ${current_price}")
        
        # Check exit
        exit_reason = None
        exit_price = None
        
        if current_price >= tp_price:
            exit_reason = "TP_HIT"
            exit_price = tp_price
        elif current_price <= sl_price:
            exit_reason = "SL_HIT"
            exit_price = sl_price
        elif unrealized_pnl_pct < -3 and ind["rsi"] > 75:
            exit_reason = "REVERSAL"
            exit_price = current_price
        
        if exit_reason:
            gross_pnl = (exit_price - avg_price) * total_qty
            trading_fee = active_position["total_cost"] * 0.002
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
                logger.info(f"🎯 Position closed: {outcome} | PnL: ${net_pnl:.2f}")
        
        else:
            status_msg = f"LONG {len(layers)}/3 | Avg: ${avg_price} | TP: ${tp_price} | SL: ${sl_price} | U/R: ${unrealized_pnl:.2f}"
            with state_lock:
                bot_state["status_message"] = status_msg
    
    # ===== LOOK FOR ENTRY =====
    else:
        cooldown = 15
        time_since_trade = current_time - last_trade_time
        
        if time_since_trade < cooldown:
            with state_lock:
                bot_state["status_message"] = f"Cooldown: {int(cooldown - time_since_trade)}s"
        else:
            with state_lock:
                bot_state["status_message"] = f"Waiting... | Tech: {tech_signal} {tech_conf:.0f}% | ML: {ml_signal} {ml_conf:.0f}%"
            
            # 🟢 RELAXED ENTRY: ML >= 51 and Tech >= 55
            if final_signal == "LONG" and ml_conf >= 51 and tech_conf >= 55:
                logger.info(f"🚀 Entry Triggered: LONG | ML: {ml_conf}% | Tech: {tech_conf}%")
                
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
                        
                        avg_price, total_qty, total_cost, tp_price, sl_price = calculate_dca_levels(layers)
                        
                        active_position = {
                            "entry_time": entry_time,
                            "layers": layers,
                            "avg_price": avg_price,
                            "total_qty": total_qty,
                            "total_cost": total_cost,
                            "tp_price": tp_price,
                            "sl_price": sl_price,
                            "last_layer_price": current_price,
                            "ml_signal": ml_signal,
                            "ml_conf": ml_conf
                        }
                        
                        with state_lock:
                            bot_state["status_message"] = f"ENTRY @ ${current_price} | TP: ${tp_price} | SL: ${sl_price}"
                        
                        logger.info(f"✅ Position opened successfully!")

def bot_worker():
    """Worker thread"""
    while True:
        try:
            with state_lock:
                symbol = bot_state["symbol"]
                mode = bot_state["mode"]
                risk_pct = bot_state["risk_pct"]
                is_running = bot_state["is_running"]
            
            if is_running:
                process_bot_logic(symbol, mode, risk_pct)
            
        except Exception as e:
            logger.error(f"❌ Bot cycle error: {e}")
        
        time.sleep(3)

def start_worker_safely():
    global worker_thread_started
    if not worker_thread_started:
        with state_lock:
            if not worker_thread_started:
                t = threading.Thread(target=bot_worker, daemon=True)
                t.start()
                worker_thread_started = True
                logger.info("🚀 Worker started")

# ============================================
# 🌐 FLASK ROUTES
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
    data = request.json
    
    with state_lock:
        bot_state["mode"] = data.get("mode", "paper")
        bot_state["symbol"] = data.get("symbol", "ETHUSDT")
        bot_state["risk_pct"] = float(data.get("risk_pct", 10.0))
        
        if bot_state["mode"] == "paper":
            init_bal = float(data.get("start_balance", 1000.0))
            bot_state["virtual_balance"] = init_bal
            bot_state["current_balance"] = init_bal
        
        bot_state["is_running"] = True
    
    logger.info(f"▶️ Bot started: {bot_state['mode']} | Balance: ${bot_state['current_balance']}")
    return jsonify({"status": "success", "message": f"Bot started in {bot_state['mode']} mode!"})

@app.route("/api/stop", methods=["POST"])
def stop_bot():
    with state_lock:
        bot_state["is_running"] = False
    logger.info("⏹️ Bot stopped")
    return jsonify({"status": "success", "message": "Bot stopped"})

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
        logger.info("🗑️ Trade history cleared")
    except Exception as e:
        logger.error(f"Reset error: {e}")
    
    with state_lock:
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
            "candles": bot_state["candles"][-100:],
            "ema_20_series": bot_state["ema_20_series"],
            "ema_50_series": bot_state["ema_50_series"],
            "ema_200_series": bot_state["ema_200_series"],
            "active_trades": bot_state["active_trades"],
            "active_position": active_pos_data
        })

@app.route("/api/export_csv")
def export_csv():
    try:
        conn = sqlite3.connect(DB_FILE)
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
# ▶️ MAIN
# ============================================

if __name__ == "__main__":
    init_db()
    load_db_history()
    logger.info("=" * 60)
    logger.info("🤖 CRYPTO SPOT DCA TRADING BOT (ACTIVE TRADING)")
    logger.info("=" * 60)
    app.run(debug=False, port=5000, threaded=True)
