import asyncio
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import ccxt.async_support as ccxt
import pandas as pd
from tabulate import tabulate

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange


# =====================================================
# BUY ENTRY SCANNER SETTINGS
# =====================================================

EXCHANGES_TO_SCAN = [
    {
        "id": "coinbase",
        "quote": "USD",
        "max_symbols": 120,
    },
    {
        "id": "binanceus",
        "quote": "USDT",
        "max_symbols": 120,
    },
]

TIMEFRAME = "1h"
CANDLE_LIMIT = 150
CONCURRENCY = 5
SCAN_EVERY_SECONDS = 60

DESKTOP_HTML_FILE = Path(__file__).resolve().parent / "scanner_output" / "scanner_results.html"
REVIEW_QUEUE_FILE = Path(__file__).resolve().parent / "scanner_output" / "scanner_top3_review_queue.json"
APP_JSON_FILE = Path(__file__).resolve().parent / "scanner_output" / "scanner_app_data.json"

# Main entry idea:
# Find coins that are near oversold/bottom areas.
IDEAL_RSI_LOW = 25
IDEAL_RSI_HIGH = 35
STRICT_RSI_BUY_ZONE = 32
MAX_RSI_FOR_ENTRY = 40

# Volume confirmation
REL_VOLUME_SPIKE = 1.5
MIN_QUOTE_VOLUME_20 = 25_000

# ATR stop settings
ATR_STOP_MULTIPLIER = 1.5
TAKE_PROFIT_R_MULTIPLIER_1 = 1.5
TAKE_PROFIT_R_MULTIPLIER_2 = 2.0

# Next ceiling / resistance take-profit settings
CEILING_LOOKBACK_CANDLES = 60
MIN_CEILING_PROFIT_PCT = 0.75
CEILING_BUFFER_PCT = 0.25

# Avoid useless/risky symbols
MIN_PRICE = 0.000001

BAD_SYMBOL_WORDS = [
    "UP/", "DOWN/", "BULL/", "BEAR/",
    "3L/", "3S/", "5L/", "5S/",
    "PERP", "FUTURE", "SWAP",
]


def now_utc_string():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def is_good_symbol(symbol: str, market: dict, quote: str) -> bool:
    if not market.get("active", True):
        return False

    if not symbol.endswith(f"/{quote}"):
        return False

    for bad in BAD_SYMBOL_WORDS:
        if bad in symbol:
            return False

    base = symbol.split("/")[0]
    stable_bases = {
        "USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USD"
    }

    if base in stable_bases:
        return False

    if market.get("spot") is False:
        return False

    return True


def ohlcv_to_dataframe(ohlcv):
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["rsi_14"] = RSIIndicator(close=df["close"], window=14).rsi()

    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
    df["ema_100"] = EMAIndicator(close=df["close"], window=100).ema_indicator()

    atr = AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14
    )

    df["atr_14"] = atr.average_true_range()
    df["atr_pct"] = (df["atr_14"] / df["close"]) * 100

    df["volume_avg_20"] = df["volume"].rolling(20).mean()
    df["rel_volume"] = df["volume"] / df["volume_avg_20"]

    df["quote_volume"] = df["close"] * df["volume"]
    df["quote_volume_avg_20"] = df["quote_volume"].rolling(20).mean()

    df["roc_3"] = df["close"].pct_change(3) * 100
    df["roc_6"] = df["close"].pct_change(6) * 100
    df["roc_24"] = df["close"].pct_change(24) * 100

    df["lowest_low_14"] = df["low"].rolling(14).min()

    return df


def safe_round(value, digits=6):
    try:
        return round(float(value), digits)
    except Exception:
        return value


def find_next_ceilings(df, entry_price):
    """
    Finds realistic take-profit ceilings.

    Ceiling means recent resistance above the current entry price.
    We look at recent candle highs and choose the nearest highs above price.
    """
    recent = df.tail(CEILING_LOOKBACK_CANDLES).copy()

    min_ceiling_price = entry_price * (1 + MIN_CEILING_PROFIT_PCT / 100)

    highs_above = recent[recent["high"] > min_ceiling_price]["high"].tolist()

    if not highs_above:
        return None, None

    # Round highs a little so tiny candle differences do not create messy levels.
    rounded_highs = sorted(set([round(float(h), 8) for h in highs_above]))

    ceiling_1 = rounded_highs[0]

    ceiling_2 = None
    for h in rounded_highs:
        if h > ceiling_1 * 1.01:
            ceiling_2 = h
            break

    if ceiling_2 is None:
        ceiling_2 = rounded_highs[-1]

    # Sell slightly before the ceiling, not exactly at it.
    # This helps avoid missing the take profit by a tiny amount.
    tp1 = ceiling_1 * (1 - CEILING_BUFFER_PCT / 100)
    tp2 = ceiling_2 * (1 - CEILING_BUFFER_PCT / 100)

    return tp1, tp2


def calculate_atr_trade_plan(entry_price, atr_value, recent_low, df):
    """
    Stop loss still uses ATR.
    Take profit now uses the next ceiling / resistance rule.

    Stop:
    ATR Stop = entry - 1.5 ATR, compared with recent support.

    Take profit:
    TP1 = nearest recent ceiling above entry
    TP2 = next higher ceiling above entry

    If no ceiling is found, it falls back to the old R-multiple math.
    """
    atr_stop = entry_price - (atr_value * ATR_STOP_MULTIPLIER)

    support_stop = recent_low - (atr_value * 0.25)

    stop_loss = min(atr_stop, support_stop)

    risk_per_coin = entry_price - stop_loss

    if risk_per_coin <= 0:
        return None

    stop_loss_pct = (risk_per_coin / entry_price) * 100

    fallback_tp_1 = entry_price + (risk_per_coin * TAKE_PROFIT_R_MULTIPLIER_1)
    fallback_tp_2 = entry_price + (risk_per_coin * TAKE_PROFIT_R_MULTIPLIER_2)

    ceiling_tp_1, ceiling_tp_2 = find_next_ceilings(df, entry_price)

    if ceiling_tp_1 is not None:
        take_profit_1 = ceiling_tp_1
        take_profit_2 = ceiling_tp_2 if ceiling_tp_2 is not None else fallback_tp_2
        tp_method = "next ceiling / resistance"
    else:
        take_profit_1 = fallback_tp_1
        take_profit_2 = fallback_tp_2
        tp_method = "fallback risk multiple"

    tp1_profit_pct = ((take_profit_1 - entry_price) / entry_price) * 100
    tp2_profit_pct = ((take_profit_2 - entry_price) / entry_price) * 100

    reward_risk_1 = (take_profit_1 - entry_price) / risk_per_coin
    reward_risk_2 = (take_profit_2 - entry_price) / risk_per_coin

    return {
        "entry_price": safe_round(entry_price, 8),
        "atr_14": safe_round(atr_value, 8),
        "recent_low_14": safe_round(recent_low, 8),
        "stop_loss": safe_round(stop_loss, 8),
        "stop_loss_pct": safe_round(stop_loss_pct, 2),
        "take_profit_1": safe_round(take_profit_1, 8),
        "take_profit_2": safe_round(take_profit_2, 8),
        "tp1_profit_pct": safe_round(tp1_profit_pct, 2),
        "tp2_profit_pct": safe_round(tp2_profit_pct, 2),
        "reward_risk_1": safe_round(reward_risk_1, 2),
        "reward_risk_2": safe_round(reward_risk_2, 2),
        "tp_method": tp_method,
        "risk_per_coin": safe_round(risk_per_coin, 8),
    }


def score_symbol(exchange_id: str, symbol: str, df: pd.DataFrame):
    if len(df) < 110:
        return None, "not enough candles"

    df = add_indicators(df)
    latest = df.iloc[-1]

    required_cols = [
        "close", "low", "rsi_14", "rel_volume", "quote_volume_avg_20",
        "ema_50", "ema_100", "atr_14", "atr_pct",
        "roc_3", "roc_6", "roc_24", "lowest_low_14"
    ]

    for col in required_cols:
        value = latest[col]
        if value is None or pd.isna(value) or math.isinf(value):
            return None, f"bad indicator value: {col}"

    price = float(latest["close"])
    rsi = float(latest["rsi_14"])
    rel_volume = float(latest["rel_volume"])
    quote_volume_avg_20 = float(latest["quote_volume_avg_20"])
    ema_50 = float(latest["ema_50"])
    ema_100 = float(latest["ema_100"])
    atr_value = float(latest["atr_14"])
    atr_pct = float(latest["atr_pct"])
    roc_3 = float(latest["roc_3"])
    roc_6 = float(latest["roc_6"])
    roc_24 = float(latest["roc_24"])
    recent_low = float(latest["lowest_low_14"])

    if price < MIN_PRICE:
        return None, "price too low"

    if quote_volume_avg_20 < MIN_QUOTE_VOLUME_20:
        return None, "volume too low"

    trade_plan = calculate_atr_trade_plan(price, atr_value, recent_low, df)

    if not trade_plan:
        return None, "bad ATR trade plan"

    entry_score = 0
    reasons = []
    warnings = []
    categories = []

    # =====================================================
    # RSI entry zone scoring
    # =====================================================

    if IDEAL_RSI_LOW <= rsi <= IDEAL_RSI_HIGH:
        entry_score += 40
        categories.append("Ideal RSI Buy Zone")
        reasons.append(f"RSI is in the ideal buy-entry zone at {rsi:.1f}")
    elif rsi < IDEAL_RSI_LOW:
        entry_score += 25
        categories.append("Very Oversold")
        reasons.append(f"RSI is extremely oversold at {rsi:.1f}, but may still be falling")
        warnings.append("RSI is very low, so wait for bounce confirmation")
    elif rsi <= MAX_RSI_FOR_ENTRY:
        entry_score += 20
        categories.append("Oversold")
        reasons.append(f"RSI is still oversold/low at {rsi:.1f}")
    else:
        entry_score -= 20
        reasons.append(f"RSI is not low enough for your bottom-entry plan at {rsi:.1f}")

    # =====================================================
    # Volume confirmation
    # =====================================================

    if rel_volume >= REL_VOLUME_SPIKE:
        entry_score += 30
        categories.append("Volume Confirmed")
        reasons.append(f"volume spike confirms interest at {rel_volume:.2f}x normal")
    elif rel_volume >= 1.1:
        entry_score += 10
        reasons.append(f"volume is slightly above normal at {rel_volume:.2f}x")
    else:
        entry_score -= 10
        warnings.append(f"volume is weak/normal at {rel_volume:.2f}x")

    # =====================================================
    # Bounce confirmation
    # =====================================================

    if roc_3 > 0:
        entry_score += 15
        categories.append("Bounce Starting")
        reasons.append(f"3-candle momentum is turning up +{roc_3:.2f}%")
    else:
        warnings.append(f"3-candle momentum is still weak {roc_3:.2f}%")

    if roc_6 > 0:
        entry_score += 10
        reasons.append(f"6-candle momentum is positive +{roc_6:.2f}%")
    else:
        warnings.append(f"6-candle momentum is still weak {roc_6:.2f}%")

    # =====================================================
    # Avoid coins still crashing
    # =====================================================

    if roc_24 < -12:
        entry_score -= 25
        warnings.append(f"large 24-candle drop {roc_24:.2f}%, may still be crashing")
    elif roc_24 < -8:
        entry_score -= 10
        warnings.append(f"24-candle drop is heavy at {roc_24:.2f}%")
    else:
        entry_score += 5
        reasons.append(f"24-candle move is not a severe crash at {roc_24:.2f}%")

    # =====================================================
    # ATR risk quality
    # =====================================================

    if 1.0 <= atr_pct <= 8.0:
        entry_score += 10
        reasons.append(f"ATR volatility is usable at {atr_pct:.2f}%")
    elif atr_pct > 12:
        entry_score -= 20
        warnings.append(f"ATR is very risky at {atr_pct:.2f}%")
    elif atr_pct < 0.5:
        entry_score -= 10
        warnings.append(f"ATR is very low at {atr_pct:.2f}%, may not move enough")
    else:
        reasons.append(f"ATR is acceptable at {atr_pct:.2f}%")

    # =====================================================
    # Trend info, but not the main focus
    # Since you are looking for bottoms, price may be below EMA.
    # =====================================================

    if price > ema_50:
        entry_score += 5
        reasons.append("price is already back above EMA 50")
    else:
        warnings.append("price is below EMA 50, so this is still a reversal setup")

    if ema_50 > ema_100:
        entry_score += 5
        reasons.append("EMA 50 is above EMA 100")
    else:
        warnings.append("EMA 50 is below EMA 100, larger trend is still weak")

    strict_buy_entry = (
        rsi <= STRICT_RSI_BUY_ZONE
        and rel_volume >= REL_VOLUME_SPIKE
        and quote_volume_avg_20 >= MIN_QUOTE_VOLUME_20
        and 0.5 <= atr_pct <= 12.0
        and roc_24 > -12
    )

    if strict_buy_entry:
        categories.append("Strict Buy Entry")
        entry_score += 25

    return {
        "exchange": exchange_id,
        "symbol": symbol,
        "entry_score": round(entry_score, 2),
        "price": safe_round(price, 8),
        "rsi": round(rsi, 2),
        "rel_volume": round(rel_volume, 2),
        "atr_pct": round(atr_pct, 2),
        "roc_3": round(roc_3, 2),
        "roc_6": round(roc_6, 2),
        "roc_24": round(roc_24, 2),
        "quote_volume_avg_20": round(quote_volume_avg_20, 2),
        "strict_buy_entry": strict_buy_entry,
        "categories": ", ".join(sorted(set(categories))) if categories else "Watchlist",
        "reasons": "; ".join(reasons),
        "warnings": "; ".join(warnings) if warnings else "No major warnings",
        "trade_plan": trade_plan,
    }, None


async def fetch_and_score(exchange, exchange_id, symbol, semaphore):
    async with semaphore:
        try:
            ohlcv = await exchange.fetch_ohlcv(
                symbol,
                timeframe=TIMEFRAME,
                limit=CANDLE_LIMIT
            )

            df = ohlcv_to_dataframe(ohlcv)
            result, skipped_reason = score_symbol(exchange_id, symbol, df)

            if result:
                return result

            return {
                "exchange": exchange_id,
                "symbol": symbol,
                "skipped": skipped_reason,
            }

        except Exception as e:
            return {
                "exchange": exchange_id,
                "symbol": symbol,
                "error": str(e),
            }


async def scan_one_exchange(exchange_settings):
    exchange_id = exchange_settings["id"]
    quote = exchange_settings["quote"]
    max_symbols = exchange_settings["max_symbols"]

    print()
    print("=" * 80)
    print(f"SCANNING EXCHANGE: {exchange_id.upper()} | QUOTE: {quote}")
    print("=" * 80)

    try:
        exchange_class = getattr(ccxt, exchange_id)
    except AttributeError:
        print(f"Exchange not found in CCXT: {exchange_id}")
        return [], [], []

    exchange = exchange_class({
        "enableRateLimit": True,
    })

    try:
        markets = await exchange.load_markets()

        symbols = [
            symbol
            for symbol, market in markets.items()
            if is_good_symbol(symbol, market, quote)
        ]

        symbols = symbols[:max_symbols]

        print(f"Found {len(symbols)} symbols to scan on {exchange_id}.")

        semaphore = asyncio.Semaphore(CONCURRENCY)

        tasks = [
            fetch_and_score(exchange, exchange_id, symbol, semaphore)
            for symbol in symbols
        ]

        raw_results = await asyncio.gather(*tasks)

        results = [
            r for r in raw_results
            if r and "error" not in r and "skipped" not in r
        ]

        errors = [
            r for r in raw_results
            if r and "error" in r
        ]

        skipped = [
            r for r in raw_results
            if r and "skipped" in r
        ]

        print(f"{exchange_id} valid scored results: {len(results)}")
        print(f"{exchange_id} skipped: {len(skipped)}")
        print(f"{exchange_id} errors: {len(errors)}")

        return results, skipped, errors

    finally:
        await exchange.close()


def pick_number_one_choice(all_results):
    """
    Picks the #1 buy-entry choice.

    Hard rule:
    The #1 choice MUST have RSI <= MAX_RSI_FOR_ENTRY.
    This prevents high-RSI coins from becoming #1 just because they have
    strong volume or momentum.

    Preferred:
    Strict buy entries first.
    Then ideal RSI zone entries.
    Then any low-RSI entry.
    If nothing has low RSI, return None.
    """
    if not all_results:
        return None

    # Hard filter: do not allow high-RSI coins to become #1.
    low_rsi_results = [
        r for r in all_results
        if r.get("rsi", 999) <= MAX_RSI_FOR_ENTRY
    ]

    if not low_rsi_results:
        return None

    # Best case: strict buy-entry setup.
    strict_entries = [
        r for r in low_rsi_results
        if r.get("strict_buy_entry")
    ]

    if strict_entries:
        return sorted(
            strict_entries,
            key=lambda x: x["entry_score"],
            reverse=True
        )[0]

    # Second best: RSI in the ideal 25-35 buy zone.
    ideal_rsi_entries = [
        r for r in low_rsi_results
        if IDEAL_RSI_LOW <= r.get("rsi", 999) <= IDEAL_RSI_HIGH
    ]

    if ideal_rsi_entries:
        return sorted(
            ideal_rsi_entries,
            key=lambda x: x["entry_score"],
            reverse=True
        )[0]

    # Last fallback: RSI is still low enough, but not perfect.
    return sorted(
        low_rsi_results,
        key=lambda x: x["entry_score"],
        reverse=True
    )[0]


def get_coin_key(result):
    return f"{result['exchange']}|{result['symbol']}"


def compact_review_item(result):
    tp = result["trade_plan"]

    return {
        "key": get_coin_key(result),
        "exchange": result["exchange"],
        "symbol": result["symbol"],
        "entry_score": result["entry_score"],
        "entry_price": tp["entry_price"],
        "rsi": result["rsi"],
        "rel_volume": result["rel_volume"],
        "atr_pct": result["atr_pct"],
        "stop_loss": tp["stop_loss"],
        "risk_pct": tp["stop_loss_pct"],
        "take_profit_1": tp["take_profit_1"],
        "take_profit_2": tp["take_profit_2"],
        "categories": result["categories"],
        "reasons": result["reasons"],
        "warnings": result["warnings"],
        "last_seen": now_utc_string(),
    }


def load_review_queue():
    if not REVIEW_QUEUE_FILE.exists():
        return []

    try:
        data = json.loads(REVIEW_QUEUE_FILE.read_text())
        if isinstance(data, list):
            return data[:3]
    except Exception:
        return []

    return []


def save_review_queue(queue):
    REVIEW_QUEUE_FILE.write_text(json.dumps(queue[:3], indent=2), encoding="utf-8")


def update_review_queue(current_number_one):
    """
    Keeps a Top 3 list so good setups do not disappear too fast.

    If the current #1 is new:
    new #1 goes on top
    old #1 moves to #2
    old #2 moves to #3
    old #3 falls off
    """
    queue = load_review_queue()

    if not current_number_one:
        return queue[:3]

    new_item = compact_review_item(current_number_one)
    new_key = new_item["key"]

    # If the same coin is already in the queue, remove the old copy.
    queue = [
        item for item in queue
        if item.get("key") != new_key
    ]

    # Put current best coin at the top.
    queue.insert(0, new_item)

    queue = queue[:3]
    save_review_queue(queue)

    return queue


def make_review_queue_html(queue):
    if not queue:
        return "<p>No review queue yet. Let the scanner run for a few rounds.</p>"

    cards = ""

    for i, item in enumerate(queue, start=1):
        cards += f"""
        <div class="review-card">
            <h3>#{i} — {html.escape(item["exchange"])} | {html.escape(item["symbol"])}</h3>
            <div class="grid">
                <div><b>Entry Score:</b> {item["entry_score"]}</div>
                <div><b>Entry Price:</b> {item["entry_price"]}</div>
                <div><b>RSI:</b> {item["rsi"]}</div>
                <div><b>Rel Volume:</b> {item["rel_volume"]}x</div>
                <div><b>ATR %:</b> {item["atr_pct"]}%</div>
                <div><b>ATR Stop:</b> {item["stop_loss"]}</div>
                <div><b>Risk %:</b> {item["risk_pct"]}%</div>
                <div><b>TP1:</b> {item["take_profit_1"]}</div>
                <div><b>TP2:</b> {item["take_profit_2"]}</div>
                <div><b>Last Seen:</b> {html.escape(item["last_seen"])}</div>
            </div>
            <p><b>Why:</b> {html.escape(item["reasons"])}</p>
            <p><b>Warnings:</b> {html.escape(item["warnings"])}</p>
        </div>
        """

    return cards


def make_html_table(rows):
    if not rows:
        return "<p>No results found.</p>"

    table_rows = ""

    for r in rows:
        tp = r["trade_plan"]

        table_rows += f"""
        <tr>
            <td>{html.escape(str(r["exchange"]))}</td>
            <td>{html.escape(str(r["symbol"]))}</td>
            <td>{r["entry_score"]}</td>
            <td>{r["price"]}</td>
            <td>{r["rsi"]}</td>
            <td>{r["rel_volume"]}x</td>
            <td>{r["atr_pct"]}%</td>
            <td>{r["roc_3"]}%</td>
            <td>{r["roc_6"]}%</td>
            <td>{tp["stop_loss"]}</td>
            <td>{tp["stop_loss_pct"]}%</td>
            <td>{tp["take_profit_1"]}</td>
            <td>{tp.get("tp1_profit_pct", "n/a")}%</td>
            <td>{tp["take_profit_2"]}</td>
            <td>{tp.get("tp_method", "n/a")}</td>
            <td>{html.escape(str(r["categories"]))}</td>
        </tr>
        """

    return f"""
    <table>
        <tr>
            <th>Exchange</th>
            <th>Symbol</th>
            <th>Entry Score</th>
            <th>Entry</th>
            <th>RSI</th>
            <th>Rel Vol</th>
            <th>ATR%</th>
            <th>ROC 3</th>
            <th>ROC 6</th>
            <th>ATR Stop</th>
            <th>Risk%</th>
            <th>TP 1</th>
            <th>TP1 %</th>
            <th>TP 2</th>
            <th>TP Method</th>
            <th>Category</th>
        </tr>
        {table_rows}
    </table>
    """


def write_desktop_html(all_results, all_skipped, all_errors):
    all_results = sorted(all_results, key=lambda x: x["entry_score"], reverse=True)

    top_10 = all_results[:10]

    strict_buy_entries = [
        r for r in all_results
        if r["strict_buy_entry"]
    ][:5]

    number_one = pick_number_one_choice(all_results)
    review_queue = update_review_queue(number_one)

    if number_one:
        tp = number_one["trade_plan"]

        number_one_html = f"""
        <div class="number-one">
            <h2>#1 BUY ENTRY CHOICE BASED ON YOUR TERMS</h2>
            <h1>{html.escape(number_one["exchange"])} | {html.escape(number_one["symbol"])}</h1>

            <div class="grid">
                <div><b>Entry Score:</b> {number_one["entry_score"]}</div>
                <div><b>Entry Price:</b> {tp["entry_price"]}</div>
                <div><b>RSI:</b> {number_one["rsi"]}</div>
                <div><b>Relative Volume:</b> {number_one["rel_volume"]}x</div>
                <div><b>ATR 14:</b> {tp["atr_14"]}</div>
                <div><b>ATR %:</b> {number_one["atr_pct"]}%</div>
                <div><b>ATR Stop Loss:</b> {tp["stop_loss"]}</div>
                <div><b>Risk to Stop:</b> {tp["stop_loss_pct"]}%</div>
                <div><b>Take Profit 1:</b> {tp["take_profit_1"]}</div>
                <div><b>TP1 Profit:</b> {tp.get("tp1_profit_pct", "n/a")}%</div>
                <div><b>Take Profit 2:</b> {tp["take_profit_2"]}</div>
                <div><b>TP2 Profit:</b> {tp.get("tp2_profit_pct", "n/a")}%</div>
                <div><b>TP Method:</b> {html.escape(str(tp.get("tp_method", "n/a")))}</div>
            </div>

            <p><b>Why it was picked:</b> {html.escape(number_one["reasons"])}</p>
            <p><b>Warnings:</b> {html.escape(number_one["warnings"])}</p>
        </div>
        """
    else:
        number_one_html = """
        <div class="number-one">
            <h2>#1 BUY ENTRY CHOICE BASED ON YOUR TERMS</h2>
            <p>No valid #1 choice found yet.</p>
        </div>
        """

    top_10_reasons = ""

    for i, r in enumerate(top_10, start=1):
        tp = r["trade_plan"]

        top_10_reasons += f"""
        <div class="card">
            <h3>{i}. {html.escape(r["exchange"])} | {html.escape(r["symbol"])} — Entry Score: {r["entry_score"]}</h3>
            <p><b>Entry:</b> {tp["entry_price"]}</p>
            <p><b>ATR Stop:</b> {tp["stop_loss"]} | <b>Risk:</b> {tp["stop_loss_pct"]}%</p>
            <p><b>TP1:</b> {tp["take_profit_1"]} | <b>TP2:</b> {tp["take_profit_2"]}</p>
            <p><b>Reasons:</b> {html.escape(r["reasons"])}</p>
            <p><b>Warnings:</b> {html.escape(r["warnings"])}</p>
        </div>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="20">
        <title>Crypto Buy Entry Scanner</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #0f172a;
                color: #e5e7eb;
                padding: 24px;
            }}

            h1, h2, h3 {{
                margin-bottom: 8px;
            }}

            .small {{
                color: #94a3b8;
                margin-bottom: 20px;
            }}

            .number-one {{
                background: #14532d;
                border: 2px solid #22c55e;
                padding: 20px;
                border-radius: 16px;
                margin-bottom: 24px;
            }}

            .section {{
                background: #111827;
                padding: 18px;
                border-radius: 16px;
                margin-bottom: 24px;
            }}

            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 10px;
                margin: 16px 0;
            }}

            .grid div {{
                background: #052e16;
                border: 1px solid #22c55e;
                padding: 10px;
                border-radius: 10px;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 12px;
                font-size: 13px;
            }}

            th, td {{
                border: 1px solid #334155;
                padding: 8px;
                text-align: left;
            }}

            th {{
                background: #1e293b;
            }}

            tr:nth-child(even) {{
                background: #172033;
            }}

            .card {{
                background: #1e293b;
                border-radius: 12px;
                padding: 12px;
                margin-bottom: 12px;
            }}

            .warning {{
                background: #451a03;
                border: 1px solid #f97316;
                padding: 12px;
                border-radius: 12px;
                margin-bottom: 20px;
            }}
        </style>
    </head>

    <body>
        <h1>Crypto Buy Entry Scanner</h1>
        <p class="small">
            Last updated: {now_utc_string()} |
            Browser refreshes every 20 seconds |
            Scanner runs every {SCAN_EVERY_SECONDS} seconds
        </p>

        <div class="warning">
            Paper-trading / research scanner only. This is not financial advice.
            This scanner is designed to find possible buy-entry setups using RSI, volume, momentum, and ATR risk.
        </div>

        <div class="section review-section">
            <h2>Top 3 To Check Before They Disappear</h2>
            <p>This keeps the latest #1 choices. When a new #1 appears, the old #1 moves down to #2, then #3 before dropping off.</p>
            {make_review_queue_html(review_queue)}
        </div>

        {number_one_html}

        <div class="section">
            <h2>Top 10 Buy Entry Setups</h2>
            {make_html_table(top_10)}
        </div>

        <div class="section">
            <h2>Top 5 Strict Buy Entries</h2>
            <p>Strict setup means RSI near 30, volume spike, usable ATR, enough liquidity, and not a severe 24-candle crash.</p>
            {make_html_table(strict_buy_entries)}
        </div>

        <div class="section">
            <h2>ATR Stop Loss Math</h2>
            <p><b>Entry Price:</b> current close price.</p>
            <p><b>ATR Stop:</b> entry price minus 1.5 × ATR, compared with recent 14-candle low.</p>
            <p><b>Risk %:</b> distance from entry to stop loss.</p>
            <p><b>TP1:</b> nearest recent ceiling/resistance above entry.</p>
            <p><b>TP2:</b> next higher recent ceiling/resistance above entry.</p>
            <p><b>Backup:</b> if no ceiling is found, the scanner falls back to the old 1.5R / 2R target math.</p>
        </div>

        <div class="section">
            <h2>Reasons For Top 10</h2>
            {top_10_reasons}
        </div>

        <div class="section">
            <h2>Scan Summary</h2>
            <p>Valid scored results: {len(all_results)}</p>
            <p>Skipped results: {len(all_skipped)}</p>
            <p>Errors: {len(all_errors)}</p>
        </div>
    </body>
    </html>
    """

    DESKTOP_HTML_FILE.write_text(html_content, encoding="utf-8")



def write_app_json(all_results, all_skipped, all_errors):
    """
    Writes a clean JSON file for future Android/iPhone/web apps.
    This does NOT change scanner scoring logic. It only exports the results.
    """
    sorted_results = sorted(all_results, key=lambda x: x["entry_score"], reverse=True)

    top_10 = sorted_results[:10]

    strict_buy_entries = [
        r for r in sorted_results
        if r["strict_buy_entry"]
    ][:5]

    number_one = pick_number_one_choice(sorted_results)
    review_queue = load_review_queue()

    data = {
        "app": "SignalDrone AI",
        "source": "scanner_v1.py",
        "mode": "paper_trading_research_only",
        "disclaimer": "Paper-trading research only. Not financial advice. No real-money trading is performed.",
        "updated_at": now_utc_string(),
        "timeframe": TIMEFRAME,
        "scan_every_seconds": SCAN_EVERY_SECONDS,
        "exchanges": EXCHANGES_TO_SCAN,
        "number_one": number_one,
        "top_3_review_queue": review_queue,
        "top_10": top_10,
        "top_5_strict_buy_entries": strict_buy_entries,
        "summary": {
            "valid_scored_results": len(all_results),
            "skipped_results": len(all_skipped),
            "errors": len(all_errors),
        },
    }

    APP_JSON_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def print_terminal_results(all_results, all_skipped, all_errors):
    all_results = sorted(all_results, key=lambda x: x["entry_score"], reverse=True)

    top_10 = all_results[:10]

    strict_buy_entries = [
        r for r in all_results
        if r["strict_buy_entry"]
    ][:5]

    number_one = pick_number_one_choice(all_results)
    review_queue = update_review_queue(number_one)

    print()
    print("=" * 80)
    print("TOP 3 TO CHECK BEFORE THEY DISAPPEAR")
    print("=" * 80)

    if review_queue:
        for i, item in enumerate(review_queue, start=1):
            print()
            print(f"#{i}: {item['exchange']} | {item['symbol']}")
            print(f"Entry Score: {item['entry_score']}")
            print(f"Entry Price: {item['entry_price']}")
            print(f"RSI: {item['rsi']}")
            print(f"Relative Volume: {item['rel_volume']}x")
            print(f"ATR Stop: {item['stop_loss']}")
            print(f"Risk: {item['risk_pct']}%")
            print(f"TP1: {item['take_profit_1']}")
            print(f"TP2: {item['take_profit_2']}")
            print(f"Last Seen: {item['last_seen']}")
    else:
        print("No review queue yet.")

    print()
    print("=" * 80)
    print("#1 BUY ENTRY CHOICE BASED ON YOUR TERMS")
    print("=" * 80)

    if number_one:
        tp = number_one["trade_plan"]

        print(f"{number_one['exchange']} | {number_one['symbol']}")
        print(f"Entry Score: {number_one['entry_score']}")
        print(f"Entry Price: {tp['entry_price']}")
        print(f"RSI: {number_one['rsi']}")
        print(f"Relative Volume: {number_one['rel_volume']}x")
        print(f"ATR 14: {tp['atr_14']}")
        print(f"ATR Stop Loss: {tp['stop_loss']}")
        print(f"Risk to Stop: {tp['stop_loss_pct']}%")
        print(f"Take Profit 1: {tp['take_profit_1']} ({tp.get('tp1_profit_pct', 'n/a')}%)")
        print(f"Take Profit 2: {tp['take_profit_2']} ({tp.get('tp2_profit_pct', 'n/a')}%)")
        print(f"TP Method: {tp.get('tp_method', 'n/a')}")
        print(f"Reason: {number_one['reasons']}")
        print(f"Warnings: {number_one['warnings']}")
    else:
        print("No valid #1 choice found.")

    print()
    print("=" * 80)
    print("FULL SCAN SUMMARY")
    print("=" * 80)
    print(f"Valid scored results: {len(all_results)}")
    print(f"Skipped results: {len(all_skipped)}")
    print(f"Errors: {len(all_errors)}")
    print(f"Desktop file: {DESKTOP_HTML_FILE}")

    print()
    print("=" * 80)
    print("TOP 10 BUY ENTRY SETUPS")
    print("=" * 80)

    if top_10:
        table = []

        for r in top_10:
            tp = r["trade_plan"]

            table.append([
                r["exchange"],
                r["symbol"],
                r["entry_score"],
                tp["entry_price"],
                r["rsi"],
                r["rel_volume"],
                r["atr_pct"],
                r["roc_3"],
                r["roc_6"],
                tp["stop_loss"],
                tp["stop_loss_pct"],
                tp["take_profit_1"],
                tp.get("tp1_profit_pct", "n/a"),
                tp.get("tp_method", "n/a"),
                r["categories"],
            ])

        print(tabulate(
            table,
            headers=[
                "Exchange", "Symbol", "Entry", "Price", "RSI", "RelVol",
                "ATR%", "ROC3", "ROC6", "ATR Stop", "Risk%", "TP1", "TP1%", "TP Method", "Category"
            ],
            tablefmt="grid"
        ))
    else:
        print("No valid results found.")

    print()
    print("=" * 80)
    print("TOP 5 STRICT BUY ENTRIES")
    print("RSI near 30 + volume spike + ATR stop math")
    print("=" * 80)

    if strict_buy_entries:
        table = []

        for r in strict_buy_entries:
            tp = r["trade_plan"]

            table.append([
                r["exchange"],
                r["symbol"],
                r["entry_score"],
                tp["entry_price"],
                r["rsi"],
                r["rel_volume"],
                tp["stop_loss"],
                tp["stop_loss_pct"],
                tp["take_profit_1"],
                tp.get("tp1_profit_pct", "n/a"),
                tp["take_profit_2"],
            ])

        print(tabulate(
            table,
            headers=[
                "Exchange", "Symbol", "Entry", "Price", "RSI",
                "RelVol", "ATR Stop", "Risk%", "TP1", "TP1%", "TP2"
            ],
            tablefmt="grid"
        ))
    else:
        print("No strict buy entries right now.")
        print("That is okay. It means RSI + volume + risk are not lining up cleanly yet.")


async def run_one_scan():
    print()
    print("=" * 80)
    print("CRYPTO BUY ENTRY SCANNER V1.4")
    print("Coinbase + BinanceUS | RSI bottom search + ATR stop math")
    print("=" * 80)
    print(f"Scanner started: {now_utc_string()}")
    print(f"Timeframe: {TIMEFRAME}")

    all_results = []
    all_skipped = []
    all_errors = []

    for exchange_settings in EXCHANGES_TO_SCAN:
        results, skipped, errors = await scan_one_exchange(exchange_settings)

        all_results.extend(results)
        all_skipped.extend(skipped)
        all_errors.extend(errors)

    print_terminal_results(all_results, all_skipped, all_errors)
    write_desktop_html(all_results, all_skipped, all_errors)
    write_app_json(all_results, all_skipped, all_errors)

    print()
    print("=" * 80)
    print(f"Scan finished: {now_utc_string()}")
    print(f"Open this file on your Desktop: scanner_results.html")
    print("=" * 80)


async def main():
    first_run = True

    while True:
        await run_one_scan()

        if first_run:
            first_run = False
            print()
            print("HTML dashboard saved. Auto-open is turned off.")

        print()
        print(f"Waiting {SCAN_EVERY_SECONDS} seconds before next scan...")
        print("Press CTRL + C to stop.")
        await asyncio.sleep(SCAN_EVERY_SECONDS)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
        print("Scanner stopped by user.")
