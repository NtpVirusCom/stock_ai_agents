#import os
import sys
import json
import time
import re
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from contextlib import asynccontextmanager

import yfinance as yf
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# ========== Load Environment ==========
load_dotenv()

# OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
WATCHLIST = os.environ.get("WATCHLIST", "NVDA,ASML,AVGO").split(",")
CHECK_INTERVAL_MINUTES = int(os.environ.get("CHECK_INTERVAL_MINUTES", "2"))

# ========== Pydantic Models ==========
class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"

class AgentStatus(BaseModel):
    name: str
    status: str
    tasks: int
    errors: int
    last_action: str

class SignalRecord(BaseModel):
    time: str
    ticker: str
    signal: str
    confidence: float

class PortfolioSnapshot(BaseModel):
    time: str
    cash: float
    holdings: Dict[str, int]
    positions: int
    trades: int

class ErrorRecord(BaseModel):
    time: str
    ticker: str
    error: str
    source: str

class AnalysisRequest(BaseModel):
    ticker: str

class AnalysisResult(BaseModel):
    ticker: str
    timestamp: str
    overall_signal: str
    confidence: float
    risk_level: str
    suggested_position: str
    execution_plan: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    error: Optional[str] = None

# ========== AgentMonitor (Thread-safe Singleton) ==========
class AgentMonitor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.reset()
        return cls._instance

    def reset(self):
        self.agent_status: Dict[str, Dict] = {
            "ResearchAgent": {"status": "ready", "last_action": "-", "tasks": 0, "errors": 0},
            "AnalystAgent": {"status": "ready", "last_action": "-", "tasks": 0, "errors": 0},
            "StrategyAgent": {"status": "ready", "last_action": "-", "tasks": 0, "errors": 0},
            "RiskAgent": {"status": "ready", "last_action": "-", "tasks": 0, "errors": 0},
            "ExecutionAgent": {"status": "ready", "last_action": "-", "tasks": 0, "errors": 0},
        }
        self.signal_history: deque = deque(maxlen=100)
        self.error_log: deque = deque(maxlen=50)
        self.portfolio_history: deque = deque(maxlen=30)
        self.system_start_time = datetime.now()
        self.total_analyses = 0
        self.total_trades = 0
        self.failed_tickers: set = set()
        self.current_ticker = "-"
        self.pipeline_stage = "idle"

    def update_agent(self, name: str, status: str, action: str = "", error: bool = False):
        if name in self.agent_status:
            self.agent_status[name]["status"] = status
            if action:
                self.agent_status[name]["last_action"] = action
            self.agent_status[name]["tasks"] += 1
            if error:
                self.agent_status[name]["errors"] += 1

    def add_signal(self, ticker: str, signal: str, confidence: float):
        self.signal_history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "ticker": ticker, "signal": signal, "confidence": confidence
        })
        self.total_analyses += 1

    def add_error(self, ticker: str, error_msg: str, source: str = "System"):
        self.error_log.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "ticker": ticker, "error": error_msg[:100], "source": source
        })
        self.failed_tickers.add(ticker.upper())

    def add_portfolio(self, cash: float, holdings: Dict, trades: int):
        self.portfolio_history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "cash": cash, "holdings": dict(holdings),
            "positions": len(holdings), "trades": trades
        })
        self.total_trades = trades

    def set_pipeline(self, ticker: str, stage: str):
        self.current_ticker = ticker
        self.pipeline_stage = stage

    def get_uptime(self) -> str:
        delta = datetime.now() - self.system_start_time
        hours, rem = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def to_dict(self) -> Dict:
        return {
            "agents": self.agent_status,
            "signals": list(self.signal_history),
            "errors": list(self.error_log),
            "portfolio": list(self.portfolio_history),
            "stats": {
                "total_analyses": self.total_analyses,
                "total_trades": self.total_trades,
                "uptime": self.get_uptime(),
                "current_ticker": self.current_ticker,
                "pipeline_stage": self.pipeline_stage,
                "failed_tickers": list(self.failed_tickers)
            }
        }

MONITOR = AgentMonitor()

# ========== Tool Registry ==========
class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict] = {}
        self._failed_tickers: set = set()

    def register(self, name: str, func: Callable, schema: Dict):
        self._tools[name] = func
        self._schemas[name] = schema

    def call(self, name: str, **kwargs) -> Any:
        return self._tools[name](**kwargs)

    def get_schemas(self) -> List[Dict]:
        return [{"type": "function", "function": {"name": n, "description": s.get("description", ""), "parameters": s.get("parameters", {})}} 
                for n, s in self._schemas.items()]

    def mark_failed(self, ticker: str):
        self._failed_tickers.add(ticker.upper())
        MONITOR.add_error(ticker, "No data", "Tool")

    def is_failed(self, ticker: str) -> bool:
        return ticker.upper() in self._failed_tickers

# ========== Tools ==========
def fetch_stock_data(ticker: str, period: str = "6mo", interval: str = "1d") -> Dict:
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return {"error": f"No data for {ticker}"}
        df.index = df.index.tz_localize(None)
        latest = df.iloc[-1]
        return {
            "ticker": ticker, "price": round(latest['Close'], 2),
            "volume": int(latest['Volume']), "high": round(latest['High'], 2),
            "low": round(latest['Low'], 2), "open": round(latest['Open'], 2),
            "date": str(df.index[-1]), "rows": len(df)
        }
    except Exception as e:
        return {"error": str(e)[:100]}

def calculate_technical_indicators(ticker: str, period: str = "6mo") -> Dict:
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        if len(df) < 50:
            return {"error": "Insufficient data"}
        close = df['Close']
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        rsi = ta.rsi(close, length=14)
        rsi_14 = rsi.iloc[-1] if rsi is not None else None
        macd_df = ta.macd(close)
        macd_val = None
        macd_sig = None
        if macd_df is not None:
            macd_val = macd_df['MACD_12_26_9'].iloc[-1] if 'MACD_12_26_9' in macd_df.columns else None
            macd_sig = macd_df['MACDs_12_26_9'].iloc[-1] if 'MACDs_12_26_9' in macd_df.columns else None
        bb = ta.bbands(close, length=20, std=2)
        bb_upper = bb['BBU_20_2.0'].iloc[-1] if bb is not None and 'BBU_20_2.0' in bb.columns else None
        bb_lower = bb['BBL_20_2.0'].iloc[-1] if bb is not None and 'BBL_20_2.0' in bb.columns else None
        trend = "UPTREND" if sma_20 > sma_50 else "DOWNTREND" if sma_20 < sma_50 else "SIDEWAYS"
        return {
            "ticker": ticker, "price": round(close.iloc[-1], 2),
            "sma_20": round(sma_20, 2), "sma_50": round(sma_50, 2),
            "rsi_14": round(rsi_14, 2) if rsi_14 and pd.notna(rsi_14) else None,
            "macd": round(macd_val, 4) if macd_val and pd.notna(macd_val) else None,
            "macd_signal": round(macd_sig, 4) if macd_sig and pd.notna(macd_sig) else None,
            "bb_upper": round(bb_upper, 2) if bb_upper and pd.notna(bb_upper) else None,
            "bb_lower": round(bb_lower, 2) if bb_lower and pd.notna(bb_lower) else None,
            "volume": int(df['Volume'].iloc[-1]) if pd.notna(df['Volume'].iloc[-1]) else None,
            "trend": trend
        }
    except Exception as e:
        return {"error": str(e)[:100]}

def analyze_market_sentiment(ticker: str) -> Dict:
    try:
        df = yf.Ticker(ticker).history(period="1mo", interval="1d", auto_adjust=True)
        if len(df) < 5:
            return {"error": "No data"}
        returns = df['Close'].pct_change().dropna()
        return {
            "ticker": ticker, "sentiment": "BULLISH" if returns.mean() > 0.005 else "BEARISH" if returns.mean() < -0.005 else "NEUTRAL",
            "score": 0.7 if returns.mean() > 0.005 else 0.3 if returns.mean() < -0.005 else 0.5,
            "avg_return": round(returns.mean(), 4), "volatility": round(returns.std(), 4)
        }
    except Exception as e:
        return {"error": str(e)[:100]}

def calculate_position_size(cash: float, price: float, risk_percent: float = 0.02) -> Dict:
    if price <= 0:
        return {"error": "Invalid price"}
    max_risk = cash * risk_percent
    stop_dist = price * 0.05
    shares = int(max_risk / stop_dist) if stop_dist > 0 else 0
    return {
        "cash": round(cash, 2), "price": round(price, 2), "risk": risk_percent,
        "max_risk": round(max_risk, 2), "shares": shares, "cost": round(shares * price, 2),
        "stop_loss": round(price * 0.95, 2), "take_profit": round(price * 1.10, 2)
    }

def get_support_resistance(ticker: str, period: str = "3mo") -> Dict:
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        if len(df) < 20:
            return {"error": "Insufficient data"}
        return {
            "ticker": ticker, "all_high": round(df['High'].max(), 2),
            "all_low": round(df['Low'].min(), 2),
            "resistance": round(df['High'].tail(20).max(), 2),
            "support": round(df['Low'].tail(20).min(), 2),
            "current": round(df['Close'].iloc[-1], 2)
        }
    except Exception as e:
        return {"error": str(e)[:100]}

# ========== Base Agent ==========
class BaseAgent:
    def __init__(self, name: str, llm_client: Optional[OpenAI], tools: ToolRegistry, model: str = "gpt-4o-mini"):
        self.name = name
        self.llm = llm_client
        self.tools = tools
        self.model = model
        self.max_steps = 5

    def _llm_chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None, temperature: float = 0.3) -> Dict:
        if not self.llm:
            return {"content": "LLM not available", "tool_calls": None}
        try:
            kwargs = {"model": self.model, "messages": messages, "temperature": temperature}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            resp = self.llm.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            return {"content": msg.content or "", "tool_calls": msg.tool_calls}
        except Exception as e:
            return {"content": f"LLM Error: {e}", "tool_calls": None}

    def _execute_tool(self, tc) -> str:
        try:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            result = self.tools.call(name, **args)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return f"Tool Error: {e}"

    def think_and_act(self, task: str, context: Dict) -> Dict:
        MONITOR.update_agent(self.name, "thinking", f"Analyzing {context.get('ticker', '?')}")
        system_prompt = f"""You are {self.name} in an AI Trading System for US stocks.
Rules: 1. Think step by step 2. Use Tools if needed 3. Final answer as JSON with signal, confidence, reasoning 4. If unsure, return HOLD 5. If no data, return signal:HOLD, confidence:0, reasoning:No data available
Context: {json.dumps(context, ensure_ascii=False, default=str)[:2000]}"""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Task: {task}"}]
        tool_schemas = self.tools.get_schemas()

        for step in range(self.max_steps):
            resp = self._llm_chat(messages, tools=tool_schemas if tool_schemas else None)
            content = resp["content"]
            tool_calls = resp["tool_calls"]

            if tool_calls:
                for tc in tool_calls:
                    params = json.loads(tc.function.arguments)
                    ticker_param = params.get('ticker', '')
                    if ticker_param and self.tools.is_failed(ticker_param):
                        obs = '{"error": "Ticker not available"}'
                    else:
                        obs = self._execute_tool(tc)
                        if '"error"' in obs and 'No data' in obs:
                            self.tools.mark_failed(ticker_param)
                    messages.append({"role": "assistant", "content": content, "tool_calls": [tc.model_dump()]})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs})
                continue

            decision = self._parse_decision(content)
            MONITOR.update_agent(self.name, "ready", f"Done {context.get('ticker', '?')}")
            return decision

        MONITOR.update_agent(self.name, "stuck", f"Timeout {context.get('ticker', '?')}", error=True)
        return {"signal": "HOLD", "confidence": 0.3, "reasoning": "Max steps reached"}

    def _parse_decision(self, content: str) -> Dict:
        try:
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if m:
                r = json.loads(m.group())
                return {"signal": r.get("signal", "HOLD"), "confidence": float(r.get("confidence", 0.5)), "reasoning": r.get("reasoning", "No reasoning")}
        except:
            pass
        cu = content.upper()
        if "STRONG_BUY" in cu or ("BUY" in cu and "STRONG" in cu):
            return {"signal": "STRONG_BUY", "confidence": 0.8, "reasoning": content[:200]}
        elif "STRONG_SELL" in cu or ("SELL" in cu and "STRONG" in cu):
            return {"signal": "STRONG_SELL", "confidence": 0.8, "reasoning": content[:200]}
        elif "BUY" in cu:
            return {"signal": "BUY", "confidence": 0.6, "reasoning": content[:200]}
        elif "SELL" in cu:
            return {"signal": "SELL", "confidence": 0.6, "reasoning": content[:200]}
        return {"signal": "HOLD", "confidence": 0.5, "reasoning": content[:200]}

    def run(self, ticker: str, context: Dict) -> Dict:
        raise NotImplementedError

# ========== Specialized Agents ==========
class ResearchAgent(BaseAgent):
    def __init__(self, llm, tools):
        super().__init__("ResearchAgent", llm, tools)
    def run(self, ticker: str, context: Dict) -> Dict:
        return self.think_and_act(f"Analyze {ticker} - fetch data, calculate indicators, sentiment", context)

class AnalystAgent(BaseAgent):
    def __init__(self, llm, tools):
        super().__init__("AnalystAgent", llm, tools)
    def run(self, ticker: str, context: Dict) -> Dict:
        tech = calculate_technical_indicators(ticker)
        sentiment = analyze_market_sentiment(ticker)
        sr = get_support_resistance(ticker)
        ctx = {**context, "technical": tech, "sentiment": sentiment, "levels": sr}
        return self.think_and_act(f"Deep analysis of {ticker} with technical, sentiment, support/resistance", ctx)

class StrategyAgent(BaseAgent):
    def __init__(self, llm, tools):
        super().__init__("StrategyAgent", llm, tools)
    def run(self, ticker: str, context: Dict) -> Dict:
        opinions = [{"agent": d["agent_name"], "signal": d["signal"], "reasoning": d["reasoning"]} for d in context.get("decisions", [])]
        ctx = {**context, "other_opinions": opinions}
        return self.think_and_act(f"Strategy for {ticker} considering other agents' opinions. Return JSON with stop_loss, take_profit", ctx)

class RiskAgent(BaseAgent):
    def __init__(self, llm, tools):
        super().__init__("RiskAgent", llm, tools)
    def run(self, ticker: str, context: Dict) -> Dict:
        tech = context.get("technical", calculate_technical_indicators(ticker))
        if isinstance(tech, dict) and tech.get("error"):
            return {"signal": "HOLD", "confidence": 0.9, "reasoning": f"Cannot assess risk: {tech['error']}"}
        risk_score = 0.0
        reasons = []
        if tech.get("bb_upper") and tech.get("bb_lower") and tech.get("price"):
            bbw = (tech["bb_upper"] - tech["bb_lower"]) / tech["price"]
            if bbw > 0.10: risk_score += 0.3; reasons.append(f"High volatility ({bbw:.1%})")
        if tech.get("rsi_14") and (tech["rsi_14"] < 20 or tech["rsi_14"] > 80):
            risk_score += 0.2; reasons.append(f"RSI extreme ({tech['rsi_14']})")
        if tech.get("volume") and tech["volume"] < 1_000_000:
            risk_score += 0.1; reasons.append("Low volume")
        decisions = context.get("decisions", [])
        buy_count = sum(1 for d in decisions if d["signal"] in ("BUY", "STRONG_BUY"))
        sell_count = sum(1 for d in decisions if d["signal"] in ("SELL", "STRONG_SELL"))
        if buy_count > 0 and sell_count > 0:
            risk_score += 0.2; reasons.append("Agent disagreement")
        if risk_score > 0.5:
            return {"signal": "HOLD", "confidence": min(0.95, 0.5 + risk_score), "reasoning": " | ".join(reasons + ["High risk -> HOLD"])}
        return {"signal": "HOLD", "confidence": 0.5, "reasoning": " | ".join(reasons + ["Risk acceptable"])}

class ExecutionAgent(BaseAgent):
    def __init__(self, llm, tools, initial_cash: float = 10000.0):
        super().__init__("ExecutionAgent", llm, tools)
        self.cash = initial_cash
        self.holdings: Dict[str, int] = {}
        self.trade_history: List[Dict] = []

    def run(self, ticker: str, context: Dict) -> Dict:
        final_rec = context.get("final_recommendation")
        tech = context.get("technical", calculate_technical_indicators(ticker))
        price = tech.get("price", 0) if isinstance(tech, dict) and not tech.get("error") else 0
        if not final_rec or not price:
            return {"signal": "HOLD", "confidence": 0.0, "reasoning": "No data to execute"}
        signal = final_rec["signal"]
        pos = calculate_position_size(self.cash, price)
        shares = pos.get("shares", 0)
        if signal in ("BUY", "STRONG_BUY") and shares > 0:
            cost = shares * price
            self.cash -= cost
            self.holdings[ticker] = self.holdings.get(ticker, 0) + shares
            self.trade_history.append({"time": datetime.now().isoformat(), "ticker": ticker, "action": "BUY", "shares": shares, "price": price})
            MONITOR.add_portfolio(self.cash, self.holdings, len(self.trade_history))
            return {"signal": "BUY", "confidence": 0.95, "reasoning": f"Bought {shares} shares @ ${price}"}
        elif signal in ("SELL", "STRONG_SELL") and self.holdings.get(ticker, 0) > 0:
            shares = self.holdings[ticker]
            revenue = shares * price
            self.cash += revenue
            del self.holdings[ticker]
            self.trade_history.append({"time": datetime.now().isoformat(), "ticker": ticker, "action": "SELL", "shares": shares, "price": price})
            MONITOR.add_portfolio(self.cash, self.holdings, len(self.trade_history))
            return {"signal": "SELL", "confidence": 0.95, "reasoning": f"Sold {shares} shares @ ${price}"}
        return {"signal": "HOLD", "confidence": 0.5, "reasoning": f"Hold {self.holdings.get(ticker, 0)} shares"}

    def get_portfolio(self) -> Dict:
        return {"cash": round(self.cash, 2), "holdings": self.holdings, "positions": len(self.holdings), "trades": len(self.trade_history)}

# ========== Orchestrator ==========
class AgentOrchestrator:
    def __init__(self):
        self.tools = ToolRegistry()
        self._register_tools()
        llm = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_AVAILABLE and OPENAI_API_KEY else None
        self.research = ResearchAgent(llm, self.tools)
        self.analyst = AnalystAgent(llm, self.tools)
        self.strategy = StrategyAgent(llm, self.tools)
        self.risk = RiskAgent(llm, self.tools)
        self.execution = ExecutionAgent(llm, self.tools)

    def _register_tools(self):
        self.tools.register("fetch_stock_data", fetch_stock_data, {
            "description": "Fetch stock data from Yahoo Finance",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}, "period": {"type": "string", "default": "6mo"}, "interval": {"type": "string", "default": "1d"}}, "required": ["ticker"]}
        })
        self.tools.register("calculate_technical_indicators", calculate_technical_indicators, {
            "description": "Calculate technical indicators",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}, "period": {"type": "string", "default": "6mo"}}, "required": ["ticker"]}
        })
        self.tools.register("analyze_market_sentiment", analyze_market_sentiment, {
            "description": "Analyze market sentiment",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}
        })
        self.tools.register("calculate_position_size", calculate_position_size, {
            "description": "Calculate position size",
            "parameters": {"type": "object", "properties": {"cash": {"type": "number"}, "price": {"type": "number"}, "risk_percent": {"type": "number", "default": 0.02}}, "required": ["cash", "price"]}
        })
        self.tools.register("get_support_resistance", get_support_resistance, {
            "description": "Get support and resistance levels",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}, "period": {"type": "string", "default": "3mo"}}, "required": ["ticker"]}
        })

    def analyze(self, ticker: str) -> AnalysisResult:
        MONITOR.set_pipeline(ticker, "validating")
        if not ticker or not re.match(r'^[A-Z0-9\-\.]+$', ticker.upper()):
            err = f"Invalid ticker: {ticker}"
            MONITOR.add_error(ticker, err, "Validator")
            return AnalysisResult(ticker=ticker, timestamp=datetime.now().isoformat(), overall_signal="HOLD",
                                  confidence=0.0, risk_level="UNKNOWN", suggested_position="No recommendation",
                                  execution_plan="None", error=err)

        MONITOR.set_pipeline(ticker, "ResearchAgent")
        d1 = self.research.run(ticker, {"ticker": ticker, "decisions": []})
        d1["agent_name"] = "ResearchAgent"

        MONITOR.set_pipeline(ticker, "AnalystAgent")
        d2 = self.analyst.run(ticker, {"ticker": ticker, "decisions": [d1]})
        d2["agent_name"] = "AnalystAgent"

        MONITOR.set_pipeline(ticker, "StrategyAgent")
        d3 = self.strategy.run(ticker, {"ticker": ticker, "decisions": [d1, d2]})
        d3["agent_name"] = "StrategyAgent"

        MONITOR.set_pipeline(ticker, "RiskAgent")
        d4 = self.risk.run(ticker, {"ticker": ticker, "decisions": [d1, d2, d3], "technical": calculate_technical_indicators(ticker)})
        d4["agent_name"] = "RiskAgent"

        MONITOR.set_pipeline(ticker, "Consensus")
        weights = {"ResearchAgent": 0.15, "AnalystAgent": 0.30, "StrategyAgent": 0.30, "RiskAgent": 0.25}
        buy_score = sum(d["confidence"] * weights.get(d["agent_name"], 0.2) for d in [d1, d2, d3] if d["signal"] in ("BUY", "STRONG_BUY"))
        sell_score = sum(d["confidence"] * weights.get(d["agent_name"], 0.2) for d in [d1, d2, d3] if d["signal"] in ("SELL", "STRONG_SELL"))
        hold_score = sum(d["confidence"] * weights.get(d["agent_name"], 0.2) for d in [d1, d2, d3, d4] if d["signal"] == "HOLD")
        total = buy_score + sell_score + hold_score
        if total == 0:
            overall_signal, overall_confidence = "HOLD", 0.0
        else:
            scores = [("BUY", buy_score), ("SELL", sell_score), ("HOLD", hold_score)]
            overall_signal, max_score = max(scores, key=lambda x: x[1])
            overall_confidence = max_score / total if total > 0 else 0

        if d4["signal"] == "HOLD" and d4["confidence"] > 0.6:
            overall_signal = "HOLD"

        MONITOR.set_pipeline(ticker, "Execution")
        ctx = {"ticker": ticker, "final_recommendation": {"signal": overall_signal, "confidence": overall_confidence},
               "technical": calculate_technical_indicators(ticker), "decisions": [d1, d2, d3, d4]}
        d5 = self.execution.run(ticker, ctx)
        d5["agent_name"] = "ExecutionAgent"

        MONITOR.add_signal(ticker, overall_signal, overall_confidence)
        pf = self.execution.get_portfolio()
        MONITOR.add_portfolio(pf["cash"], pf["holdings"], pf["trades"])
        MONITOR.set_pipeline(ticker, "done")

        return AnalysisResult(
            ticker=ticker, timestamp=datetime.now().isoformat(), overall_signal=overall_signal,
            confidence=round(overall_confidence, 2), risk_level="MEDIUM",
            suggested_position="Buy" if overall_signal in ("BUY", "STRONG_BUY") else "Sell" if overall_signal in ("SELL", "STRONG_SELL") else "Hold",
            execution_plan=d5["reasoning"], error=None
        )

ORCH = AgentOrchestrator()

# ========== FastAPI App ==========
scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[Startup] AI Trading Agent System initializing...")
    # Run initial analysis
    for ticker in WATCHLIST[:1]:
        try:
            ORCH.analyze(ticker.strip().upper())
        except Exception as e:
            print(f"[Startup] Error analyzing {ticker}: {e}")
    # Start scheduler
    scheduler.add_job(auto_analysis_job, IntervalTrigger(minutes=CHECK_INTERVAL_MINUTES), id="auto_analysis", replace_existing=True)
    scheduler.start()
    print(f"[Scheduler] Started with interval: {CHECK_INTERVAL_MINUTES} minutes")
    yield
    # Shutdown
    scheduler.shutdown()
    print("[Shutdown] Scheduler stopped.")

app = FastAPI(title="AI Trading Agent System", version="1.0.0", lifespan=lifespan)

def auto_analysis_job():
    """Background job for auto scanning watchlist"""
    print(f"\n[Auto] Running scheduled analysis at {datetime.now().isoformat()}")
    for ticker in WATCHLIST:
        ticker = ticker.strip().upper()
        try:
            result = ORCH.analyze(ticker)
            print(f"[Auto] {ticker}: {result.overall_signal} ({result.confidence})")
        except Exception as e:
            print(f"[Auto] Error {ticker}: {e}")
            MONITOR.add_error(ticker, str(e), "AutoJob")

@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat(), "uptime": MONITOR.get_uptime()}

@app.get("/")
def root():
    return {
        "service": "AI Trading Agent System",
        "version": "1.0.0",
        "watchlist": WATCHLIST,
        "interval_minutes": CHECK_INTERVAL_MINUTES,
        "endpoints": ["/health", "/analyze/{ticker}", "/dashboard", "/api/status", "/api/signals", "/api/errors"]
    }

@app.post("/analyze", response_model=AnalysisResult)
def analyze_stock(req: AnalysisRequest):
    """Analyze a single stock on-demand"""
    return ORCH.analyze(req.ticker.strip().upper())

@app.get("/analyze/{ticker}")
def analyze_stock_get(ticker: str):
    """Analyze a single stock via GET"""
    return ORCH.analyze(ticker.strip().upper())

@app.get("/api/status")
def get_status():
    """Get full system status"""
    return MONITOR.to_dict()

@app.get("/api/agents")
def get_agents():
    """Get agent statuses"""
    return [{"name": k, **v} for k, v in MONITOR.agent_status.items()]

@app.get("/api/signals")
def get_signals(limit: int = Query(20, ge=1, le=100)):
    """Get recent signals"""
    return list(MONITOR.signal_history)[-limit:]

@app.get("/api/errors")
def get_errors(limit: int = Query(20, ge=1, le=50)):
    """Get recent errors"""
    return list(MONITOR.error_log)[-limit:]

@app.get("/api/portfolio")
def get_portfolio():
    """Get current portfolio"""
    if MONITOR.portfolio_history:
        return MONITOR.portfolio_history[-1]
    return {"cash": 10000.0, "holdings": {}, "positions": 0, "trades": 0}

@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    """HTML Dashboard - Auto refresh every 10 seconds"""
    data = MONITOR.to_dict()
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta http-equiv="refresh" content="10"><title>AI Agent Dashboard</title>
<style>
body{{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:20px}}
.header{{text-align:center;padding:20px;background:#1e293b;border-radius:12px;margin-bottom:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}}
.card{{background:#1e293b;border-radius:12px;padding:20px;border-left:4px solid #3b82f6}}
.card.error{{border-left-color:#ef4444}}.card.success{{border-left-color:#22c55e}}.card.warning{{border-left-color:#f59e0b}}
h2{{margin:0 0 15px 0;font-size:18px;color:#94a3b8}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #334155}}
th{{color:#94a3b8;font-weight:600}}.badge{{padding:4px 8px;border-radius:4px;font-size:12px;font-weight:bold}}
.badge.buy{{background:#22c55e;color:#fff}}.badge.sell{{background:#ef4444;color:#fff}}.badge.hold{{background:#64748b;color:#fff}}
.timestamp{{color:#64748b;font-size:12px}}.metric{{font-size:28px;font-weight:bold;color:#3b82f6}}
</style></head>
<body>
<div class="header">
<h1>🤖 AI Trading Agent Dashboard</h1>
<p>Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Uptime: {data['stats']['uptime']} | Analyses: {data['stats']['total_analyses']} | Trades: {data['stats']['total_trades']}</p>
</div>
<div class="grid">
<div class="card"><h2>⚙️ System Status</h2><p>Current: <strong>{data['stats']['current_ticker']}</strong></p><p>Stage: <strong>{data['stats']['pipeline_stage']}</strong></p><p>Failed Tickers: {', '.join(data['stats']['failed_tickers']) or 'None'}</p></div>
<div class="card"><h2>🤖 Agents</h2><table><tr><th>Agent</th><th>Status</th><th>Tasks</th><th>Errors</th></tr>
"""
    for name, info in data['agents'].items():
        html += f'<tr><td>{name}</td><td>{info["status"]}</td><td>{info["tasks"]}</td><td>{info["errors"]}</td></tr>\n'
    html += '</table></div>\n'

    html += '<div class="card success"><h2>📊 Recent Signals</h2><table><tr><th>Time</th><th>Ticker</th><th>Signal</th><th>Confidence</th></tr>\n'
    for sig in list(data['signals'])[-10:]:
        cls = sig['signal'].lower()
        html += f'<tr><td class="timestamp">{sig["time"]}</td><td>{sig["ticker"]}</td><td><span class="badge {cls}">{sig["signal"]}</span></td><td>{sig["confidence"]:.0%}</td></tr>\n'
    if not data['signals']:
        html += '<tr><td colspan="4">No signals yet</td></tr>\n'
    html += '</table></div>\n'

    html += '<div class="card"><h2>💰 Portfolio</h2>'
    if data['portfolio']:
        latest = data['portfolio'][-1]
        html += f'<p>Cash: <span class="metric">${latest["cash"]}</span></p><p>Holdings: {latest["holdings"]}</p><p>Positions: {latest["positions"]}</p><p>Trades: {latest["trades"]}</p>'
    else:
        html += '<p>No portfolio data</p>'
    html += '</div>\n'

    html += '<div class="card error"><h2>⚠️ Errors</h2><table><tr><th>Time</th><th>Ticker</th><th>Source</th><th>Error</th></tr>\n'
    for err in list(data['errors'])[-10:]:
        html += f'<tr><td class="timestamp">{err["time"]}</td><td>{err["ticker"]}</td><td>{err["source"]}</td><td>{err["error"]}</td></tr>\n'
    if not data['errors']:
        html += '<tr><td colspan="4">No errors</td></tr>\n'
    html += '</table></div>\n'

    html += '</div></body></html>'
    return html

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
