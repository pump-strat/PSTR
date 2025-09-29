import asyncio
import aiohttp
import json
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
import sqlite3
import hashlib
import hmac
from decimal import Decimal, ROUND_DOWN
import math


class HealthStatus(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    CRITICAL = "critical"


class InterventionTier(Enum):
    NONE = 0
    TIER_1 = 10
    TIER_2 = 25
    TIER_3 = 40


@dataclass
class PriceData:
    timestamp: int
    price: float
    volume: float
    confidence: float
    source: str


@dataclass
class TreasuryState:
    pump_balance: float
    pump_price: float
    total_value_usd: float
    pstr_market_cap: float
    reserve_ratio: float
    intervention_capacity: int
    last_intervention: Optional[int] = None
    accumulated_today: float = 0.0
    total_accumulated: float = 0.0


@dataclass
class InterventionRecord:
    timestamp: int
    trigger_price: float
    vwap_30d: float
    drawdown_pct: float
    deployment_pct: float
    capital_deployed: float
    pump_sold: float
    pstr_bought: float
    execution_cost: float
    price_impact: float
    success: bool


@dataclass
class LiquidityPool:
    pool_id: str
    token_a: str
    token_b: str
    liquidity_a: float
    liquidity_b: float
    total_value_usd: float
    fee_tier: float
    volume_24h: float
    price_lower: float
    price_upper: float
    in_range: bool


@dataclass
class MonitoringMetrics:
    timestamp: int
    pstr_price: float
    pump_price: float
    vwap_30d: float
    trigger_ratio: float
    drawdown_pct: float
    reserve_ratio: float
    health_score: float
    liquidity_depth: float
    intervention_ready: bool
    recommended_action: str


class DatabaseManager:
    def __init__(self, db_path: str = "pstr_monitor.db"):
        self.db_path = db_path
        self.conn = None
        self.initialize_database()

    def initialize_database(self):
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                token TEXT NOT NULL,
                price REAL NOT NULL,
                volume REAL NOT NULL,
                source TEXT NOT NULL,
                confidence REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interventions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                trigger_price REAL NOT NULL,
                vwap_30d REAL NOT NULL,
                drawdown_pct REAL NOT NULL,
                deployment_pct REAL NOT NULL,
                capital_deployed REAL NOT NULL,
                pump_sold REAL NOT NULL,
                pstr_bought REAL NOT NULL,
                execution_cost REAL NOT NULL,
                price_impact REAL NOT NULL,
                success INTEGER NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS treasury_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                pump_balance REAL NOT NULL,
                pump_price REAL NOT NULL,
                total_value_usd REAL NOT NULL,
                pstr_market_cap REAL NOT NULL,
                reserve_ratio REAL NOT NULL,
                intervention_capacity INTEGER NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS monitoring_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                pstr_price REAL NOT NULL,
                pump_price REAL NOT NULL,
                vwap_30d REAL NOT NULL,
                trigger_ratio REAL NOT NULL,
                drawdown_pct REAL NOT NULL,
                reserve_ratio REAL NOT NULL,
                health_score REAL NOT NULL,
                liquidity_depth REAL NOT NULL,
                intervention_ready INTEGER NOT NULL,
                recommended_action TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS accumulation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                amount_usd REAL NOT NULL,
                pump_amount REAL NOT NULL,
                pump_price REAL NOT NULL,
                transaction_hash TEXT
            )
        """)
        
        self.conn.commit()

    def insert_price_data(self, token: str, data: PriceData):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO price_history (timestamp, token, price, volume, source, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (data.timestamp, token, data.price, data.volume, data.source, data.confidence))
        self.conn.commit()

    def insert_intervention(self, record: InterventionRecord):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO interventions (
                timestamp, trigger_price, vwap_30d, drawdown_pct, deployment_pct,
                capital_deployed, pump_sold, pstr_bought, execution_cost, price_impact, success
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.timestamp, record.trigger_price, record.vwap_30d, record.drawdown_pct,
            record.deployment_pct, record.capital_deployed, record.pump_sold,
            record.pstr_bought, record.execution_cost, record.price_impact,
            1 if record.success else 0
        ))
        self.conn.commit()

    def insert_treasury_snapshot(self, state: TreasuryState):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO treasury_snapshots (
                timestamp, pump_balance, pump_price, total_value_usd,
                pstr_market_cap, reserve_ratio, intervention_capacity
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            int(time.time()), state.pump_balance, state.pump_price,
            state.total_value_usd, state.pstr_market_cap,
            state.reserve_ratio, state.intervention_capacity
        ))
        self.conn.commit()

    def insert_monitoring_log(self, metrics: MonitoringMetrics):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO monitoring_logs (
                timestamp, pstr_price, pump_price, vwap_30d, trigger_ratio,
                drawdown_pct, reserve_ratio, health_score, liquidity_depth,
                intervention_ready, recommended_action
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.timestamp, metrics.pstr_price, metrics.pump_price,
            metrics.vwap_30d, metrics.trigger_ratio, metrics.drawdown_pct,
            metrics.reserve_ratio, metrics.health_score, metrics.liquidity_depth,
            1 if metrics.intervention_ready else 0, metrics.recommended_action
        ))
        self.conn.commit()

    def get_price_history(self, token: str, days: int) -> List[PriceData]:
        cursor = self.conn.cursor()
        cutoff = int(time.time()) - (days * 86400)
        cursor.execute("""
            SELECT timestamp, price, volume, confidence, source
            FROM price_history
            WHERE token = ? AND timestamp > ?
            ORDER BY timestamp ASC
        """, (token, cutoff))
        
        results = []
        for row in cursor.fetchall():
            results.append(PriceData(
                timestamp=row[0],
                price=row[1],
                volume=row[2],
                confidence=row[3],
                source=row[4]
            ))
        return results

    def get_last_intervention_time(self) -> Optional[int]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT MAX(timestamp) FROM interventions")
        result = cursor.fetchone()
        return result[0] if result[0] else None

    def close(self):
        if self.conn:
            self.conn.close()


class OracleAggregator:
    def __init__(self, config: Dict):
        self.config = config
        self.pyth_endpoint = config.get("pyth_endpoint", "https://hermes.pyth.network")
        self.switchboard_endpoint = config.get("switchboard_endpoint", "https://api.switchboard.xyz")
        self.session = None

    async def initialize(self):
        self.session = aiohttp.ClientSession()

    async def fetch_pyth_price(self, feed_id: str) -> Optional[PriceData]:
        try:
            url = f"{self.pyth_endpoint}/api/latest_price_feeds?ids[]={feed_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        price_feed = data[0]
                        price_data = price_feed.get("price", {})
                        price = float(price_data.get("price", 0)) * (10 ** price_data.get("expo", 0))
                        confidence = float(price_data.get("conf", 0)) * (10 ** price_data.get("expo", 0))
                        
                        return PriceData(
                            timestamp=int(time.time()),
                            price=price,
                            volume=0.0,
                            confidence=confidence,
                            source="pyth"
                        )
        except Exception as e:
            logging.error(f"Pyth oracle error: {e}")
        return None

    async def fetch_switchboard_price(self, aggregator_key: str) -> Optional[PriceData]:
        try:
            url = f"{self.switchboard_endpoint}/aggregator/{aggregator_key}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("result", {})
                    
                    return PriceData(
                        timestamp=int(time.time()),
                        price=float(result.get("value", 0)),
                        volume=0.0,
                        confidence=float(result.get("std_dev", 0)),
                        source="switchboard"
                    )
        except Exception as e:
            logging.error(f"Switchboard oracle error: {e}")
        return None

    async def fetch_dex_twap(self, pool_address: str, period_seconds: int = 1800) -> Optional[PriceData]:
        try:
            current_time = int(time.time())
            observations = []
            
            for i in range(period_seconds // 60):
                timestamp = current_time - (i * 60)
                price = await self.fetch_pool_price_at_time(pool_address, timestamp)
                if price:
                    observations.append(price)
            
            if observations:
                twap = statistics.mean(observations)
                return PriceData(
                    timestamp=current_time,
                    price=twap,
                    volume=0.0,
                    confidence=statistics.stdev(observations) if len(observations) > 1 else 0.0,
                    source="dex_twap"
                )
        except Exception as e:
            logging.error(f"DEX TWAP error: {e}")
        return None

    async def fetch_pool_price_at_time(self, pool_address: str, timestamp: int) -> Optional[float]:
        return None

    async def get_consensus_price(self, token: str, feeds: Dict[str, str]) -> Tuple[Optional[float], Optional[float]]:
        prices = []
        
        pyth_feed = feeds.get("pyth")
        if pyth_feed:
            pyth_data = await self.fetch_pyth_price(pyth_feed)
            if pyth_data:
                prices.append(pyth_data.price)
        
        switchboard_feed = feeds.get("switchboard")
        if switchboard_feed:
            sb_data = await self.fetch_switchboard_price(switchboard_feed)
            if sb_data:
                prices.append(sb_data.price)
        
        dex_pool = feeds.get("dex_pool")
        if dex_pool:
            twap_data = await self.fetch_dex_twap(dex_pool)
            if twap_data:
                prices.append(twap_data.price)
        
        if not prices:
            return None, None
        
        if len(prices) == 1:
            return prices[0], 0.0
        
        sorted_prices = sorted(prices)
        median_price = sorted_prices[len(sorted_prices) // 2]
        
        max_deviation = max(abs(p - median_price) / median_price for p in prices)
        
        if max_deviation > 0.05:
            logging.warning(f"High price deviation detected: {max_deviation:.2%}")
        
        return median_price, max_deviation

    async def close(self):
        if self.session:
            await self.session.close()


class VWAPCalculator:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def calculate_vwap(self, token: str, days: int) -> Optional[float]:
        price_history = self.db.get_price_history(token, days)
        
        if not price_history:
            return None
        
        total_pv = sum(p.price * p.volume for p in price_history)
        total_volume = sum(p.volume for p in price_history)
        
        if total_volume == 0:
            return statistics.mean([p.price for p in price_history])
        
        return total_pv / total_volume

    def calculate_drawdown(self, current_price: float, vwap: float) -> float:
        if vwap == 0:
            return 0.0
        return ((vwap - current_price) / vwap) * 100

    def calculate_volatility(self, token: str, days: int) -> float:
        price_history = self.db.get_price_history(token, days)
        
        if len(price_history) < 2:
            return 0.0
        
        prices = [p.price for p in price_history]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        return statistics.stdev(returns) * math.sqrt(252) * 100 if returns else 0.0


class TreasuryManager:
    def __init__(self, db_manager: DatabaseManager, oracle: OracleAggregator, config: Dict):
        self.db = db_manager
        self.oracle = oracle
        self.config = config
        self.treasury_state = None
        
        self.total_capital = config.get("total_capital", 1000000)
        self.accumulation_period_days = config.get("accumulation_period", 365)
        self.treasury_allocation = config.get("treasury_allocation", 0.90)
        self.liquidity_allocation = config.get("liquidity_allocation", 0.10)
        self.min_reserve_ratio = config.get("min_reserve_ratio", 3.0)
        self.max_pump_concentration = config.get("max_pump_concentration", 0.80)

    def calculate_daily_budget(self) -> float:
        return (self.total_capital * self.treasury_allocation) / self.accumulation_period_days

    def calculate_hourly_budget(self) -> float:
        return self.calculate_daily_budget() / 24

    async def get_treasury_state(self) -> TreasuryState:
        pump_price, _ = await self.oracle.get_consensus_price("PUMP", self.config.get("pump_feeds", {}))
        pstr_price, _ = await self.oracle.get_consensus_price("PSTR", self.config.get("pstr_feeds", {}))
        
        if not pump_price or not pstr_price:
            return self.treasury_state
        
        pump_balance = self.config.get("pump_balance", 0)
        total_value = pump_balance * pump_price
        
        pstr_supply = self.config.get("pstr_total_supply", 10000000)
        pstr_market_cap = pstr_supply * pstr_price
        
        reserve_ratio = total_value / pstr_market_cap if pstr_market_cap > 0 else 0
        intervention_capacity = int(reserve_ratio / 0.40) if reserve_ratio > 0 else 0
        
        self.treasury_state = TreasuryState(
            pump_balance=pump_balance,
            pump_price=pump_price,
            total_value_usd=total_value,
            pstr_market_cap=pstr_market_cap,
            reserve_ratio=reserve_ratio,
            intervention_capacity=intervention_capacity,
            last_intervention=self.db.get_last_intervention_time()
        )
        
        return self.treasury_state

    def determine_deployment_percentage(self, drawdown_pct: float) -> InterventionTier:
        if drawdown_pct >= 30:
            return InterventionTier.TIER_3
        elif drawdown_pct >= 20:
            return InterventionTier.TIER_2
        elif drawdown_pct >= 15:
            return InterventionTier.TIER_1
        else:
            return InterventionTier.NONE

    def calculate_capital_deployment(self, treasury_value: float, deployment_pct: float) -> float:
        return treasury_value * (deployment_pct / 100)

    def calculate_pump_to_sell(self, capital_deployed: float, pump_price: float) -> float:
        return capital_deployed / pump_price

    def calculate_pstr_to_buy(self, capital_deployed: float, pstr_price: float, slippage: float = 0.02) -> float:
        effective_capital = capital_deployed * (1 - slippage)
        return effective_capital / pstr_price

    def calculate_price_impact(self, buy_amount: float, price: float, liquidity_depth: float) -> float:
        if liquidity_depth == 0:
            return 0.0
        return ((buy_amount * price) / liquidity_depth) * 100

    def calculate_recharge_multiplier(self, deployed_amount: float, remaining_treasury: float) -> float:
        if remaining_treasury == 0:
            return 1.0
        multiplier = 1 + (deployed_amount / remaining_treasury)
        return min(multiplier, 3.0)

    def calculate_days_to_recharge(self, deployed_amount: float, daily_budget: float, multiplier: float) -> int:
        accelerated_budget = daily_budget * multiplier
        if accelerated_budget == 0:
            return 0
        return int(deployed_amount / accelerated_budget)


class LiquidityManager:
    def __init__(self, config: Dict):
        self.config = config
        self.pools: List[LiquidityPool] = []
        
        self.sol_pool_weight = config.get("sol_pool_weight", 0.60)
        self.usdc_pool_weight = config.get("usdc_pool_weight", 0.40)
        self.concentration_range = config.get("concentration_range", 0.10)
        self.rebalance_threshold = config.get("rebalance_threshold", 0.15)

    def calculate_liquidity_budget(self, total_capital: float, allocation: float) -> float:
        return total_capital * allocation

    def calculate_pool_allocation(self, liquidity_budget: float) -> Tuple[float, float]:
        sol_allocation = liquidity_budget * self.sol_pool_weight
        usdc_allocation = liquidity_budget * self.usdc_pool_weight
        return sol_allocation, usdc_allocation

    def calculate_price_range(self, current_price: float) -> Tuple[float, float]:
        price_lower = current_price * (1 - self.concentration_range)
        price_upper = current_price * (1 + self.concentration_range)
        return price_lower, price_upper

    def calculate_capital_efficiency(self, liquidity_in_range: float, total_liquidity: float) -> float:
        if total_liquidity == 0:
            return 0.0
        return liquidity_in_range / total_liquidity

    def calculate_impermanent_loss(self, price_initial: float, price_final: float) -> float:
        if price_initial == 0:
            return 0.0
        price_ratio = price_final / price_initial
        il = (2 * math.sqrt(price_ratio)) / (1 + price_ratio) - 1
        return il * 100

    def check_rebalance_needed(self, current_price: float, range_center: float) -> bool:
        if range_center == 0:
            return False
        deviation = abs(current_price - range_center) / range_center
        return deviation > self.rebalance_threshold

    async def get_pool_liquidity_depth(self, pool_id: str) -> float:
        return 0.0

    async def get_pool_stats(self, pool_id: str) -> Optional[LiquidityPool]:
        return None


class InterventionEngine:
    def __init__(self, 
                 db_manager: DatabaseManager,
                 treasury_manager: TreasuryManager,
                 vwap_calculator: VWAPCalculator,
                 config: Dict):
        self.db = db_manager
        self.treasury = treasury_manager
        self.vwap_calc = vwap_calculator
        self.config = config
        
        self.trigger_threshold = config.get("trigger_threshold", 0.85)
        self.cooldown_hours = config.get("cooldown_hours", 48)
        self.max_slippage = config.get("max_slippage", 0.02)
        self.gas_estimate = config.get("gas_estimate", 100)

    def check_intervention_conditions(self, 
                                     current_price: float,
                                     vwap_30d: float,
                                     last_intervention_time: Optional[int]) -> Tuple[bool, str]:
        if last_intervention_time:
            cooldown_seconds = self.cooldown_hours * 3600
            time_since_last = int(time.time()) - last_intervention_time
            if time_since_last < cooldown_seconds:
                remaining_hours = (cooldown_seconds - time_since_last) / 3600
                return False, f"Cooldown active: {remaining_hours:.1f} hours remaining"
        
        trigger_ratio = current_price / vwap_30d if vwap_30d > 0 else 1.0
        
        if trigger_ratio >= self.trigger_threshold:
            return False, f"No intervention needed: ratio {trigger_ratio:.4f} above threshold {self.trigger_threshold}"
        
        return True, "Intervention conditions met"

    def calculate_expected_benefit(self, 
                                  current_price: float,
                                  target_price: float,
                                  pstr_to_buy: float) -> float:
        price_recovery = target_price - current_price
        return pstr_to_buy * price_recovery

    def calculate_execution_cost(self, capital_deployed: float) -> float:
        slippage_cost = capital_deployed * self.max_slippage
        return slippage_cost + self.gas_estimate

    def estimate_intervention_roi(self,
                                 expected_benefit: float,
                                 execution_cost: float,
                                 capital_deployed: float) -> float:
        net_benefit = expected_benefit - execution_cost
        if capital_deployed == 0:
            return 0.0
        return (net_benefit / capital_deployed) * 100

    async def execute_intervention(self,
                                  drawdown_pct: float,
                                  pstr_price: float,
                                  pump_price: float,
                                  vwap_30d: float) -> Optional[InterventionRecord]:
        treasury_state = await self.treasury.get_treasury_state()
        
        deployment_tier = self.treasury.determine_deployment_percentage(drawdown_pct)
        if deployment_tier == InterventionTier.NONE:
            return None
        
        deployment_pct = deployment_tier.value
        capital_deployed = self.treasury.calculate_capital_deployment(
            treasury_state.total_value_usd,
            deployment_pct
        )
        
        remaining_treasury = treasury_state.total_value_usd - capital_deployed
        new_reserve_ratio = remaining_treasury / treasury_state.pstr_market_cap
        
        if new_reserve_ratio < 1.5:
            capital_deployed = capital_deployed * 0.5
            deployment_pct = deployment_pct * 0.5
        
        pump_to_sell = self.treasury.calculate_pump_to_sell(capital_deployed, pump_price)
        pstr_to_buy = self.treasury.calculate_pstr_to_buy(capital_deployed, pstr_price, self.max_slippage)
        
        execution_cost = self.calculate_execution_cost(capital_deployed)
        
        target_price = vwap_30d * 0.90
        expected_benefit = self.calculate_expected_benefit(pstr_price, target_price, pstr_to_buy)
        
        roi = self.estimate_intervention_roi(expected_benefit, execution_cost, capital_deployed)
        
        if roi < 0:
            logging.warning(f"Intervention ROI negative: {roi:.2f}%")
            return None
        
        price_impact = self.treasury.calculate_price_impact(
            pstr_to_buy,
            pstr_price,
            treasury_state.pstr_market_cap * 0.1
        )
        
        record = InterventionRecord(
            timestamp=int(time.time()),
            trigger_price=pstr_price,
            vwap_30d=vwap_30d,
            drawdown_pct=drawdown_pct,
            deployment_pct=deployment_pct,
            capital_deployed=capital_deployed,
            pump_sold=pump_to_sell,
            pstr_bought=pstr_to_buy,
            execution_cost=execution_cost,
            price_impact=price_impact,
            success=True
        )
        
        self.db.insert_intervention(record)
        
        return record


class HealthMonitor:
    def __init__(self, 
                 treasury_manager: TreasuryManager,
                 liquidity_manager: LiquidityManager,
                 config: Dict):
        self.treasury = treasury_manager
        self.liquidity = liquidity_manager
        self.config = config
        
        self.weight_reserve_ratio = config.get("weight_rr", 0.333)
        self.weight_intervention_capacity = config.get("weight_ic", 0.333)
        self.weight_concentration = config.get("weight_conc", 0.334)

    async def calculate_health_score(self) -> float:
        treasury_state = await self.treasury.get_treasury_state()
        
        rr_score = min(treasury_state.reserve_ratio / 3.0, 2.0)
        ic_score = min(treasury_state.intervention_capacity / 3.0, 2.0)
        
        pump_concentration = treasury_state.total_value_usd / (treasury_state.total_value_usd + 0.01)
        concentration_score = 1 - pump_concentration
        
        health_score = (
            self.weight_reserve_ratio * rr_score +
            self.weight_intervention_capacity * ic_score +
            self.weight_concentration * concentration_score
        )
        
        return health_score

    def determine_health_status(self, health_score: float) -> HealthStatus:
        if health_score >= 1.5:
            return HealthStatus.EXCELLENT
        elif health_score >= 1.0:
            return HealthStatus.GOOD
        elif health_score >= 0.8:
            return HealthStatus.MODERATE
        elif health_score >= 0.6:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    async def generate_health_report(self) -> Dict:
        treasury_state = await self.treasury.get_treasury_state()
        health_score = await self.calculate_health_score()
        health_status = self.determine_health_status(health_score)
        
        report = {
            "timestamp": int(time.time()),
            "health_score": health_score,
            "health_status": health_status.value,
            "treasury_value_usd": treasury_state.total_value_usd,
            "reserve_ratio": treasury_state.reserve_ratio,
            "intervention_capacity": treasury_state.intervention_capacity,
            "pump_balance": treasury_state.pump_balance,
            "pump_price": treasury_state.pump_price,
            "pstr_market_cap": treasury_state.pstr_market_cap,
            "recommendations": []
        }
        
        if treasury_state.reserve_ratio < 2.0:
            report["recommendations"].append("Reserve ratio below 2.0: Increase accumulation rate")
        
        if treasury_state.intervention_capacity < 3:
            report["recommendations"].append("Low intervention capacity: Limit next deployment to 20%")
        
        if health_status in [HealthStatus.POOR, HealthStatus.CRITICAL]:
            report["recommendations"].append("Critical health status: Consider emergency governance vote")
        
        return report


class AlertSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.alert_thresholds = {
            "critical_reserve_ratio": config.get("alert_rr_critical", 1.5),
            "warning_reserve_ratio": config.get("alert_rr_warning", 2.0),
            "high_drawdown": config.get("alert_drawdown", 25.0),
            "oracle_deviation": config.get("alert_oracle_dev", 0.05),
            "low_liquidity": config.get("alert_liquidity", 100000)
        }
        self.alert_history = []

    def check_reserve_ratio_alert(self, reserve_ratio: float) -> Optional[Dict]:
        if reserve_ratio < self.alert_thresholds["critical_reserve_ratio"]:
            return {
                "level": "CRITICAL",
                "type": "reserve_ratio",
                "message": f"Reserve ratio critically low: {reserve_ratio:.2f}",
                "timestamp": int(time.time()),
                "value": reserve_ratio
            }
        elif reserve_ratio < self.alert_thresholds["warning_reserve_ratio"]:
            return {
                "level": "WARNING",
                "type": "reserve_ratio",
                "message": f"Reserve ratio below target: {reserve_ratio:.2f}",
                "timestamp": int(time.time()),
                "value": reserve_ratio
            }
        return None

    def check_drawdown_alert(self, drawdown_pct: float) -> Optional[Dict]:
        if drawdown_pct >= self.alert_thresholds["high_drawdown"]:
            return {
                "level": "HIGH",
                "type": "drawdown",
                "message": f"High drawdown detected: {drawdown_pct:.2f}%",
                "timestamp": int(time.time()),
                "value": drawdown_pct
            }
        return None

    def check_oracle_deviation_alert(self, deviation: float) -> Optional[Dict]:
        if deviation > self.alert_thresholds["oracle_deviation"]:
            return {
                "level": "WARNING",
                "type": "oracle_deviation",
                "message": f"High oracle price deviation: {deviation:.2%}",
                "timestamp": int(time.time()),
                "value": deviation
            }
        return None

    def check_liquidity_alert(self, liquidity_depth: float) -> Optional[Dict]:
        if liquidity_depth < self.alert_thresholds["low_liquidity"]:
            return {
                "level": "WARNING",
                "type": "low_liquidity",
                "message": f"Low liquidity depth: ${liquidity_depth:,.2f}",
                "timestamp": int(time.time()),
                "value": liquidity_depth
            }
        return None

    def process_alerts(self, metrics: MonitoringMetrics) -> List[Dict]:
        alerts = []
        
        rr_alert = self.check_reserve_ratio_alert(metrics.reserve_ratio)
        if rr_alert:
            alerts.append(rr_alert)
        
        dd_alert = self.check_drawdown_alert(metrics.drawdown_pct)
        if dd_alert:
            alerts.append(dd_alert)
        
        liq_alert = self.check_liquidity_alert(metrics.liquidity_depth)
        if liq_alert:
            alerts.append(liq_alert)
        
        for alert in alerts:
            self.alert_history.append(alert)
            logging.warning(f"[{alert['level']}] {alert['message']}")
        
        return alerts

    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        cutoff = int(time.time()) - (hours * 3600)
        return [a for a in self.alert_history if a["timestamp"] > cutoff]


class PerformanceAnalyzer:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def calculate_intervention_success_rate(self, days: int = 30) -> float:
        cursor = self.db.conn.cursor()
        cutoff = int(time.time()) - (days * 86400)
        
        cursor.execute("""
            SELECT COUNT(*) as total, SUM(success) as successful
            FROM interventions
            WHERE timestamp > ?
        """, (cutoff,))
        
        result = cursor.fetchone()
        total, successful = result[0], result[1] or 0
        
        if total == 0:
            return 0.0
        return (successful / total) * 100

    def calculate_average_intervention_impact(self, days: int = 30) -> Dict:
        cursor = self.db.conn.cursor()
        cutoff = int(time.time()) - (days * 86400)
        
        cursor.execute("""
            SELECT AVG(price_impact), AVG(capital_deployed), AVG(execution_cost)
            FROM interventions
            WHERE timestamp > ?
        """, (cutoff,))
        
        result = cursor.fetchone()
        
        return {
            "avg_price_impact": result[0] or 0.0,
            "avg_capital_deployed": result[1] or 0.0,
            "avg_execution_cost": result[2] or 0.0
        }

    def calculate_treasury_growth_rate(self, days: int = 30) -> float:
        cursor = self.db.conn.cursor()
        cutoff_start = int(time.time()) - (days * 86400)
        cutoff_end = int(time.time())
        
        cursor.execute("""
            SELECT total_value_usd
            FROM treasury_snapshots
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
            LIMIT 1
        """, (cutoff_start - 86400, cutoff_start + 86400))
        start_value = cursor.fetchone()
        
        cursor.execute("""
            SELECT total_value_usd
            FROM treasury_snapshots
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (cutoff_end - 86400, cutoff_end))
        end_value = cursor.fetchone()
        
        if not start_value or not end_value or start_value[0] == 0:
            return 0.0
        
        growth = ((end_value[0] - start_value[0]) / start_value[0]) * 100
        return growth

    def calculate_portfolio_metrics(self) -> Dict:
        cursor = self.db.conn.cursor()
        
        cursor.execute("""
            SELECT total_value_usd, pstr_market_cap, reserve_ratio
            FROM treasury_snapshots
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        latest = cursor.fetchone()
        
        if not latest:
            return {}
        
        cursor.execute("""
            SELECT timestamp, total_value_usd
            FROM treasury_snapshots
            ORDER BY timestamp ASC
        """)
        snapshots = cursor.fetchall()
        
        if len(snapshots) < 2:
            return {
                "current_value": latest[0],
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0
            }
        
        values = [s[1] for s in snapshots]
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        
        avg_return = statistics.mean(returns) if returns else 0.0
        std_return = statistics.stdev(returns) if len(returns) > 1 else 0.0
        
        sharpe_ratio = (avg_return / std_return) * math.sqrt(365) if std_return > 0 else 0.0
        
        peak = values[0]
        max_dd = 0.0
        for value in values:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak) * 100 if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        
        volatility = std_return * math.sqrt(365) * 100 if std_return > 0 else 0.0
        
        return {
            "current_value": latest[0],
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_dd,
            "volatility": volatility,
            "avg_daily_return": avg_return * 100
        }


class RiskManager:
    def __init__(self, config: Dict):
        self.config = config
        self.max_daily_interventions = config.get("max_daily_interventions", 1)
        self.max_deployment_per_intervention = config.get("max_deployment", 0.40)
        self.min_reserve_ratio_threshold = config.get("min_rr_threshold", 1.5)
        self.max_concentration_threshold = config.get("max_concentration", 0.80)
        
        self.intervention_count_today = 0
        self.last_intervention_date = None

    def reset_daily_counter(self):
        current_date = datetime.now().date()
        if self.last_intervention_date != current_date:
            self.intervention_count_today = 0
            self.last_intervention_date = current_date

    def check_intervention_limits(self) -> Tuple[bool, str]:
        self.reset_daily_counter()
        
        if self.intervention_count_today >= self.max_daily_interventions:
            return False, f"Daily intervention limit reached: {self.intervention_count_today}/{self.max_daily_interventions}"
        
        return True, "Within intervention limits"

    def check_reserve_safety(self, reserve_ratio: float, deployment_pct: float) -> Tuple[bool, str]:
        simulated_rr = reserve_ratio * (1 - deployment_pct / 100)
        
        if simulated_rr < self.min_reserve_ratio_threshold:
            recommended_pct = ((reserve_ratio - self.min_reserve_ratio_threshold) / reserve_ratio) * 100
            return False, f"Deployment would breach minimum RR. Recommended: {recommended_pct:.1f}%"
        
        return True, "Reserve levels safe"

    def check_concentration_risk(self, pump_value: float, total_value: float) -> Tuple[bool, str]:
        if total_value == 0:
            return True, "No concentration risk"
        
        concentration = pump_value / total_value
        
        if concentration > self.max_concentration_threshold:
            return False, f"PUMP concentration too high: {concentration:.1%}"
        
        return True, f"Concentration acceptable: {concentration:.1%}"

    def calculate_value_at_risk(self, portfolio_value: float, confidence: float = 0.95, horizon_days: int = 1) -> float:
        volatility_annual = 1.5
        volatility_daily = volatility_annual / math.sqrt(365)
        
        z_score = 1.645 if confidence == 0.95 else 2.326
        
        var = portfolio_value * volatility_daily * z_score * math.sqrt(horizon_days)
        return var

    def calculate_expected_shortfall(self, var: float) -> float:
        return var * 1.3

    def assess_market_conditions(self, volatility: float, volume_24h: float) -> str:
        if volatility > 100 or volume_24h < 10000:
            return "HIGH_RISK"
        elif volatility > 50 or volume_24h < 50000:
            return "MODERATE_RISK"
        else:
            return "LOW_RISK"


class GovernanceMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.proposals = []
        self.voting_power_cache = {}

    def calculate_voting_power(self, address: str, balance: float, staked: bool = False) -> float:
        multiplier = 1.5 if staked else 1.0
        
        if balance > 1000000:
            voting_power = math.sqrt(balance) * multiplier
        else:
            voting_power = balance * multiplier
        
        return voting_power

    def check_quorum(self, votes_cast: float, total_supply: float, proposal_type: str) -> Tuple[bool, float]:
        quorum_requirements = {
            "parameter_adjustment": 0.10,
            "treasury_strategy": 0.15,
            "protocol_upgrade": 0.20,
            "emergency": 0.25
        }
        
        required_quorum = quorum_requirements.get(proposal_type, 0.10)
        participation_rate = votes_cast / total_supply if total_supply > 0 else 0
        
        return participation_rate >= required_quorum, participation_rate

    def calculate_approval_threshold(self, proposal_type: str) -> float:
        thresholds = {
            "parameter_adjustment": 0.51,
            "treasury_strategy": 0.66,
            "protocol_upgrade": 0.75,
            "emergency": 0.75
        }
        return thresholds.get(proposal_type, 0.51)

    def validate_proposal(self, proposer_balance: float, proposal_type: str) -> Tuple[bool, str]:
        min_balance_requirements = {
            "parameter_adjustment": 0.01,
            "treasury_strategy": 0.02,
            "protocol_upgrade": 0.03,
            "emergency": 0.05
        }
        
        required_pct = min_balance_requirements.get(proposal_type, 0.01)
        total_supply = self.config.get("pstr_total_supply", 10000000)
        required_balance = total_supply * required_pct
        
        if proposer_balance < required_balance:
            return False, f"Insufficient balance: {proposer_balance:,.0f} < {required_balance:,.0f}"
        
        return True, "Proposer eligible"


class StrategyMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.db = DatabaseManager(config.get("db_path", "pstr_monitor.db"))
        self.oracle = OracleAggregator(config)
        self.vwap_calc = VWAPCalculator(self.db)
        self.treasury = TreasuryManager(self.db, self.oracle, config)
        self.liquidity = LiquidityManager(config)
        self.intervention_engine = InterventionEngine(self.db, self.treasury, self.vwap_calc, config)
        self.health_monitor = HealthMonitor(self.treasury, self.liquidity, config)
        self.alert_system = AlertSystem(config)
        self.performance_analyzer = PerformanceAnalyzer(self.db)
        self.risk_manager = RiskManager(config)
        self.governance_monitor = GovernanceMonitor(config)
        
        self.monitoring_interval = config.get("monitoring_interval", 300)
        self.is_running = False

    async def initialize(self):
        await self.oracle.initialize()
        logging.info("Strategy monitor initialized successfully")

    async def fetch_latest_prices(self) -> Tuple[Optional[float], Optional[float]]:
        pstr_price, pstr_deviation = await self.oracle.get_consensus_price(
            "PSTR",
            self.config.get("pstr_feeds", {})
        )
        
        pump_price, pump_deviation = await self.oracle.get_consensus_price(
            "PUMP",
            self.config.get("pump_feeds", {})
        )
        
        if pstr_price:
            self.db.insert_price_data("PSTR", PriceData(
                timestamp=int(time.time()),
                price=pstr_price,
                volume=0.0,
                confidence=pstr_deviation or 0.0,
                source="consensus"
            ))
        
        if pump_price:
            self.db.insert_price_data("PUMP", PriceData(
                timestamp=int(time.time()),
                price=pump_price,
                volume=0.0,
                confidence=pump_deviation or 0.0,
                source="consensus"
            ))
        
        if pstr_deviation and pstr_deviation > 0.05:
            self.alert_system.check_oracle_deviation_alert(pstr_deviation)
        
        return pstr_price, pump_price

    async def perform_monitoring_cycle(self):
        try:
            pstr_price, pump_price = await self.fetch_latest_prices()
            
            if not pstr_price or not pump_price:
                logging.error("Failed to fetch prices, skipping cycle")
                return
            
            vwap_30d = self.vwap_calc.calculate_vwap("PSTR", 30)
            if not vwap_30d:
                logging.warning("Insufficient price history for VWAP calculation")
                vwap_30d = pstr_price
            
            trigger_ratio = pstr_price / vwap_30d if vwap_30d > 0 else 1.0
            drawdown_pct = self.vwap_calc.calculate_drawdown(pstr_price, vwap_30d)
            
            treasury_state = await self.treasury.get_treasury_state()
            health_score = await self.health_monitor.calculate_health_score()
            
            liquidity_depth = await self.liquidity.get_pool_liquidity_depth(
                self.config.get("main_pool_id", "")
            )
            
            last_intervention = self.db.get_last_intervention_time()
            intervention_conditions_met, condition_message = self.intervention_engine.check_intervention_conditions(
                pstr_price,
                vwap_30d,
                last_intervention
            )
            
            metrics = MonitoringMetrics(
                timestamp=int(time.time()),
                pstr_price=pstr_price,
                pump_price=pump_price,
                vwap_30d=vwap_30d,
                trigger_ratio=trigger_ratio,
                drawdown_pct=drawdown_pct,
                reserve_ratio=treasury_state.reserve_ratio,
                health_score=health_score,
                liquidity_depth=liquidity_depth,
                intervention_ready=intervention_conditions_met,
                recommended_action=self.generate_recommendation(
                    trigger_ratio,
                    drawdown_pct,
                    treasury_state.reserve_ratio,
                    intervention_conditions_met
                )
            )
            
            self.db.insert_monitoring_log(metrics)
            self.db.insert_treasury_snapshot(treasury_state)
            
            alerts = self.alert_system.process_alerts(metrics)
            
            if intervention_conditions_met and drawdown_pct >= 15:
                limits_ok, limit_message = self.risk_manager.check_intervention_limits()
                
                if limits_ok:
                    deployment_tier = self.treasury.determine_deployment_percentage(drawdown_pct)
                    
                    safety_ok, safety_message = self.risk_manager.check_reserve_safety(
                        treasury_state.reserve_ratio,
                        deployment_tier.value
                    )
                    
                    if safety_ok:
                        logging.info(f"Executing intervention: {deployment_tier.value}% deployment")
                        
                        intervention_record = await self.intervention_engine.execute_intervention(
                            drawdown_pct,
                            pstr_price,
                            pump_price,
                            vwap_30d
                        )
                        
                        if intervention_record:
                            self.risk_manager.intervention_count_today += 1
                            logging.info(f"Intervention executed successfully: {intervention_record.capital_deployed:,.2f} USD deployed")
                        else:
                            logging.warning("Intervention execution failed or not profitable")
                    else:
                        logging.warning(f"Intervention blocked by safety check: {safety_message}")
                else:
                    logging.warning(f"Intervention blocked by limits: {limit_message}")
            
            await self.generate_status_report(metrics, treasury_state, alerts)
            
        except Exception as e:
            logging.error(f"Error in monitoring cycle: {e}", exc_info=True)

    def generate_recommendation(self,
                               trigger_ratio: float,
                               drawdown_pct: float,
                               reserve_ratio: float,
                               intervention_ready: bool) -> str:
        if not intervention_ready:
            return "MONITOR_ONLY"
        
        if drawdown_pct >= 30 and reserve_ratio > 2.5:
            return "INTERVENTION_TIER_3"
        elif drawdown_pct >= 20 and reserve_ratio > 2.0:
            return "INTERVENTION_TIER_2"
        elif drawdown_pct >= 15 and reserve_ratio > 1.8:
            return "INTERVENTION_TIER_1"
        elif reserve_ratio < 2.0:
            return "ACCELERATE_ACCUMULATION"
        else:
            return "CONTINUE_ACCUMULATION"

    async def generate_status_report(self,
                                    metrics: MonitoringMetrics,
                                    treasury_state: TreasuryState,
                                    alerts: List[Dict]):
        health_report = await self.health_monitor.generate_health_report()
        performance_metrics = self.performance_analyzer.calculate_portfolio_metrics()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "prices": {
                "pstr": metrics.pstr_price,
                "pump": metrics.pump_price,
                "vwap_30d": metrics.vwap_30d
            },
            "treasury": {
                "pump_balance": treasury_state.pump_balance,
                "total_value_usd": treasury_state.total_value_usd,
                "reserve_ratio": treasury_state.reserve_ratio,
                "intervention_capacity": treasury_state.intervention_capacity
            },
            "market_conditions": {
                "trigger_ratio": metrics.trigger_ratio,
                "drawdown_pct": metrics.drawdown_pct,
                "liquidity_depth": metrics.liquidity_depth
            },
            "health": {
                "score": health_report["health_score"],
                "status": health_report["health_status"],
                "recommendations": health_report["recommendations"]
            },
            "performance": performance_metrics,
            "alerts": alerts,
            "recommendation": metrics.recommended_action
        }
        
        logging.info(f"Status Report: {json.dumps(report, indent=2)}")
        
        return report

    async def run_accumulation_scheduler(self):
        while self.is_running:
            try:
                hourly_budget = self.treasury.calculate_hourly_budget()
                pump_price, _ = await self.oracle.get_consensus_price(
                    "PUMP",
                    self.config.get("pump_feeds", {})
                )
                
                if pump_price and pump_price > 0:
                    pump_amount = hourly_budget / pump_price
                    
                    logging.info(f"Hourly accumulation: ${hourly_budget:.2f} → {pump_amount:.2f} PUMP @ ${pump_price:.4f}")
                    
                    cursor = self.db.conn.cursor()
                    cursor.execute("""
                        INSERT INTO accumulation_history (timestamp, amount_usd, pump_amount, pump_price, transaction_hash)
                        VALUES (?, ?, ?, ?, ?)
                    """, (int(time.time()), hourly_budget, pump_amount, pump_price, "simulated"))
                    self.db.conn.commit()
                
                await asyncio.sleep(3600)
                
            except Exception as e:
                logging.error(f"Error in accumulation scheduler: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def run_liquidity_rebalancer(self):
        while self.is_running:
            try:
                pstr_price, _ = await self.oracle.get_consensus_price(
                    "PSTR",
                    self.config.get("pstr_feeds", {})
                )
                
                if pstr_price:
                    for pool_id in self.config.get("pool_ids", []):
                        pool_stats = await self.liquidity.get_pool_stats(pool_id)
                        
                        if pool_stats:
                            price_lower, price_upper = self.liquidity.calculate_price_range(pstr_price)
                            
                            needs_rebalance = self.liquidity.check_rebalance_needed(
                                pstr_price,
                                (price_lower + price_upper) / 2
                            )
                            
                            if needs_rebalance:
                                logging.info(f"Pool {pool_id} requires rebalancing")
                
                await asyncio.sleep(86400)
                
            except Exception as e:
                logging.error(f"Error in liquidity rebalancer: {e}", exc_info=True)
                await asyncio.sleep(3600)

    async def run_performance_reporter(self):
        while self.is_running:
            try:
                success_rate = self.performance_analyzer.calculate_intervention_success_rate()
                impact_metrics = self.performance_analyzer.calculate_average_intervention_impact()
                growth_rate = self.performance_analyzer.calculate_treasury_growth_rate()
                portfolio_metrics = self.performance_analyzer.calculate_portfolio_metrics()
                
                performance_report = {
                    "timestamp": datetime.now().isoformat(),
                    "intervention_success_rate": success_rate,
                    "average_intervention_metrics": impact_metrics,
                    "treasury_growth_rate_30d": growth_rate,
                    "portfolio_metrics": portfolio_metrics
                }
                
                logging.info(f"Performance Report: {json.dumps(performance_report, indent=2)}")
                
                await asyncio.sleep(21600)
                
            except Exception as e:
                logging.error(f"Error in performance reporter: {e}", exc_info=True)
                await asyncio.sleep(3600)

    async def run_risk_monitor(self):
        while self.is_running:
            try:
                treasury_state = await self.treasury.get_treasury_state()
                
                var_95 = self.risk_manager.calculate_value_at_risk(treasury_state.total_value_usd, 0.95, 1)
                es = self.risk_manager.calculate_expected_shortfall(var_95)
                
                volatility = self.vwap_calc.calculate_volatility("PSTR", 30)
                volume_24h = 100000
                
                market_risk = self.risk_manager.assess_market_conditions(volatility, volume_24h)
                
                concentration_ok, conc_message = self.risk_manager.check_concentration_risk(
                    treasury_state.total_value_usd,
                    treasury_state.total_value_usd
                )
                
                risk_report = {
                    "timestamp": datetime.now().isoformat(),
                    "value_at_risk_95": var_95,
                    "expected_shortfall": es,
                    "market_risk_level": market_risk,
                    "concentration_status": conc_message,
                    "portfolio_volatility": volatility
                }
                
                logging.info(f"Risk Report: {json.dumps(risk_report, indent=2)}")
                
                await asyncio.sleep(7200)
                
            except Exception as e:
                logging.error(f"Error in risk monitor: {e}", exc_info=True)
                await asyncio.sleep(3600)

    async def start(self):
        self.is_running = True
        logging.info("Starting PSTR Strategy Monitor...")
        
        await self.initialize()
        
        tasks = [
            asyncio.create_task(self.main_monitoring_loop()),
            asyncio.create_task(self.run_accumulation_scheduler()),
            asyncio.create_task(self.run_liquidity_rebalancer()),
            asyncio.create_task(self.run_performance_reporter()),
            asyncio.create_task(self.run_risk_monitor())
        ]
        
        await asyncio.gather(*tasks)

    async def main_monitoring_loop(self):
        while self.is_running:
            try:
                await self.perform_monitoring_cycle()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logging.error(f"Error in main monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def stop(self):
        self.is_running = False
        await self.oracle.close()
        self.db.close()
        logging.info("Strategy monitor stopped")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pstr_monitor.log'),
            logging.StreamHandler()
        ]
    )
    
    config = {
        "total_capital": 1000000,
        "accumulation_period": 365,
        "treasury_allocation": 0.90,
        "liquidity_allocation": 0.10,
        "min_reserve_ratio": 3.0,
        "max_pump_concentration": 0.80,
        "trigger_threshold": 0.85,
        "cooldown_hours": 48,
        "max_slippage": 0.02,
        "gas_estimate": 100,
        "monitoring_interval": 300,
        "pump_balance": 5000000,
        "pstr_total_supply": 10000000,
        "db_path": "pstr_monitor.db",
        "pyth_endpoint": "https://hermes.pyth.network",
        "switchboard_endpoint": "https://api.switchboard.xyz",
        "pstr_feeds": {
            "pyth": "pstr_feed_id",
            "switchboard": "pstr_aggregator",
            "dex_pool": "pstr_pool_address"
        },
        "pump_feeds": {
            "pyth": "pump_feed_id",
            "switchboard": "pump_aggregator",
            "dex_pool": "pump_pool_address"
        },
        "pool_ids": ["main_pool_id"],
        "main_pool_id": "main_pool_address",
        "sol_pool_weight": 0.60,
        "usdc_pool_weight": 0.40,
        "concentration_range": 0.10,
        "rebalance_threshold": 0.15,
        "weight_rr": 0.333,
        "weight_ic": 0.333,
        "weight_conc": 0.334,
        "max_daily_interventions": 1,
        "max_deployment": 0.40,
        "min_rr_threshold": 1.5,
        "max_concentration": 0.80,
        "alert_rr_critical": 1.5,
        "alert_rr_warning": 2.0,
        "alert_drawdown": 25.0,
        "alert_oracle_dev": 0.05,
        "alert_liquidity": 100000
    }
    
    monitor = StrategyMonitor(config)
    
    try:
        asyncio.run(monitor.start())
    except KeyboardInterrupt:
        logging.info("Received shutdown signal")
        asyncio.run(monitor.stop())


if __name__ == "__main__":
    main()
