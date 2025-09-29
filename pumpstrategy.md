# PSTR Protocol - Strategy & Calculations

## Table of Contents
1. [Overview](#overview)
2. [Treasury Accumulation Strategy](#treasury-accumulation-strategy)
3. [Price Monitoring & Trigger Calculations](#price-monitoring--trigger-calculations)
4. [Intervention Deployment Calculations](#intervention-deployment-calculations)
5. [Reserve Health Metrics](#reserve-health-metrics)
6. [Liquidity Management Calculations](#liquidity-management-calculations)
7. [Recharge Mechanism](#recharge-mechanism)
8. [Advanced Scenarios](#advanced-scenarios)

---

## Overview

PSTR Protocol uses mathematical formulas to determine when and how much capital to rotate between PUMP and PSTR tokens. This document provides detailed calculations for all strategic decisions.

---

## Treasury Accumulation Strategy

### Daily Accumulation Rate

The protocol accumulates PUMP tokens based on available capital and a time-based schedule.

**Formula:**

```
Daily_PUMP_Budget = Total_Available_Capital × 0.90 × Daily_Rate_Factor

where:
Daily_Rate_Factor = 1 / Accumulation_Period_Days
```

**Variables:**
- `Total_Available_Capital (C)`: Total capital allocated to protocol
- `0.90`: 90% allocation to treasury
- `Accumulation_Period_Days (T)`: Target accumulation timeframe (e.g., 365 days)

**Example Calculation:**

```
Given:
C = $1,000,000
T = 365 days

Daily_PUMP_Budget = $1,000,000 × 0.90 × (1/365)
                  = $900,000 × 0.00274
                  = $2,465.75 per day
```

### Hourly Micro-Purchase Amount

To minimize market impact, purchases are split into hourly intervals:

```
Hourly_Purchase_Amount = Daily_PUMP_Budget / 24

H = (C × 0.90 × (1/T)) / 24
```

**Example:**
```
H = $2,465.75 / 24
  = $102.74 per hour
```

### Cumulative Treasury Value

At any time `t` (in days), the accumulated PUMP value is:

```
Treasury_Value(t) = Σ(Daily_PUMP_Budget × PUMP_Price(i)) for i = 0 to t

For constant price:
TV(t) = Daily_PUMP_Budget × t × P_pump

For variable price with average:
TV(t) = Daily_PUMP_Budget × t × P_pump_avg
```

where:
- `P_pump`: Price of PUMP token
- `P_pump_avg`: Average PUMP price over period

---

## Price Monitoring & Trigger Calculations

### Volume-Weighted Average Price (VWAP)

The 30-day VWAP is calculated to smooth out short-term volatility:

```
VWAP_30 = Σ(P_i × V_i) / Σ(V_i)  for i = 1 to 30 days

where:
P_i = Price on day i
V_i = Volume on day i
```

**Detailed Calculation:**

```
VWAP_30 = (P₁×V₁ + P₂×V₂ + ... + P₃₀×V₃₀) / (V₁ + V₂ + ... + V₃₀)
```

**Example:**
```
Day 1: P₁ = $1.50, V₁ = 100,000
Day 2: P₂ = $1.48, V₂ = 120,000
Day 3: P₃ = $1.52, V₃ = 90,000

VWAP_3 = (1.50×100,000 + 1.48×120,000 + 1.52×90,000) / (100,000 + 120,000 + 90,000)
       = (150,000 + 177,600 + 136,800) / 310,000
       = 464,400 / 310,000
       = $1.498
```

### Intervention Trigger Ratio

The trigger determines when treasury intervention is needed:

```
Trigger_Ratio = P_current / VWAP_30

where:
P_current = Current PSTR spot price
```

**Intervention Condition:**

```
IF Trigger_Ratio < Threshold THEN
    Intervention_Required = TRUE
END IF

where:
Threshold = 0.85 (represents 15% drawdown)
```

### Drawdown Percentage Calculation

```
Drawdown_Percentage = ((VWAP_30 - P_current) / VWAP_30) × 100

D% = ((V₃₀ - P_c) / V₃₀) × 100
```

**Example:**
```
Given:
VWAP_30 = $2.00
P_current = $1.60

D% = ((2.00 - 1.60) / 2.00) × 100
   = (0.40 / 2.00) × 100
   = 0.20 × 100
   = 20%

Since 20% falls in the 20-30% range:
Deployment = 25% of reserves
```

---

## Intervention Deployment Calculations

### Tiered Deployment Formula

```
Deployment_Percentage = f(D%)

where f(D%) is defined as:

f(D%) = {
    0.10  if 15 ≤ D% < 20
    0.25  if 20 ≤ D% < 30
    0.40  if D% ≥ 30
    0     otherwise
}
```

### Capital Deployment Amount

```
Capital_Deployed = Treasury_Value × Deployment_Percentage

CD = TV × f(D%)
```

**Example Calculation:**

```
Given:
Treasury_Value = $3,000,000 (in PUMP)
Drawdown = 22%

Step 1: Determine deployment percentage
Since 20 ≤ 22 < 30:
f(22%) = 0.25

Step 2: Calculate capital deployed
CD = $3,000,000 × 0.25
   = $750,000
```

### PUMP Tokens to Sell

```
PUMP_Tokens_Sell = Capital_Deployed / P_pump_current

where:
P_pump_current = Current PUMP token price
```

**Example:**
```
Given:
Capital_Deployed = $750,000
P_pump_current = $0.50

PUMP_Tokens_Sell = $750,000 / $0.50
                 = 1,500,000 PUMP tokens
```

### PSTR Tokens to Purchase

After selling PUMP, the protocol buys PSTR:

```
PSTR_Buy_Amount = (Capital_Deployed × (1 - Slippage)) / P_pstr_current

where:
Slippage = Expected slippage percentage (e.g., 0.02 for 2%)
P_pstr_current = Current PSTR token price
```

**Example:**
```
Given:
Capital_Deployed = $750,000
Slippage = 0.02 (2%)
P_pstr_current = $1.60

PSTR_Buy_Amount = ($750,000 × (1 - 0.02)) / $1.60
                = ($750,000 × 0.98) / $1.60
                = $735,000 / $1.60
                = 459,375 PSTR tokens
```

### Expected Price Impact

The expected price impact from the buy pressure:

```
Price_Impact = (PSTR_Buy_Amount × P_pstr) / Liquidity_Depth

ΔP = (Q × P) / L

where:
Q = Quantity purchased
P = Current price
L = Available liquidity
```

**Example:**
```
Given:
PSTR_Buy_Amount = 459,375 tokens
P_pstr = $1.60
Liquidity_Depth = $2,000,000

Price_Impact = (459,375 × $1.60) / $2,000,000
             = $735,000 / $2,000,000
             = 0.3675 or 36.75%

Expected_New_Price = P_pstr × (1 + Price_Impact)
                   = $1.60 × 1.3675
                   = $2.188
```

---

## Reserve Health Metrics

### Reserve Ratio Calculation

```
Reserve_Ratio = Treasury_Value_USD / PSTR_Market_Cap

RR = TV / MC_pstr

where:
TV = Treasury value in USD
MC_pstr = PSTR market capitalization
```

**Market Cap Calculation:**
```
MC_pstr = Total_PSTR_Supply × P_pstr_current
```

**Example:**
```
Given:
Treasury_Value = $3,000,000
Total_PSTR_Supply = 10,000,000 tokens
P_pstr_current = $0.10

MC_pstr = 10,000,000 × $0.10
        = $1,000,000

Reserve_Ratio = $3,000,000 / $1,000,000
              = 3.0

This meets the minimum 3:1 requirement ✓
```

### Intervention Capacity

Calculate how many interventions can be supported:

```
Intervention_Capacity = Reserve_Ratio / Max_Deployment_Percentage

IC = RR / 0.40

Minimum acceptable: IC ≥ 3
```

**Example:**
```
Given:
Reserve_Ratio = 3.0
Max_Deployment = 0.40 (40%)

IC = 3.0 / 0.40
   = 7.5 interventions

This exceeds minimum of 3 ✓
```

### Reserve Health Score

Composite health metric:

```
Health_Score = w₁×(RR/3) + w₂×(IC/3) + w₃×(1-PUMP_Concentration)

where:
w₁, w₂, w₃ = Weights (w₁ + w₂ + w₃ = 1)
PUMP_Concentration = TV_pump / TV_total
```

**Example with equal weights:**
```
Given:
w₁ = w₂ = w₃ = 0.333
RR = 3.5
IC = 8
PUMP_Concentration = 0.75

Health_Score = 0.333×(3.5/3) + 0.333×(8/3) + 0.333×(1-0.75)
             = 0.333×1.167 + 0.333×2.667 + 0.333×0.25
             = 0.389 + 0.888 + 0.083
             = 1.36

Interpretation:
> 1.0 = Excellent health ✓
0.8-1.0 = Good health
0.6-0.8 = Moderate health
< 0.6 = Poor health - action required
```

---

## Liquidity Management Calculations

### Liquidity Allocation Formula

```
Liquidity_Budget = Total_Available_Capital × 0.10

L_budget = C × 0.10
```

**Pool Distribution:**
```
L_sol_pool = L_budget × Pool_Weight_SOL
L_usdc_pool = L_budget × Pool_Weight_USDC

where:
Pool_Weight_SOL = 0.60 (60%)
Pool_Weight_USDC = 0.40 (40%)
```

**Example:**
```
Given:
C = $1,000,000

L_budget = $1,000,000 × 0.10
         = $100,000

L_sol_pool = $100,000 × 0.60 = $60,000
L_usdc_pool = $100,000 × 0.40 = $40,000
```

### Concentrated Liquidity Range

For concentrated liquidity (Raydium CLMM):

```
Price_Lower = P_current × (1 - Range_Percentage)
Price_Upper = P_current × (1 + Range_Percentage)

where:
Range_Percentage = 0.10 (±10%)
```

**Example:**
```
Given:
P_current = $1.50
Range_Percentage = 0.10

Price_Lower = $1.50 × (1 - 0.10)
            = $1.50 × 0.90
            = $1.35

Price_Upper = $1.50 × (1 + 0.10)
            = $1.50 × 1.10
            = $1.65

Liquidity Range: $1.35 - $1.65
```

### Capital Efficiency

```
Capital_Efficiency = Liquidity_In_Range / Total_Liquidity_Provided

CE = L_active / L_total
```

**Example:**
```
Given:
Total_Liquidity_Provided = $100,000
Liquidity_In_Range = $100,000 (all in range)

CE = $100,000 / $100,000
   = 1.0 or 100%

Concentrated liquidity typically achieves:
- Traditional pools: ~50% efficiency
- Concentrated ±10%: ~90% efficiency
- Concentrated ±5%: ~95% efficiency
```

### Impermanent Loss Calculation

```
IL = 2×√(Price_Ratio) / (1 + Price_Ratio) - 1

where:
Price_Ratio = P_final / P_initial
```

**Example:**
```
Given:
P_initial = $1.50
P_final = $2.00

Price_Ratio = $2.00 / $1.50
            = 1.333

IL = 2×√(1.333) / (1 + 1.333) - 1
   = 2×1.155 / 2.333 - 1
   = 2.310 / 2.333 - 1
   = 0.990 - 1
   = -0.010 or -1.0%

This represents a 1% impermanent loss
```

---

## Recharge Mechanism

### Accelerated Accumulation Post-Intervention

After an intervention, the protocol accelerates PUMP accumulation to restore reserves:

```
Recharge_Multiplier = 1 + (Deployed_Amount / Remaining_Treasury)

Daily_Budget_Recharge = Daily_PUMP_Budget × Recharge_Multiplier

Maximum: Recharge_Multiplier ≤ 3.0
```

**Example:**
```
Given:
Daily_PUMP_Budget = $2,465.75
Deployed_Amount = $750,000
Remaining_Treasury = $2,250,000

Recharge_Multiplier = 1 + ($750,000 / $2,250,000)
                    = 1 + 0.333
                    = 1.333

Daily_Budget_Recharge = $2,465.75 × 1.333
                      = $3,286.83 per day

This represents a 33.3% increase in accumulation rate
```

### Time to Full Recharge

```
Days_To_Recharge = Deployed_Amount / Daily_Budget_Recharge

T_recharge = DA / (DB × RM)
```

**Example:**
```
Given:
Deployed_Amount = $750,000
Daily_Budget_Recharge = $3,286.83

T_recharge = $750,000 / $3,286.83
           = 228 days

At normal rate ($2,465.75/day):
T_normal = $750,000 / $2,465.75
         = 304 days

Savings: 304 - 228 = 76 days faster
```

---

## Advanced Scenarios

### Scenario 1: Cascading Drawdown

Multiple interventions in short succession:

```
Given Initial Conditions:
TV₀ = $3,000,000
MC_pstr = $1,000,000
RR₀ = 3.0

Intervention 1:
D%₁ = 20%
Deployment₁ = 0.25 × $3,000,000 = $750,000
TV₁ = $3,000,000 - $750,000 = $2,250,000
RR₁ = $2,250,000 / $1,000,000 = 2.25

48-hour cooldown...

Intervention 2 (if price drops again):
D%₂ = 25%
Deployment₂ = 0.25 × $2,250,000 = $562,500
TV₂ = $2,250,000 - $562,500 = $1,687,500
RR₂ = $1,687,500 / $1,000,000 = 1.69

Status: Still above minimum (1.69 < 2.0) - Yellow Alert
Action: Increase accumulation 2x, reduce next deployment to 50%
```

### Scenario 2: PUMP Price Surge

Treasury value increases due to PUMP appreciation:

```
Given:
Initial PUMP holdings: 5,000,000 tokens
P_pump_initial = $0.50
TV_initial = 5,000,000 × $0.50 = $2,500,000

After 50% PUMP surge:
P_pump_new = $0.75
TV_new = 5,000,000 × $0.75 = $3,750,000

Treasury Gain = $3,750,000 - $2,500,000
              = $1,250,000 (50% increase)

New Reserve Ratio:
MC_pstr = $1,000,000 (unchanged)
RR_new = $3,750,000 / $1,000,000
       = 3.75

Result: Increased intervention capacity from 7.5 to 9.375 interventions
```

### Scenario 3: Reserve Ratio Recovery

Calculate time to restore target reserve ratio:

```
Target_RR = 3.0
Current_RR = 1.8
MC_pstr = $1,000,000

Required_Treasury = Target_RR × MC_pstr
                  = 3.0 × $1,000,000
                  = $3,000,000

Current_Treasury = 1.8 × $1,000,000
                 = $1,800,000

Deficit = $3,000,000 - $1,800,000
        = $1,200,000

With accelerated accumulation:
Daily_Accumulation = $3,286.83

Days_To_Target = $1,200,000 / $3,286.83
               = 365 days

Alternative with fee increase (2x):
Daily_Accumulation_2x = $6,573.66
Days_To_Target = $1,200,000 / $6,573.66
               = 183 days
```

### Scenario 4: Optimal Intervention Timing

Calculate the optimal moment to intervene considering gas costs and price impact:

```
Intervention_Benefit = Expected_Price_Recovery - Execution_Cost

where:
Expected_Price_Recovery = Target_Price - Current_Price
Execution_Cost = Gas_Fees + Slippage_Cost

Slippage_Cost = Deployed_Amount × Slippage_Percentage
```

**Example:**
```
Given:
Current_Price = $1.60
Target_Price = $1.80 (back to 90% of VWAP)
Deployed_Amount = $750,000
Slippage = 2%
Gas_Fees = $100

Expected_Recovery = $1.80 - $1.60 = $0.20 per token
PSTR_Purchased ≈ 459,375 tokens
Total_Benefit = 459,375 × $0.20 = $91,875

Execution_Cost = $100 + ($750,000 × 0.02)
               = $100 + $15,000
               = $15,100

Net_Benefit = $91,875 - $15,100
            = $76,775

ROI = ($76,775 / $750,000) × 100
    = 10.24%

Decision: INTERVENE (positive expected value)
```

---

## Rotation Decision Matrix

### Complete Decision Algorithm

```
FUNCTION decide_rotation():
    
    // Step 1: Calculate current metrics
    P_current = get_pstr_price()
    VWAP_30 = calculate_vwap(30)
    Trigger_Ratio = P_current / VWAP_30
    
    // Step 2: Check cooldown
    IF time_since_last_intervention < 48_hours THEN
        RETURN NO_ACTION
    END IF
    
    // Step 3: Check trigger condition
    IF Trigger_Ratio >= 0.85 THEN
        RETURN NO_ACTION
    END IF
    
    // Step 4: Calculate drawdown
    D% = ((VWAP_30 - P_current) / VWAP_30) × 100
    
    // Step 5: Determine deployment percentage
    IF 15 ≤ D% < 20 THEN
        Deployment% = 0.10
    ELSE IF 20 ≤ D% < 30 THEN
        Deployment% = 0.25
    ELSE IF D% ≥ 30 THEN
        Deployment% = 0.40
    END IF
    
    // Step 6: Check reserve capacity
    TV = get_treasury_value()
    Capital_Deploy = TV × Deployment%
    Remaining = TV - Capital_Deploy
    
    MC_pstr = get_pstr_market_cap()
    New_RR = Remaining / MC_pstr
    
    IF New_RR < 1.5 THEN
        Deployment% = Deployment% × 0.5  // Halve deployment
    END IF
    
    // Step 7: Calculate expected impact
    Expected_Benefit = estimate_price_recovery()
    Execution_Cost = estimate_costs()
    
    IF Expected_Benefit > Execution_Cost THEN
        RETURN EXECUTE_INTERVENTION(Deployment%)
    ELSE
        RETURN NO_ACTION
    END IF
    
END FUNCTION
```

---

## Summary of Key Formulas

### Accumulation Phase
```
Daily_Budget = C × 0.90 × (1/T)
Hourly_Amount = Daily_Budget / 24
Treasury_Value(t) = Σ(Daily_Budget × P_pump(i))
```

### Monitoring Phase
```
VWAP_30 = Σ(P_i × V_i) / Σ(V_i)
Trigger_Ratio = P_current / VWAP_30
Drawdown% = ((VWAP_30 - P_current) / VWAP_30) × 100
```

### Intervention Phase
```
Deployment% = f(Drawdown%)
Capital_Deployed = TV × Deployment%
PUMP_Sell = Capital_Deployed / P_pump
PSTR_Buy = (Capital_Deployed × (1 - Slippage)) / P_pstr
```

### Reserve Health
```
Reserve_Ratio = TV / MC_pstr
Intervention_Capacity = RR / Max_Deployment%
Health_Score = Σ(w_i × Metric_i)
```

### Recharge Phase
```
Recharge_Multiplier = 1 + (Deployed / Remaining)
Daily_Recharge = Daily_Budget × RM
Days_To_Restore = Deployed / Daily_Recharge
```

---

## Appendix: Variable Definitions

| Symbol | Name | Unit | Description |
|--------|------|------|-------------|
| C | Total Capital | USD | Total available capital |
| T | Accumulation Period | days | Target timeframe for accumulation |
| TV | Treasury Value | USD | Current value of PUMP reserves |
| P_pstr | PSTR Price | USD | Current PSTR token price |
| P_pump | PUMP Price | USD | Current PUMP token price |
| VWAP_30 | 30-day VWAP | USD | Volume-weighted average price |
| D% | Drawdown Percentage | % | Price decline from VWAP |
| RR | Reserve Ratio | ratio | Treasury value to market cap ratio |
| MC_pstr | Market Cap | USD | PSTR total market capitalization |
| IC | Intervention Capacity | count | Number of possible interventions |
| CD | Capital Deployed | USD | Amount deployed in intervention |
| L_budget | Liquidity Budget | USD | Capital allocated to liquidity |
| RM | Recharge Multiplier | ratio | Acceleration factor post-intervention |

---

**Document Version:** 1.0  
**Last Updated:** September 2025  
**Next Review:** December 2025
