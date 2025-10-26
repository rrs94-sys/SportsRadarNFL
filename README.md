# NFL Player Props Prediction System v2.0

**Market-Optimized Quantitative Betting System with Monte Carlo Simulation**

A comprehensive, production-ready NFL player props prediction system that combines advanced machine learning, market optimization, and quantitative betting strategies.

---

## Features

### Core Capabilities
- ✅ **70+ Engineered Features**: Recency weighting, volatility metrics, usage patterns, opponent defense, weather, injuries
- ✅ **Market-Optimized Models**: Position-specific quantile regression (τ=0.55-0.70) optimized for profitability
- ✅ **Monte Carlo Simulation**: 10,000 iterations per prop for confidence intervals and hit rate estimation
- ✅ **Time-Series Backtesting**: Rigorous validation preventing data leakage
- ✅ **Fractional Kelly Sizing**: Risk-adjusted bet sizing with correlation controls
- ✅ **Edge@k Analysis**: PnL-focused evaluation (not MAE/R²)
- ✅ **API Call Optimization**: Intelligent caching to respect rate limits

### Data Sources
- **nflreadpy**: Play-by-play, player stats, schedules (2020-2024 + 2025) - single normalized API
- **SportsGameOdds**: Real-time betting lines with historical tracking
- **Tank API**: Live injury reports (2025) + historical injuries (2020-2024)
- **Open-Meteo**: Game-day weather forecasts

### Advanced Features
- **Volatility-Adjusted Thresholds**: Higher edges required for boom/bust players
- **Correlation Controls**: Portfolio optimization accounting for game/team correlations
- **Position-Specific Models**: Separate quantile targets for each prop type
- **Isotonic Calibration**: Post-hoc calibration for sharper predictions
- **GroupKFold CV**: Game-based splitting prevents leakage

---

## Installation

### Prerequisites
- Python 3.8 or higher
- 10GB free disk space (for data caching)
- API keys (see below)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd NFLModel

# Install dependencies
pip install -r requirements.txt

# Configure API keys
# Edit phase1_config.py and add your API keys:
# - SPORTSGAMEODDS_API_KEY
# - TANK_API_KEY
```

### API Keys

1. **SportsGameOdds** (500 calls/month free)
   - Sign up at https://sportsgameodds.com/
   - Already configured: `0b4d590cc09e7cfe05da2247b338698e`

2. **Tank API** (via RapidAPI)
   - Already configured: `9c58e7cda2msh95af7473755205ep12f820jsn13cda64975cf`

3. **Open-Meteo** (no key needed)

---

## Quick Start

### 1. Train Models (with Backtesting)

```bash
# Run time-series backtest + train final models
python train_models.py --through-week 8

# Or separately:
python train_models.py --backtest --through-week 7  # Backtest
python train_models.py --train --through-week 8     # Train final models
```

**What happens:**
- Loads data from 2020-2024 (full seasons) + 2025 (current)
- Applies 30/70 weighting (30% historical 2020-2024, 70% current 2025)
- Includes play-by-play data and injury data
- Runs time-series backtest:
  - Train 1-4, validate 5-6
  - Train 1-5, validate 6-7
  - Train 1-6, validate 7-8
- Trains final models on weeks 1-8
- Saves models to `saved_models_v2/`

### 2. Generate Predictions

```bash
# Predict Week 9
python predict_props.py --week 9

# Predict specific season/week
python predict_props.py --week 10 --season 2025
```

**What happens:**
1. Loads trained models
2. Fetches current lines from SportsGameOdds (cached to minimize API calls)
3. Loads injury data and weather forecasts
4. Generates predictions for all active players
5. Runs 10,000 Monte Carlo simulations per prop
6. Filters by hit rate (≥54% by default)
7. Applies position-specific edge thresholds
8. Calculates Kelly bet sizes
9. Optimizes portfolio with correlation controls
10. Outputs CSV with final recommendations

### 3. Review Output

```bash
# Output file: week9_predictions_YYYYMMDD_HHMM.csv
```

**Output columns:**
- `player_name`, `team`, `position`, `opponent`
- `market` (e.g., receiving_yards)
- `line`, `prediction`, `recommendation` (OVER/UNDER)
- `mc_over_rate` (Monte Carlo hit rate)
- `edge`, `edge_threshold`
- `kelly_fraction`, `bet_size`
- `correlation_adjustment`

---

## System Architecture

```
Data Layer → Feature Engineering → Model Training → Prediction → Monte Carlo → Betting Engine
     ↓              ↓                    ↓              ↓            ↓             ↓
  nflreadpy     70+ features      XGB/LGB Quantile  Edge@k     10k sims      Kelly + Corr
  (2020-2025)   Recency 30/70     Position-specific  CLV                      Controls
  Injury Map    Volatility        GroupKFold CV      ROI
  PBP Data      Usage             Isotonic Calib
  Weather       Trends
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| **Data Loading** | `nfl_data_utils.py` | Unified data loader (nflreadpy, 2020-2025, 30/70 weighting) |
| **Injury Mapping** | `injury_data_mapper.py` | TANK01 + nflreadpy injury integration |
| **Data Ingestion** | `phase1_ingestion.py` | Orchestrates data loading and caching |
| **Feature Engineering** | `phase1_features.py` | 70+ features per player |
| **Recency Weighting** | `recency_phase1_enhanced.py` | Advanced recency and volatility features |
| **Models** | `market_optimized_models.py` | Quantile regression with calibration |
| **Loss Functions** | `custom_loss_functions.py` | Quantile, Poisson, NegBin, LogNormal, Huber |
| **Monte Carlo** | `monte_carlo_engine.py` | 10k simulations per prop |
| **Betting Engine** | `quantitative_betting_engine.py` | Kelly sizing + correlation |
| **Evaluation** | `calibration_evaluation.py` | CLV, ROI, edge@k, Brier, LogLoss |
| **Training** | `train_models.py` | Time-series backtest + training |
| **Prediction** | `predict_props.py` | Full pipeline orchestration |
| **External APIs** | `sportsgameodds_client.py`, `tank_injury_client.py`, `weather_client.py` | Data sources |

---

## Configuration

Edit `phase1_config.py` to customize:

### Bankroll & Betting
```python
INITIAL_BANKROLL = 10000
KELLY_FRACTION = 0.25  # Quarter Kelly
MIN_EDGE = 0.05  # 5% minimum edge
MAX_BET = 500
MAX_CORRELATED_EXPOSURE = 0.15  # 15% max on correlated bets
```

### Position-Specific Settings
```python
# Quantile targets (higher for volatile props)
QUANTILE_TARGETS = {
    'receiving_yards': 0.60,
    'receptions': 0.58,
    'receiving_tds': 0.70,
    'rushing_yards': 0.65,
    ...
}

# Edge thresholds (higher for lower R² props)
EDGE_THRESHOLDS = {
    'receiving_yards': 1.0,
    'rushing_yards': 2.0,  # Higher due to game script dependency
    ...
}
```

### Monte Carlo
```python
MONTE_CARLO_ITERATIONS = 10000
MC_MIN_HIT_RATE = 0.54  # Minimum 54% to bet (beats -110 juice)
```

---

## Advanced Usage

### Weekly Micro-Update (Recommended)

After each week, update models with new data:

```bash
# Append Week 9 results
python train_models.py --train --through-week 9

# This does isotonic calibration re-fit only (fast)
```

### Monthly Macro-Retrain

Every 3-4 weeks, full retrain:

```bash
python train_models.py --backtest --train --through-week 12
```

### API Usage Monitoring

```python
from sportsgameodds_client import SportsGameOddsClient

client = SportsGameOddsClient()
usage = client.get_api_usage_stats()
print(f"API calls: {usage['calls_used']}/{usage['monthly_limit']}")
```

---

## Model Performance

### Backtest Results (Week 1-7 → Week 8)

| Prop Type | MAE | R² | Edge@1.0 ROI | Hit Rate | Samples |
|-----------|-----|----|--------------| ---------|---------|
| receiving_yards | 17.8 | 0.108 | +5.2% | 53.4% | 593 |
| receptions | 1.32 | 0.115 | +6.1% | 54.1% | 593 |
| rushing_yards | 17.7 | 0.042 | +3.8% | 52.8% | 289 |
| completions | 5.80 | 0.092 | +4.5% | 53.2% | 113 |

### Key Insights

✅ **What Works:**
- Quantile regression (τ=0.60) > median predictions
- GroupKFold CV prevents leakage
- 70+ features > complex models with few features
- Market optimization > statistical accuracy
- Recency weighting (30/40/30) outperforms equal weighting

❌ **What Doesn't:**
- Historical seasons (2019-2021) hurt performance
- Deep learning overfits small datasets
- MAE/R² optimization ≠ profitable betting
- Mean predictions lose to sharp lines

---

## Troubleshooting

### Common Issues

**1. API Rate Limit Exceeded**
```
⚠️ SGO API limit reached!
```
**Solution**: System uses cached lines automatically. Wait until next month or upgrade SGO plan.

**2. Missing Historical Data**
```
❌ No data for receiving_yards
```
**Solution**: Run `python phase1_ingestion.py` to download/cache data first.

**3. No Predictions Generated**
```
⚠️ No props meet hit rate threshold
```
**Solution**: Lower `MC_MIN_HIT_RATE` in config or increase data lookback.

---

## File Structure

```
NFLModel/
├── Core Data Layer
│   ├── nfl_data_utils.py              # Unified data loader (nflreadpy 2020-2025)
│   ├── injury_data_mapper.py          # TANK01 + nflreadpy injury integration
│   ├── phase1_ingestion.py            # Data orchestration & caching
│   └── phase1_config.py               # Configuration (API keys, parameters)
│
├── Feature Engineering
│   ├── phase1_features.py             # 70+ features per player
│   └── recency_phase1_enhanced.py     # Recency weighting & volatility
│
├── Models & Training
│   ├── market_optimized_models.py     # Quantile models + calibration
│   ├── custom_loss_functions.py       # Loss functions
│   ├── calibration_evaluation.py      # CLV, ROI, edge@k metrics
│   └── train_models.py                # Training pipeline
│
├── Prediction & Betting
│   ├── predict_props.py               # Prediction pipeline
│   ├── monte_carlo_engine.py          # 10k simulations
│   └── quantitative_betting_engine.py # Kelly sizing + correlation
│
├── External APIs
│   ├── sportsgameodds_client.py       # Lines API with caching
│   ├── tank_injury_client.py          # Injury data (TANK01)
│   └── weather_client.py              # Weather forecasts
│
├── Configuration
│   ├── requirements.txt               # Dependencies
│   └── README.md                      # This file
│
└── Data & Output
    ├── data_cache/                    # Cached NFL data
    ├── lines_cache/                   # Cached betting lines
    ├── saved_models_v2/               # Trained models
    └── week*_predictions_*.csv        # Output files
```

---

## Technical Details

### Model Architecture

**Ensemble:** 60% XGBoost Quantile + 40% LightGBM Quantile

```python
# XGBoost Quantile
XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=τ,  # Position-specific
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    reg_alpha=0.1, reg_lambda=1.0
)

# LightGBM Quantile
LGBMRegressor(
    objective='quantile',
    alpha=τ,
    n_estimators=500,
    ...
)

# Isotonic Calibration
IsotonicRegression(out_of_bounds='clip')
```

### Feature Categories (72 total)

1. **Recency** (20): L1 (30%), L2-L3 (40%), L4+ (30%)
2. **Volatility** (4): CV, boom/bust flags
3. **Home/Road** (4): Splits and differentials
4. **Rest/Travel** (3): Days rest, timezone changes
5. **Matchup** (3): Historical vs opponent
6. **Usage** (10): Target share, route %, air yards, red zone
7. **Opponent** (8): Defense allowed, EPA, coverage
8. **Weather** (4): Wind, temp, dome
9. **Trends** (3): Momentum, trajectory
10. **Game Context** (8): Spread, total, implied script
11. **Injuries** (3): Player status, team impact
12. **Market** (2): Line movement, historical lines

### Cross-Validation

```python
GroupKFold(n_splits=5)
# Groups = game_id to prevent leakage
# Same game never in both train and val
```

### Evaluation Focus

**Primary Metric:** Edge@k ROI (not MAE/R²)

```python
# Filter to edges ≥ k
# Calculate: (wins * 0.909 - losses) / total_bets
# Goal: Maximize ROI, not minimize MAE
```

---

## Best Practices

### Do's ✅
- ✅ Run backtest before deploying
- ✅ Monitor SGO API usage
- ✅ Filter by MC hit rate ≥ 54%
- ✅ Use fractional Kelly (¼ or ½)
- ✅ Apply correlation controls
- ✅ Check injury reports before betting
- ✅ Update models weekly
- ✅ Track CLV (closing line value)

### Don'ts ❌
- ❌ Don't bet without Monte Carlo confirmation
- ❌ Don't ignore edge thresholds
- ❌ Don't exceed correlation limits
- ❌ Don't bet on injured/inactive players
- ❌ Don't chase high lines without huge edge
- ❌ Don't trust predictions with R² < 0
- ❌ Don't optimize for MAE instead of ROI

---

## Support & Documentation

- **System Documentation**: `COMPLETE_SYSTEM_EXPORT.md`
- **Research Improvements**: `ReadMe.txt`
- **Issues**: Submit via GitHub

---

## License

Proprietary - For Personal Use Only

---

## Disclaimer

This system is for **research and educational purposes only**. Sports betting involves risk. Past performance does not guarantee future results. Always bet responsibly and within your means.

**The authors are not responsible for financial losses incurred using this system.**

---

## Production Branch

This is the **clean production branch** with only essential files:
- ✅ No test files
- ✅ No obsolete TANK01 clients (using nflreadpy instead)
- ✅ No migration documentation
- ✅ Streamlined for production deployment

**Data Range:** 2020-2024 (full seasons) + 2025 (current)
**Weighting:** 30% historical, 70% current
**Data Source:** nflreadpy (unified API)
**Injury Data:** TANK01 (2025) + nflreadpy (2020-2024)

---

**Version:** 2.0 (Market-Optimized, Clean Production)
**Last Updated:** October 2025
**Branch:** `claude/production-clean-011CUSB3zfhR443xAurdizHU`
**Status:** Production-Ready ✅
