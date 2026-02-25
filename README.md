# news-sentiment-volatility
News sentiment and stock market volatility analysis (2020–2025)
# News Sentiment as a Factor Influencing Stock Market Volatility (2020–2025)
## About

This project investigates whether negative news sentiment has a stronger impact 
on stock market volatility than positive sentiment, with a focus on high-uncertainty 
periods.
## Dataset
- **Period:** January 2020 – January 2026  
- **Assets:** 15 U.S. stocks & ETFs: SPY, AAPL, MSFT, GOOGL, META, AMZN, NVDA, TSLA, 
  AMD, NFLX, JPM, GS, JNJ, XOM, DIS  
- **Rows:** ~22,300 ticker-day observations  
- **Sources:** Yahoo Finance · Polygon.io (66k+ articles) · CNN Fear & Greed · Google Trends

## Results Summary

| Hypothesis | Result |
|---|---|
| H1: Asymmetric sentiment effect | ✅ Supported (t=2.14, p=0.016, d=0.157) |
| H2: Crisis amplification | ✅ Supported (stress slope r=+0.099 vs calm r=−0.027) |
| H3: XGBoost improvement | ❌ Not supported (ΔAUC=−0.017) |

## How to Run

**Option 1 — Google Colab (recommended):**  
Open `news_sentiment_final.ipynb` directly in Colab.

**Option 2 — Local:**
```bash
pip install -r requirements.txt
python parse_and_prepare_dataset.py 
jupyter notebook news_sentiment_final.ipynb
```
## Note on API Keys
The `parse_and_prepare_dataset.py` script requires a **Polygon.io API key** 
for news data collection. 
