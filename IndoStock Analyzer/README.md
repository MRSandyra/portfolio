# 📊 Stock Analyzer — Personal Stock Analysis Tool

An automated stock analysis tool that combines **news sentiment**, **fundamental analysis**, and **technical analysis** to generate data-driven investment recommendations for IDX (Indonesia Stock Exchange) equities.

---

## ✨ Key Features

- **🔍 News Scraping** — Automatically fetches the latest headlines from IndoPremier using Selenium
- **💬 Sentiment Analysis** — Analyzes news sentiment per stock ticker using TextBlob
- **💰 Fundamental Analysis** — Evaluates PER, PBV, ROE, DER, and earnings growth via Yahoo Finance
- **📈 Technical Analysis** — Calculates MA, RSI, MACD, Support/Resistance levels, and volume surges
- **🎯 Final Recommendation** — A weighted composite score across all three dimensions produces signals ranging from Strong Buy to Strong Sell

---

## 📋 Prerequisites

- Python 3.8+
- Google Chrome (latest version)
- ChromeDriver (matching your Chrome version)

---

## 🚀 Installation

**1. Clone the repository:**
```bash
git clone https://github.com/username/stock-analyzer.git
cd stock-analyzer
```

**2. Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**3. Download ChromeDriver:**

Download the ChromeDriver that matches your Chrome version from [chromedriver.chromium.org](https://chromedriver.chromium.org/downloads), then place the file at:
```
stock-analyzer/
└── drivers/
    └── chromedriver.exe   ← place it here
```

---

## 📦 Dependencies

Create a `requirements.txt` file with the following:

```
requests
beautifulsoup4
numpy
textblob
selenium
yfinance
pandas
```

---

## ▶️ Usage

```bash
python stock_analyzer.py
```

The program runs through 4 stages automatically:

| Stage | Process |
|-------|---------|
| 1 | Scrape latest news from IndoPremier |
| 2 | Run sentiment analysis per stock ticker |
| 3 | Fetch and evaluate fundamental + technical data |
| 4 | Generate report and save results to JSON |

---

## 📁 Output Structure

After running, the program creates a JSON file named:
```
stock_analysis_YYYYMMDD_HHMMSS.json
```

Example output:
```json
{
  "BBRI": {
    "sentiment_score": 3.6,
    "sentiment_articles": 4,
    "fundamental": { "score": 4, "grade": "B (Buy)" },
    "technical": { "score": 4, "grade": "B (Buy)" },
    "final_recommendation": "✅ BUY",
    "final_score": 4.02,
    "final_grade": "A-"
  }
}
```

---

## 🧮 Scoring Methodology

### Final Recommendation Weights

| Dimension | Weight | Rationale |
|-----------|--------|-----------|
| News Sentiment | 20% | Confirmation from news catalysts |
| Fundamental | 50% | Primary driver for long-term value |
| Technical | 30% | Optimal entry/exit timing |

### Recommendation Scale

| Final Score | Recommendation |
|------------|----------------|
| ≥ 4.2 | 🔥 STRONG BUY |
| ≥ 3.7 | ✅ BUY |
| ≥ 3.0 | 😐 HOLD |
| ≥ 2.3 | ⚠️ SELL |
| < 2.3 | ❌ STRONG SELL |

### Technical Indicators

| Indicator | Bullish Signal | Bearish Signal |
|-----------|---------------|----------------|
| MA 20/50 | Golden Cross | Death Cross |
| RSI | < 30 (Oversold) | > 70 (Overbought) |
| MACD | Histogram > 0 | Histogram < 0 |
| Volume | Surge + price up | Surge + price down |
| Price Position | Near support | Near resistance |

---

## ⚙️ Configuration

You can customize several parameters inside the `StockAnalyzer` class:

```python
self.max_news = 30                                    # Maximum number of news articles to fetch
self.chrome_driver_path = "drivers/chromedriver.exe"  # Path to ChromeDriver
```

To expand the list of recognized stock tickers, edit the `known_stocks` list inside `extract_stock_codes()`:

```python
known_stocks = ['BMRI', 'BBCA', 'TLKM', ...]  # add more tickers here
```

---

## ⚠️ Important Notes

> **Disclaimer**: This tool is built for personal research and educational purposes only. The recommendations it generates **do not constitute professional financial advice**. Always conduct your own due diligence and consult a qualified financial advisor before making any investment decisions.

- Sentiment analysis quality depends on TextBlob's ability to process Indonesian-language text — accuracy may be limited for purely Indonesian headlines
- Fundamental and technical data is sourced from Yahoo Finance using the `.JK` suffix for IDX stocks; data availability is not guaranteed for all issuers
- Ensure a stable internet connection before running the program

---

## 🗂️ Project Structure

```
stock-analyzer/
├── stock_analyzer.py       # Main script
├── drivers/
│   └── chromedriver.exe    # ChromeDriver (not included in repo)
├── requirements.txt        # Python dependencies
└── README.md               # This documentation
```

---

## 📄 License

This is a personal project, free to use for individual purposes.
