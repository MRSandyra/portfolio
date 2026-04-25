# stock_analyzer.py
# Tools analisis saham pribadi - menggabungkan sentimen berita, fundamental, dan teknikal
# Author: [Nama Anda]
# Last updated: 25 April 2026

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import json
from collections import defaultdict
import numpy as np
from textblob import TextBlob
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import os
import yfinance as yf
import pandas as pd

class StockAnalyzer:
    def __init__(self):
        self.results = []
        self.sentiment_history = defaultdict(list)
        self.max_news = 30
        self.chrome_driver_path = os.path.join(os.getcwd(), "drivers", "chromedriver.exe")
        
    # ==================== 1. NEWS SCRAPING ====================
    def scrape_indopremier_news(self):
        """Mengambil daftar berita terbaru dari IndoPremier"""
        print("📥 Mengambil berita dari IndoPremier...")
        
        if not os.path.exists(self.chrome_driver_path):
            print("⚠️ ChromeDriver tidak ditemukan di folder drivers/")
            return []
        
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1920,1080")
        
        service = Service(executable_path=self.chrome_driver_path)
        driver = webdriver.Chrome(service=service, options=options)
        
        news_list = []
        
        try:
            url = "https://www.indopremier.com/ipotnews/newsPages.php?level4=topnews"
            driver.get(url)
            time.sleep(5)
            
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            items = driver.find_elements(By.CSS_SELECTOR, "dl.listNews")
            print(f"   Ditemukan {len(items)} berita")
            
            for item in items[:self.max_news]:
                try:
                    link_elem = item.find_element(By.TAG_NAME, "a")
                    title = link_elem.text.strip()
                    if title and len(title) > 15:
                        news_list.append(title)
                except:
                    continue
                    
        except Exception as e:
            print(f"   Error scraping: {e}")
        finally:
            driver.quit()
        
        return news_list
    
    def extract_stock_codes(self, text):
        """Mengekstrak kode saham (4 huruf besar) dari teks berita"""
        stocks = re.findall(r'\b([A-Z]{4})\b', text)
        
        # Filter kata umum yang bukan kode saham
        common_words = ['DARI', 'UNTUK', 'YANG', 'DAN', 'INI', 'ITU', 'WIB', 'JAM',
                       'IHSG', 'EMAS', 'MINYAK', 'ASIA', 'JUTA', 'MILAR', 'RI', 'US',
                       'BULAN', 'TAHUN', 'HARI', 'SAHAM', 'INDEKS', 'RUPS']
        
        stocks = [s for s in stocks if s not in common_words]
        
        # Daftar kode saham yang dikenal (bisa ditambah sesuai kebutuhan)
        known_stocks = ['BMRI', 'ADHI', 'AUTO', 'TOBA', 'BAPA', 'COAL', 'ESSA',
                       'BBNI', 'BBRI', 'BBCA', 'TLKM', 'ASII', 'AVIA', 'GOOD',
                       'AKRA', 'ARTO', 'AMRT', 'MAPI', 'ACES', 'JPFA', 'BTPN',
                       'SDRA', 'WSKT', 'IPCC']
        
        for ks in known_stocks:
            if ks in text and ks not in stocks:
                stocks.append(ks)
        
        return list(set(stocks))
    
    # ==================== 2. SENTIMENT ANALYSIS ====================
    def analyze_sentiment(self, text):
        """Analisis sentimen menggunakan TextBlob, mengembalikan polarity (-1 s/d +1) dan klasifikasi"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.2:
            sentiment_class = 'POSITIF'
        elif polarity < -0.2:
            sentiment_class = 'NEGATIF'
        else:
            sentiment_class = 'NETRAL'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment_class': sentiment_class
        }
    
    def process_news_sentiment(self, news_list):
        """Memproses semua berita dan mengelompokkan sentimen per kode saham"""
        print("\n🔍 Menganalisis sentimen berita...")
        print("-"*70)
        
        for news in news_list:
            sentiment = self.analyze_sentiment(news)
            stocks = self.extract_stock_codes(news)
            
            if stocks:
                for stock in stocks:
                    self.sentiment_history[stock].append(sentiment)
                    self.results.append({
                        'tanggal': datetime.now().strftime('%Y-%m-%d'),
                        'judul': news,
                        'kode_saham': stock,
                        'polarity': sentiment['polarity'],
                        'sentimen': sentiment['sentiment_class']
                    })
                    print(f"   {stock}: {sentiment['sentiment_class']} (polarity: {sentiment['polarity']:+.2f})")
        
        print(f"\n✅ Selesai. {len(self.results)} entri sentimen dari {len(self.sentiment_history)} saham")
    
    # ==================== 3. FUNDAMENTAL ANALYSIS ====================
    def get_fundamental_data(self, stock_code):
        """Mengambil data fundamental dari Yahoo Finance (PER, PBV, ROE, DER, growth)"""
        try:
            ticker = f"{stock_code}.JK"
            stock = yf.Ticker(ticker)
            info = stock.info
            
            fundamental = {
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'pe_ratio': info.get('trailingPE', None),
                'pb_ratio': info.get('priceToBook', None),
                'roe': info.get('returnOnEquity', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'earnings_growth': info.get('earningsGrowth', None),
                'dividend_yield': info.get('dividendYield', None),
                'market_cap': info.get('marketCap', None),
            }
            
            return {k: v for k, v in fundamental.items() if v is not None}
            
        except Exception as e:
            print(f"      ⚠️ Gagal ambil fundamental {stock_code}: {e}")
            return None
    
    def rate_fundamental(self, fundamental):
        """Memberikan rating fundamental (1-5) berdasarkan metrik kunci"""
        if not fundamental:
            return {'score': 3, 'grade': 'C', 'description': 'Data tidak tersedia'}
        
        score = 3
        reasons = []
        
        # PER (Price to Earnings)
        pe = fundamental.get('pe_ratio')
        if pe:
            if pe < 10:
                score += 1
                reasons.append(f"PER rendah ({pe:.1f}x) → undervalued")
            elif pe > 25:
                score -= 1
                reasons.append(f"PER tinggi ({pe:.1f}x) → overvalued")
            else:
                reasons.append(f"PER wajar ({pe:.1f}x)")
        
        # PBV (Price to Book Value)
        pb = fundamental.get('pb_ratio')
        if pb:
            if pb < 1:
                score += 1
                reasons.append(f"PBV < 1 ({pb:.2f}x) → aset murah")
            elif pb > 3:
                score -= 1
                reasons.append(f"PBV tinggi ({pb:.2f}x)")
        
        # ROE (Return on Equity)
        roe = fundamental.get('roe')
        if roe:
            if roe > 0.15:
                score += 1
                reasons.append(f"ROE tinggi ({roe:.1%}) → efisien")
            elif roe < 0.05:
                score -= 1
                reasons.append(f"ROE rendah ({roe:.1%}) → tidak efisien")
        
        # Debt to Equity
        der = fundamental.get('debt_to_equity')
        if der:
            if der < 0.5:
                score += 0.5
                reasons.append(f"DER rendah ({der:.2f}) → sehat")
            elif der > 2:
                score -= 0.5
                reasons.append(f"DER tinggi ({der:.2f}) → berisiko")
        
        # Earnings Growth
        growth = fundamental.get('earnings_growth')
        if growth:
            if growth > 0.1:
                score += 1
                reasons.append(f"Growth positif ({growth:.1%})")
            elif growth < -0.05:
                score -= 1
                reasons.append(f"Growth negatif ({growth:.1%})")
        
        score = max(1, min(5, score))
        
        grade_map = {5: 'A (Strong Buy)', 4: 'B (Buy)', 3: 'C (Hold)', 
                     2: 'D (Sell)', 1: 'E (Strong Sell)'}
        
        return {
            'score': score,
            'grade': grade_map.get(score, 'C'),
            'description': ', '.join(reasons[:5])
        }
    
    # ==================== 4. TECHNICAL ANALYSIS ====================
    def get_technical_data(self, stock_code, period='1mo'):
        """Mengambil data teknikal dari Yahoo Finance (MA, RSI, MACD, Support/Resistance)"""
        try:
            ticker = f"{stock_code}.JK"
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval='1d')
            
            if hist.empty:
                return None
            
            close_prices = hist['Close']
            current_price = close_prices.iloc[-1]
            
            # Moving Averages
            ma_20 = close_prices.rolling(window=20).mean().iloc[-1] if len(close_prices) >= 20 else current_price
            ma_50 = close_prices.rolling(window=50).mean().iloc[-1] if len(close_prices) >= 50 else current_price
            
            ma_trend = 'BULLISH' if ma_20 > ma_50 else 'BEARISH' if ma_20 < ma_50 else 'NEUTRAL'
            
            # RSI (Relative Strength Index)
            delta = close_prices.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            # MACD (Moving Average Convergence Divergence)
            exp1 = close_prices.ewm(span=12, adjust=False).mean()
            exp2 = close_prices.ewm(span=26, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_histogram = macd_line - signal_line
            
            macd_signal = 'BULLISH' if macd_histogram.iloc[-1] > 0 else 'BEARISH' if macd_histogram.iloc[-1] < 0 else 'NEUTRAL'
            
            # Support & Resistance (20 hari terakhir)
            support = close_prices.tail(20).min()
            resistance = close_prices.tail(20).max()
            price_position = (current_price - support) / (resistance - support) if resistance != support else 0.5
            
            # Volume Analysis
            volumes = hist['Volume']
            avg_volume = volumes.mean()
            volume_surge = volumes.iloc[-1] > (avg_volume * 1.5)
            
            return {
                'current_price': current_price,
                'price_change_1d': ((close_prices.iloc[-1] - close_prices.iloc[-2]) / close_prices.iloc[-2]) * 100 if len(close_prices) > 1 else 0,
                'ma_trend': ma_trend,
                'rsi': current_rsi,
                'rsi_overbought': current_rsi > 70,
                'rsi_oversold': current_rsi < 30,
                'macd_signal': macd_signal,
                'support': support,
                'resistance': resistance,
                'price_position': price_position,
                'volume_surge': volume_surge,
            }
            
        except Exception as e:
            print(f"      ⚠️ Gagal ambil teknikal {stock_code}: {e}")
            return None
    
    def rate_technical(self, technical):
        """Memberikan rating teknikal (1-5)"""
        if not technical:
            return {'score': 3, 'grade': 'C', 'description': 'Data tidak tersedia'}
        
        score = 3
        reasons = []
        
        # MA Trend
        if technical['ma_trend'] == 'BULLISH':
            score += 1
            reasons.append("Golden Cross (MA bullish)")
        elif technical['ma_trend'] == 'BEARISH':
            score -= 1
            reasons.append("Death Cross (MA bearish)")
        
        # RSI
        if technical['rsi_oversold']:
            score += 1
            reasons.append(f"RSI Oversold ({technical['rsi']:.0f}) → potensi rebound")
        elif technical['rsi_overbought']:
            score -= 1
            reasons.append(f"RSI Overbought ({technical['rsi']:.0f}) → waspadai koreksi")
        
        # MACD
        if technical['macd_signal'] == 'BULLISH':
            score += 0.5
            reasons.append("MACD bullish momentum")
        elif technical['macd_signal'] == 'BEARISH':
            score -= 0.5
            reasons.append("MACD bearish momentum")
        
        # Volume
        if technical['volume_surge']:
            if technical['price_change_1d'] > 0:
                score += 0.5
                reasons.append("Volume tinggi + harga naik → valid bullish")
            else:
                score -= 0.5
                reasons.append("Volume tinggi + harga turun → panic selling")
        
        # Position (Support/Resistance)
        pos = technical['price_position']
        if pos < 0.2:
            score += 0.5
            reasons.append("Harga mendekati support → area beli")
        elif pos > 0.8:
            score -= 0.5
            reasons.append("Harga mendekati resistance → area jual")
        
        score = max(1, min(5, score))
        
        grade_map = {5: 'A (Strong Buy)', 4: 'B (Buy)', 3: 'C (Hold)', 
                     2: 'D (Sell)', 1: 'E (Strong Sell)'}
        
        return {
            'score': score,
            'grade': grade_map.get(score, 'C'),
            'description': ', '.join(reasons[:5])
        }
    
    # ==================== 5. FINAL RECOMMENDATION ====================
    def get_sentiment_score(self, stock_code):
        """Menghitung skor sentimen (1-5) berdasarkan polarity rata-rata"""
        sentiments = self.sentiment_history.get(stock_code, [])
        
        if not sentiments:
            return 3.0  # Netral jika tidak ada berita
        
        avg_polarity = np.mean([s['polarity'] for s in sentiments])
        sentiment_score = 3 + (avg_polarity * 2)  # Konversi polarity (-1..1) ke skor (1..5)
        return max(1, min(5, sentiment_score))
    
    def generate_final_recommendation(self, sentiment_score, fundamental_score, technical_score):
        """
        Menghasilkan rekomendasi akhir dengan bobot:
        - Sentimen: 20% (konfirmasi dari berita)
        - Fundamental: 50% (prioritas utama untuk jangka panjang)
        - Teknikal: 30% (timing entry/exit)
        """
        weights = {
            'sentimen': 0.2,
            'fundamental': 0.5,
            'teknikal': 0.3
        }
        
        final_score = (sentiment_score * weights['sentimen'] + 
                      fundamental_score * weights['fundamental'] + 
                      technical_score * weights['teknikal'])
        
        if final_score >= 4.2:
            return '🔥 STRONG BUY', 'A+', final_score
        elif final_score >= 3.7:
            return '✅ BUY', 'A-', final_score
        elif final_score >= 3.0:
            return '😐 HOLD', 'C', final_score
        elif final_score >= 2.3:
            return '⚠️ SELL', 'D', final_score
        else:
            return '❌ STRONG SELL', 'F', final_score
    
    # ==================== 6. MAIN EXECUTION ====================
    def analyze_all_stocks(self):
        """Menganalisis semua saham yang muncul di berita"""
        stock_list = list(self.sentiment_history.keys())
        
        if not stock_list:
            print("\n⚠️ Tidak ada kode saham ditemukan")
            return {}
        
        analysis_result = {}
        
        for stock in stock_list:
            print(f"\n{'='*50}")
            print(f"📊 ANALISIS SAHAM: {stock}")
            print('='*50)
            
            # Sentimen score
            sentiment_score = self.get_sentiment_score(stock)
            print(f"\n📰 SENTIMEN: Score {sentiment_score:.1f}/5 (dari {len(self.sentiment_history[stock])} artikel)")
            
            # Fundamental analysis
            fundamental = self.get_fundamental_data(stock)
            fundamental_rating = self.rate_fundamental(fundamental)
            print(f"\n💰 FUNDAMENTAL: {fundamental_rating['description']}")
            print(f"   Grade: {fundamental_rating['grade']} (Score: {fundamental_rating['score']}/5)")
            
            # Technical analysis
            technical = self.get_technical_data(stock)
            technical_rating = self.rate_technical(technical)
            print(f"\n📈 TEKNIKAL: {technical_rating['description']}")
            print(f"   Grade: {technical_rating['grade']} (Score: {technical_rating['score']}/5)")
            
            # Final recommendation
            final_rec, final_grade, final_score = self.generate_final_recommendation(
                sentiment_score, fundamental_rating['score'], technical_rating['score']
            )
            
            print(f"\n🎯 REKOMENDASI FINAL: {final_rec}")
            print(f"   Final Score: {final_score:.2f}/5 | Grade: {final_grade}")
            
            analysis_result[stock] = {
                'sentiment_score': sentiment_score,
                'sentiment_articles': len(self.sentiment_history[stock]),
                'fundamental': {'score': fundamental_rating['score'], 'grade': fundamental_rating['grade']},
                'technical': {'score': technical_rating['score'], 'grade': technical_rating['grade']},
                'final_recommendation': final_rec,
                'final_score': final_score,
                'final_grade': final_grade
            }
        
        return analysis_result
    
    def run(self):
        """Main entry point untuk menjalankan seluruh proses analisis"""
        print("\n" + "="*80)
        print("📊 COMPLETE STOCK ANALYZER")
        print("   Sentiment + Fundamental + Technical Analysis")
        print("="*80)
        print(f"📅 {datetime.now().strftime('%A, %d %B %Y')}")
        print(f"⏰ {datetime.now().strftime('%H:%M:%S')}\n")
        
        # Step 1: Scrape news from IndoPremier
        news_list = self.scrape_indopremier_news()
        print(f"\n📊 Total news: {len(news_list)}")
        
        # Step 2: Process sentiment from news
        self.process_news_sentiment(news_list)
        
        # Step 3: Analyze all stocks
        analysis = self.analyze_all_stocks()
        
        if not analysis:
            print("\n⚠️ No stocks found for analysis")
            return
        
        # Step 4: Final report
        print("\n" + "="*80)
        print("📈 FINAL STOCK RECOMMENDATION REPORT")
        print("="*80)
        
        sorted_analysis = sorted(analysis.items(), key=lambda x: x[1]['final_score'], reverse=True)
        
        for stock, data in sorted_analysis:
            final = data['final_recommendation']
            emoji = '🚀' if 'STRONG BUY' in final else '📈' if 'BUY' in final else '⚠️' if 'SELL' in final else '😐'
            
            print(f"\n{emoji} {stock}")
            print(f"   Final: {final} (Score: {data['final_score']:.2f})")
            print(f"   Sentiment: {data['sentiment_score']:.1f}/5 | Fundamental: {data['fundamental']['grade']} | Technical: {data['technical']['grade']}")
        
        # Save results to JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'stock_analysis_{timestamp}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Data saved to: {filename}")
        print("\n✨ Analysis complete!")

if __name__ == "__main__":
    analyzer = StockAnalyzer()
    analyzer.run()