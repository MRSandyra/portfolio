<div align="center">

# 🌟 Rizky's Data Science Portfolio

[![License](https://img.shields.io/github/license/MRSandyra/portfolio?style=flat-square)](./LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/MRSandyra/portfolio?style=flat-square)](https://github.com/MRSandyra/portfolio/commits/main)
![Repo Size](https://img.shields.io/github/repo-size/MRSandyra/portfolio?style=flat-square)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-98.8%25-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f?style=flat-square&logo=python&logoColor=white)

**From raw data to deployed intelligence.**
A collection of end-to-end data science, machine learning, and applied AI projects — spanning finance, gaming analytics, NLP, medical imaging, cybersecurity, and socio-economics.

🌐 **[mrsandyra.netlify.app](https://mrsandyra.netlify.app/)**

</div>

---

## 👋 About

I build projects across the full data lifecycle: **data ingestion → cleaning & feature engineering → modelling → interpretation → deployment.** My work emphasises methodological rigor (data-leakage auditing, class balancing, robustness checks) as much as results, and several projects ship as working web applications rather than notebooks alone.

This repository collects my favourite projects in one place. Two of them (`lung-diagnosis` and `anime-recommender`) live in their own repos and are included here as **Git submodules**.

---

## 🗂️ Projects at a Glance

| Project | Focus Area | Key Tech |
|---|---|---|
| [League of Legends Analytics](#-league-of-legends-challenger-match-analytics--win-prediction) | ML · Prediction · Interpretability | scikit-learn, XGBoost, SHAP, NetworkX |
| [Lung Disease Detection](#-lung-disease-detection--deep-learning) | Deep Learning · Medical Imaging | TensorFlow/Keras, DenseNet-169, Laravel, Flask |
| [Sentiment Analysis (Tokopedia)](#-sentiment-analysis--tokopedia-reviews) | NLP · Text Classification | scikit-learn, Naive Bayes, CountVectorizer |
| [Anime Recommender](#-anime-recommender) | Recommender Systems | Python |
| [IndoStock Analyzer](#-indostock-analyzer--personal-stock-analysis-tool) | FinTech · Automation | Selenium, TextBlob, yfinance, Pandas |
| [Cybersecurity Anomaly Detection](#-cybersecurity--anomaly-detection-system) | Security · Log Analysis | Python, PHP, MySQL |
| [Global Income Inequality](#-global-income-inequality--lorenz-curves--gini-coefficients) | Data Viz · Clustering | Plotly, scikit-learn, NumPy |
| [Capstone: Flight Sales Analysis](#-capstone--flight-ticket-sales-analysis) | EDA · Data Cleaning | Pandas, Matplotlib, Seaborn |

---

## 🤖 Machine Learning & AI

### 🎮 League of Legends: Challenger Match Analytics & Win Prediction

> [`/League of Legends Challenger Rank Analytics`](./League%20of%20Legends%20Challenger%20Rank%20Analytics)

An end-to-end study of **300 Challenger-tier matches (~311K in-game events)**, streamed from a 10-million-event public dataset without ever fully loading it into memory. Covers the complete pipeline: **streaming ingestion → feature engineering → win prediction → model interpretation → player profiling → anomaly detection → champion-synergy mining.**

- Win prediction with Logistic Regression, Random Forest, and XGBoost, explained via **SHAP**
- A dedicated **data-leakage audit** and **patch-robustness** checks for honest evaluation
- **K-Means** player archetypes, **Isolation Forest** anomaly detection, and champion-pair synergy mining with NetworkX
- Automated text report + interactive win-probability dashboard

**Tech:** Python · scikit-learn · XGBoost · SHAP · Plotly · NetworkX

---

### 🫁 Lung Disease Detection — Deep Learning

> [`/lung-diagnosis`](https://github.com/MRSandyra/Lung-Diagnosis-Web)

A web-based diagnostic system that predicts lung conditions from **chest X-ray images + patient symptoms** using a multimodal **DenseNet-169** model. A Laravel (PHP) frontend integrates with a Flask (Python) microservice that serves the TensorFlow/Keras model in real time.

- Multimodal input: 224×224 X-ray image + 13 binary symptom features → **11-class** output
- Conditions include Cardiomegaly, Effusion, Fibrosis, Emphysema, Pneumothorax, and more
- Full application: user auth, image upload with validation, and per-user diagnosis history

**Tech:** TensorFlow/Keras · DenseNet-169 · Laravel 9 · Flask · MySQL/PostgreSQL

---

### 💬 Sentiment Analysis — Tokopedia Reviews

> [`/Sentiment Analysis Tokopedia`](./Sentiment%20Analysis%20Tokopedia)

An end-to-end **NLP pipeline** classifying real-world Indonesian e-commerce reviews (Tokopedia Food & Drink) into **Positive / Negative** sentiment, reaching **~83% accuracy** on a balanced dataset.

- 8-step text-preprocessing pipeline (noise removal, emoji stripping, deduplication)
- Class balancing via **downsampling** to prevent majority-class bias
- Bag-of-Words vectorization (`CountVectorizer`) paired with **Multinomial Naive Bayes**

**Tech:** Python · scikit-learn · Multinomial Naive Bayes · CountVectorizer

---

### 🎌 Anime Recommender

> [`/anime-recommender`](https://github.com/MRSandyra/anime-recommender)

A recommendation system that suggests anime titles based on user preferences and title similarity. Maintained in its own repository and linked here as a submodule.

**Tech:** Python

---

## 🛠️ Applied Systems & Tools

### 📈 IndoStock Analyzer — Personal Stock Analysis Tool

> [`/IndoStock Analyzer`](./IndoStock%20Analyzer)

An automated analysis tool for **IDX (Indonesia Stock Exchange)** equities that fuses three signals into a single, data-driven recommendation ranging from **Strong Buy to Strong Sell**.

- **News sentiment** — scrapes IndoPremier headlines with Selenium, scores them with TextBlob
- **Fundamentals** — evaluates PER, PBV, ROE, DER, and earnings growth via Yahoo Finance
- **Technicals** — MA, RSI, MACD, support/resistance levels, and volume surges
- Combines all three into a **weighted composite score** with JSON reporting

**Tech:** Python · Selenium · TextBlob · yfinance · Pandas · NumPy

---

### 🛡️ Cybersecurity — Anomaly Detection System

> [`/Cybersecurity Anomali Detection`](./Cybersecurity%20Anomali%20Detection)

A hybrid **PHP + Python** web application that parses Apache/Nginx access logs to detect **11 types of web attacks** and anomalous behaviour, then surfaces the findings on a dashboard.

- Core detection engine (`check.py`) with URL decoding for obfuscated payloads
- Per-IP request counting for **DoS** detection and **brute-force** login tracking
- Results exported to CSV; PHP web dashboard with authentication and password reset

**Tech:** Python · PHP · MySQL

---

## 📊 Data Analysis & Visualization

### 🌍 Global Income Inequality — Lorenz Curves & Gini Coefficients

> [`/Global Income Inequality`](./Global%20Income%20Inequality%20-%20Visualizing%20Lorenz%20Curves%20%26%20Gini%20Coefficients)

A cross-country study of income inequality over roughly two decades using World Bank data, visualising the distribution of wealth through **Lorenz curves** and **Gini coefficients.**

- Interactive heatmaps of inequality across countries and time
- Country clustering based on Gini index and GDP per capita

**Tech:** Plotly · scikit-learn (clustering) · NumPy

---

### ✈️ Capstone — Flight Ticket Sales Analysis

> [`/Capstone Project - Analisis Data Penerbangan`](./Capstone%20Project%20-%20Analisis%20Data%20Penerbangan)

An end-to-end analysis of flight ticket sales data (price, airline, route, stopovers, arrival date), from raw data to insight.

- Data cleaning and feature engineering (e.g. total tickets sold per airline)
- Exploratory analysis revealing the **top 10 airlines** by sales, distribution trends, and correlation heatmaps

**Tech:** Pandas · NumPy · Matplotlib · Seaborn · Google Colab

---

## 🧰 Tech Stack

**Languages & Core**
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![PHP](https://img.shields.io/badge/PHP-777BB4?style=flat-square&logo=php&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=mysql&logoColor=white)

**Data & ML**
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006400?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)

**Visualization**
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)

**Tools & Deployment**
![Selenium](https://img.shields.io/badge/Selenium-43B02A?style=flat-square&logo=selenium&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![Laravel](https://img.shields.io/badge/Laravel-FF2D20?style=flat-square&logo=laravel&logoColor=white)
![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)

---

## 📫 Connect

- 🌐 Portfolio: **[mrsandyra.netlify.app](https://mrsandyra.netlify.app/)**
- 💻 GitHub: **[@MRSandyra](https://github.com/MRSandyra)**

<div align="center">

⭐ *If you find something here useful or interesting, a star is always appreciated!*

</div>
