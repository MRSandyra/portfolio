<div align="center">

# 🌟 Rizky's Portfolio

![License](https://img.shields.io/github/license/MRSandyra/portfolio?style=flat-square)
[![Last Commit](https://img.shields.io/github/last-commit/MRSandyra/portfolio?style=flat-square)](https://github.com/MRSandyra/portfolio/commits/main)
![Repo Size](https://img.shields.io/github/repo-size/MRSandyra/portfolio?style=flat-square)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-98.8%25-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f?style=flat-square&logo=python&logoColor=white)

**Turning data into insights and useful applications.**  
A collection of my projects in data analysis, artificial intelligence, and software development. Each project focuses on solving a practical problem, finding meaningful insights, or making information easier to use.

🌐 **[mrsandyra.netlify.app](https://mrsandyra.netlify.app/)**

</div>

---

## 👋 About

I enjoy turning complex data into **clear insights, useful tools, and real-world solutions**. My projects cover a range of interests, from understanding customer feedback and financial trends to building AI-powered applications and monitoring data quality.

I focus on more than just building a model or dashboard. I also aim to make the results **reliable, easy to understand, and useful for decision-making**. Some projects are interactive web applications, while others explore questions through data analysis.

Explore a project below for a quick overview of **what it does, why it matters, and the tools I used**. A few larger projects are maintained in separate GitHub repositories and included here as submodules.

---

## 🗂️ Projects at a Glance

| Project | What It Does | Main Tools |
|---|---|---|
| [NusaLens](#-nusalens-ai-assistant-for-indonesian-public-data) | Helps public-policy analysts prepare evidence-backed Indonesia briefs | Python, PostgreSQL, Ollama, FastAPI |
| [League of Legends Analytics](#-league-of-legends-challenger-match-analytics--win-prediction) | Audits when match predictions become reliable and explains the drivers | Python, XGBoost, SHAP |
| [Lung Disease Detection](#-lung-disease-detection-deep-learning) | Combines chest X-rays and symptoms across 11 lung conditions | TensorFlow, Laravel, Flask |
| [Sentiment Analysis (Tokopedia)](#-sentiment-analysis-tokopedia-reviews) | Finds positive and negative opinions in customer reviews | Python, scikit-learn |
| [Anime Recommender](#-anime-recommender) | Suggests anime based on user preferences | Python |
| [IndoStock Analyzer](#-indostock-analyzer-personal-stock-analysis-tool) | Brings together stock news, financial data, and market trends | Python, Selenium, yfinance |
| [Cybersecurity Anomaly Detection](#-cybersecurity-anomaly-detection-system) | Flags suspicious activity in website access logs | Python, PHP, MySQL |
| [Global Income Inequality](#-global-income-inequality-lorenz-curves--gini-coefficients) | Compares inequality and simulates redistribution scenarios | Python, Plotly, scikit-learn |
| [Capstone: Flight Sales Analysis](#-capstone-flight-ticket-sales-analysis) | Examines flight sales and airline trends | Python, Pandas |
| [FulfillIQ Cloud](#-fulfilliq-cloud-delivery-performance-dashboard) | Reconciles one delivery-SLA definition locally and in AWS | Python, DuckDB, AWS, Terraform |
| [PanganWatch](#-panganwatch-governed-food-price-intelligence) | Governs food-price data and proves a Databricks migration against its legacy baseline | Databricks, PySpark, Delta Lake |
| [Fraud & Transaction Risk Monitoring Pipeline](#-fraud--transaction-risk-monitoring-pipeline) | Demonstrates dual-path fraud monitoring and investigates misleading risk signals | Kafka, Spark, dbt, Superset |
| [Jakarta Air Quality Pipeline](#-jakarta-air-quality-pipeline) | Identifies which Jakarta sensors can support maintenance and public-health decisions | Airflow, dbt, PostgreSQL, Metabase |
| [Wikipedia Edit Stream Analytics](#-wikipedia-edit-stream-analytics) | Turns a global edit stream into an explainable moderation queue | Redpanda, DuckDB, Streamlit |

---

## 🤖 Machine Learning & AI

### 🔎 NusaLens: AI Assistant for Indonesian Public Data

> [`/NusaLens`](https://github.com/MRSandyra/NusaLens)

**Helping Indonesian public-policy analysts prepare evidence-backed country briefs.** NusaLens answers questions in Indonesian that combine World Bank indicators with indexed Wikipedia context. It exposes citations and generated SQL, and abstains when the available evidence is insufficient.

- The production retrieval pipeline improved nDCG@10 from **0.390 to 0.630 (+61.5%)**; text-to-SQL execution reached **35/35**, routing reached **119/120**, and abstain recall reached **15/15**.
- Runs locally without paid AI APIs, using a rule-first router, PostgreSQL/pgvector retrieval, and local Ollama models.
- **Deployment decision:** it is not ready for real policy queries because answer fact coverage is **68.6%**, below the 80% target. The synthesis step should be replaced or constrained and revalidated without reducing abstain recall.

**Built with:** Python · PostgreSQL · Ollama · FastAPI · Streamlit · Docker

---

### 🎮 League of Legends: Challenger Match Analytics & Win Prediction

> [`/League of Legends Challenger Rank Analytics`](./League%20of%20Legends%20Challenger%20Rank%20Analytics)

**Understanding when a competitive-match prediction becomes trustworthy.** I analysed **300 Challenger-tier League of Legends matches** and **311,266 in-game events** to study objectives, player archetypes, anomalies, and champion combinations.

- The full XGBoost model reached **0.901 ROC-AUC**, but a leakage audit showed that a 10-minute-only model reached just **0.68**; the stronger result is therefore a mid-game predictor, not an early-game claim.
- First Tower and First Herald were the strongest observed early objectives, while SHAP, permutation importance, and feature ablation were cross-checked to explain the model.

**Built with:** Python · scikit-learn · XGBoost · SHAP · Plotly · NetworkX

---

### 🫁 Lung Disease Detection: Deep Learning

> [`/lung-diagnosis`](https://github.com/MRSandyra/Lung-Diagnosis-Web)

**Exploring how AI can support the analysis of lung conditions.** This web application combines **224x224 chest X-ray images with 13 symptom features** and produces probabilities across **11 lung-related categories**.

- Uses a DenseNet-169 model with a documented **0.7586 AUC** model artifact.
- Includes an interface for uploading images, viewing predictions, and reviewing previous results.

*This is a research/software project, not a substitute for professional medical diagnosis.*

**Built with:** TensorFlow · DenseNet-169 · Laravel · Flask · MySQL/PostgreSQL

---

### 💬 Sentiment Analysis: Tokopedia Reviews

> [`/Sentiment Analysis Tokopedia`](./Sentiment%20Analysis%20Tokopedia)

**Understanding customer opinions from online reviews.** This project analyses Tokopedia food and beverage reviews and identifies whether the feedback is **positive or negative**.

- Cleans and organises review text so it can be analysed consistently.
- Achieved **approximately 83% accuracy** on a balanced evaluation dataset, showing how review analysis can help reveal customer sentiment.

**Built with:** Python · scikit-learn · Naive Bayes · CountVectorizer

---

### 🎌 Anime Recommender

> [`/anime-recommender`](https://github.com/MRSandyra/anime-recommender)

**Helping viewers discover their next anime.** A recommendation tool that suggests titles based on user preferences and similarities between shows.

**Built with:** Python

---

## 🛠️ Applied Systems & Tools

### 📈 IndoStock Analyzer: Personal Stock Analysis Tool

> [`/IndoStock Analyzer`](./IndoStock%20Analyzer)

**Making stock research easier to review in one place.** IndoStock Analyzer brings together news sentiment, company financial information, and price trends for stocks listed on the Indonesia Stock Exchange.

- Combines different signals into an overall assessment ranging from **Strong Buy to Strong Sell**.
- Helps users compare information that would otherwise require checking several sources separately.

**Built with:** Python · Selenium · TextBlob · yfinance · Pandas

---

### 🛡️ Cybersecurity: Anomaly Detection System

> [`/Cybersecurity Anomali Detection`](./Cybersecurity%20Anomali%20Detection)

**Making suspicious website activity easier to spot.** This web application checks website access records and flags patterns associated with **11 types of potential cyberattacks**, including repeated login attempts and unusually high request volumes.

- Presents findings in a dashboard and lets users export the results for further review.
- Designed to support security monitoring by highlighting records that may need investigation.

**Built with:** Python · PHP · MySQL

---

## 📊 Data Analysis & Visualization

### 🌍 Global Income Inequality: Lorenz Curves & Gini Coefficients

> [`/Global Income Inequality`](./Global%20Income%20Inequality%20-%20Visualizing%20Lorenz%20Curves%20%26%20Gini%20Coefficients)

**Comparing income inequality and possible redistribution scenarios.** This project uses **2,456 country-year distributions** from the World Bank Poverty and Inequality Platform to study Gini trends and Lorenz curves.

- Creates interactive trends and a global map, including an Indonesia 2019 baseline Gini of **0.3728**.
- Simulates targeted subsidy and progressive-tax scenarios as analytical illustrations, not policy forecasts.

**Built with:** Python · Plotly · scikit-learn · NumPy

---

### ✈️ Capstone: Flight Ticket Sales Analysis

> [`/Capstone Project - Analisis Data Penerbangan`](./Capstone%20Project%20-%20Analisis%20Data%20Penerbangan)

**Finding useful patterns in flight ticket sales.** This capstone explores airline, ticket price, route, and sales information to better understand customer travel trends.

- Organises and cleans the data before analysing it.
- Highlights the **top 10 airlines by ticket sales** and visualises pricing and travel patterns.

**Built with:** Python · Pandas · Matplotlib · Seaborn

---

## 🔧 Data Engineering

### 🚀 FulfillIQ Cloud: Delivery Performance Dashboard

> [`/fulfilliq`](https://github.com/MRSandyra/fulfilliq)

**Making a delivery-SLA dashboard earn its numbers.** FulfillIQ turns historical Olist data into one-row-per-order events, separates checkout facts from future outcomes, and independently reconciles the same metric contract in local DuckDB and AWS.

- The completed AWS replay reconciled **98,665 unique orders**, **96,384 eligible deliveries**, and **6,466 late deliveries** exactly to the frozen local reference: a **6.7086%** late-delivery rate.
- Deterministic, content-addressed batches make retries safe; all **987 placement** and **964 outcome** batches completed with zero full-run quarantine objects and passed duplicate-retry probes.
- A checkout-only model is reported honestly as a historical baseline (**ROC-AUC 0.6815**), while the browser receives aggregate data only.
- **Operational recommendation:** use 6.7% as the verified baseline, then prioritize customer regions that combine high order volume with a high late-delivery rate for investigation.

**Built with:** Python · DuckDB · AWS (S3 · Lambda · Athena) · Terraform · React · TypeScript · Vite

---

### 🌾 PanganWatch: Governed Food Price Intelligence

> [`/panganwatch`](https://github.com/MRSandyra/panganwatch)

**Proving whether a provincial food-price comparison deserves to be trusted.** PanganWatch preserves public food-price and weather responses as evidence, applies reference mapping and data-quality gates, and publishes a batch-focused Databricks Lakehouse from immutable source snapshots.

- The Databricks path uses Bronze, Silver, Gold, and governance Delta tables; its nine-task job completed successfully and feeds a four-page published dashboard.
- Source replay matched **164,750 price** and **269,496 weather** records with zero missing or unexpected rows; recovered legacy replay also had zero missing keys and zero value mismatches.
- The analysis found major coverage limits: prices cover **34 of 38 provinces**, only **8 of 14 reference commodities**, and January 2026 is nearly empty. Weather-price patterns are reported as correlations, not causation.
- **Policy recommendation:** onboard and validate feeds for the four missing Papua provinces before using provincial rankings for nationwide allocation or intervention decisions; until then, visibly flag and exclude incomplete provinces from nationally comparable rankings.

**Built with:** Python · Databricks · PySpark · Delta Lake · Unity Catalog · Airflow · dbt · PostgreSQL · MongoDB · Kafka · Docker

---

### 🚨 Fraud & Transaction Risk Monitoring Pipeline

> [`/fraud-transaction-risk-pipeline`](https://github.com/MRSandyra/fraud-transaction-risk-pipeline)

**Demonstrating the platform work behind fraud monitoring when bank-grade logs are unavailable.** The well-known public PaySim dataset is used as a stand-in for sensitive real transaction logs; the original work is the dual-path architecture, tested analytical layers, and incident investigation rather than the dataset itself.

- Kafka and Spark Structured Streaming power the live path, while Airflow runs a separate batch path through a MinIO Bronze/Silver/Gold lake; ClickHouse, dbt, and Superset provide tested serving layers.
- The analysis showed that a fraud-rate spike came from a collapse in legitimate transaction volume, not a surge in fraud count, and that a 100%-risk “merchant” was actually a grouped NULL bucket.
- **Fraud-ops recommendation:** enforce a non-null merchant identifier upstream and quarantine unmatched records before they enter any merchant risk leaderboard.

**Built with:** Kafka · Apache Spark · MinIO · ClickHouse · dbt · Airflow · Superset

---

### 🌫️ Jakarta Air Quality Pipeline

> [`/jakarta-air-quality-pipeline`](./jakarta-air-quality-pipeline)

**Helping Dinas Lingkungan Hidup decide where sensor maintenance can restore trustworthy coverage.** This hourly pipeline combines OpenAQ readings with BMKG forecasts, models them with dbt, and blocks publication behind data-quality tests.

- Of **27 stations** within 25 km of central Jakarta, **25 were stale** and only the `Jakarta` and `BMKG 1` nodes were reporting usable recent data; the backfill loaded 459 historical readings and all 11 dbt tests passed.
- The two live low-cost sensor vendors differed by roughly 4-5x, so their raw PM2.5 values should not be compared without shared reference-grade calibration.
- **Maintenance recommendation:** repair or replace the 25 stale stations first, keep the two live nodes online, and calibrate both against the same reference monitor before using comparisons for public-health decisions.

**Built with:** Python · Airflow · dbt · PostgreSQL · Metabase · Docker

---

### 🌐 Wikipedia Edit Stream Analytics

> [`/wikipedia-edit-stream-analytics`](https://github.com/MRSandyra/wikipedia-edit-stream-analytics)

**Turning Wikipedia's global edit firehose into an explainable moderation queue.** This real-time pipeline scores edits using transparent warning signals and gives platform-integrity teams a focused review feed instead of requiring them to inspect every change.

- Gives each non-bot edit a **0-100 suspicion score** using six deterministic rules, stores Bronze/Silver/Gold data, and refreshes dashboard summaries every 60 seconds.
- Runs locally with Docker Compose, Redpanda, DuckDB, and Streamlit; no cloud account, API key, or downloaded dataset is required.
- **Moderation recommendation:** queue scores **50-69** for review and immediately escalate scores **70+** to a human moderator. Do not auto-revert because the heuristic has not been precision/recall calibrated.

**Built with:** Python · Redpanda · DuckDB · Streamlit · Plotly · Docker

---

## 🧰 Tools & Technologies

**Languages & Core**
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![PHP](https://img.shields.io/badge/PHP-777BB4?style=flat-square&logo=php&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=mysql&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white)

**Data & ML**
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006400?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)

**AI Applications**
![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat-square)
![pgvector](https://img.shields.io/badge/pgvector-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)

**Visualization**
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)

**Tools & Deployment**
![Selenium](https://img.shields.io/badge/Selenium-43B02A?style=flat-square&logo=selenium&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![Laravel](https://img.shields.io/badge/Laravel-FF2D20?style=flat-square&logo=laravel&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat-square&logo=vite&logoColor=white)
![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)

**Data Platforms & Automation**
![Airflow](https://img.shields.io/badge/Apache%20Airflow-017CEE?style=flat-square&logo=apacheairflow&logoColor=white)
![dbt](https://img.shields.io/badge/dbt-FF694B?style=flat-square&logo=dbt&logoColor=white)
![Kafka](https://img.shields.io/badge/Apache%20Kafka-231F20?style=flat-square&logo=apachekafka&logoColor=white)
![Redpanda](https://img.shields.io/badge/Redpanda-E1361A?style=flat-square&logo=redpanda&logoColor=white)
![Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?style=flat-square&logo=apachespark&logoColor=white)
![ClickHouse](https://img.shields.io/badge/ClickHouse-FFCC01?style=flat-square&logo=clickhouse&logoColor=black)
![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?style=flat-square&logo=duckdb&logoColor=black)
![MinIO](https://img.shields.io/badge/MinIO-C72E49?style=flat-square&logo=minio&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=flat-square&logo=mongodb&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat-square&logo=amazonaws&logoColor=white)
![Terraform](https://img.shields.io/badge/Terraform-7B42BC?style=flat-square&logo=terraform&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat-square&logo=databricks&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Metabase](https://img.shields.io/badge/Metabase-509EE3?style=flat-square&logo=metabase&logoColor=white)
![Superset](https://img.shields.io/badge/Apache%20Superset-20A6C9?style=flat-square&logo=apachesuperset&logoColor=white)

---

## 📫 Connect

- 🌐 Portfolio: **[mrsandyra.netlify.app](https://mrsandyra.netlify.app/)**
- 💻 GitHub: **[@MRSandyra](https://github.com/MRSandyra)**

<div align="center">

⭐ *If you find something here useful or interesting, a star is always appreciated!*

</div>
