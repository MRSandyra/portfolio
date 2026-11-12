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
| [NusaLens](#-nusalens-ai-assistant-for-indonesian-public-data) | Answers questions about Indonesia using trusted public sources | Python, PostgreSQL, Ollama, Streamlit |
| [League of Legends Analytics](#-league-of-legends-challenger-match-analytics--win-prediction) | Predicts match results and explores player behaviour | Python, XGBoost, Plotly |
| [Lung Disease Detection](#-lung-disease-detection-deep-learning) | Analyses chest X-rays and symptoms with AI | TensorFlow, Laravel, Flask |
| [Sentiment Analysis (Tokopedia)](#-sentiment-analysis-tokopedia-reviews) | Finds positive and negative opinions in customer reviews | Python, scikit-learn |
| [Anime Recommender](#-anime-recommender) | Suggests anime based on user preferences | Python |
| [IndoStock Analyzer](#-indostock-analyzer-personal-stock-analysis-tool) | Brings together stock news, financial data, and market trends | Python, Selenium, yfinance |
| [Cybersecurity Anomaly Detection](#-cybersecurity-anomaly-detection-system) | Flags suspicious activity in website access logs | Python, PHP, MySQL |
| [Global Income Inequality](#-global-income-inequality-lorenz-curves--gini-coefficients) | Compares income inequality across countries | Python, Plotly |
| [Capstone: Flight Sales Analysis](#-capstone-flight-ticket-sales-analysis) | Examines flight sales and airline trends | Python, Pandas |
| [FulfillIQ Cloud](#-fulfilliq-cloud-delivery-performance-dashboard) | Checks whether online orders arrive on time, with the numbers double-verified between a local and cloud calculation | Python, DuckDB, AWS, Terraform |
| [PanganWatch](#-panganwatch-governed-food-price-intelligence) | Tracks food prices through a governed, validated Databricks Lakehouse | Databricks, PySpark, Delta Lake |
| [Fraud & Transaction Risk Monitoring Pipeline](#-fraud--transaction-risk-monitoring-pipeline) | Monitors payment activity for potential fraud risks | Kafka, Spark, Superset |
| [Jakarta Air Quality Pipeline](#-jakarta-air-quality-pipeline) | Combines air quality and weather data in a dashboard | Airflow, PostgreSQL, Metabase |
| [Wikipedia Edit Stream Analytics](#-wikipedia-edit-stream-analytics) | Tracks Wikipedia edits and highlights unusual activity | Python, DuckDB, Streamlit |

---

## 🤖 Machine Learning & AI

### 🔎 NusaLens: AI Assistant for Indonesian Public Data

> [`/NusaLens`](https://github.com/MRSandyra/NusaLens)

**Helping people explore facts about Indonesia through simple questions.** NusaLens is an AI-powered application that answers questions in Indonesian using public information from the World Bank and Wikipedia. It shows the sources behind its answers and can explain when there is not enough information to respond confidently.

- Makes statistical data and background information easier to access without manually searching through tables and articles.
- In project testing, **35 out of 35 test questions** were successfully converted into working database queries, and the search system improved its ability to find relevant information by **61.5%** compared with the initial version.
- Runs locally without paid AI APIs. **Still being improved:** some written answers do not fully reflect all available source information (68.6% fact coverage in the main evaluation).

**Built with:** Python · PostgreSQL · Ollama · FastAPI · Streamlit · Docker

---

### 🎮 League of Legends: Challenger Match Analytics & Win Prediction

> [`/League of Legends Challenger Rank Analytics`](./League%20of%20Legends%20Challenger%20Rank%20Analytics)

**Understanding what contributes to winning competitive matches.** I analysed **300 high-level League of Legends matches** and around **311,000 in-game events** to study winning patterns, player behaviour, and team combinations.

- Built models to predict match outcomes and an interactive dashboard to explore the results.
- Checked that predictions were not relying on information players would not have at the time, helping make the evaluation more realistic.

**Built with:** Python · scikit-learn · XGBoost · SHAP · Plotly · NetworkX

---

### 🫁 Lung Disease Detection: Deep Learning

> [`/lung-diagnosis`](https://github.com/MRSandyra/Lung-Diagnosis-Web)

**Exploring how AI can support the analysis of lung conditions.** This web application examines **chest X-ray images alongside patient-reported symptoms** and produces predictions across **11 lung-related categories**.

- Combines medical images and symptom information rather than relying on one input alone.
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

**Comparing income inequality around the world.** This project uses World Bank data from roughly two decades to show how income distribution differs between countries and how it changes over time.

- Creates interactive charts and maps that make country comparisons easier to understand.
- Groups countries with similar inequality and economic patterns for further exploration.

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

**Making sure a delivery performance dashboard's numbers can actually be trusted.** This project tracks whether online orders are delivered on time, using real historical e-commerce data. To prove the results are accurate, every number is calculated twice in two completely separate ways, once on a local computer and once independently in the cloud (AWS), and both have to match.

- Checked nearly 100,000 orders and found a **6.7% late-delivery rate**, with the local and cloud results matching exactly.
- Built in checks so that bad, duplicate, or incomplete data cannot quietly skew the final numbers, with zero failed or rejected records across the full run.
- Includes a simple prediction model that estimates delivery risk using only the information known at checkout, then tests it the way it would actually be used in practice.
- The dashboard only ever shows summarized numbers, never personal customer details or private login information, and is upfront about what the data can and cannot prove.

**Built with:** Python · DuckDB · AWS (S3 · Lambda · Athena) · Terraform · React · TypeScript · Vite

---

### 🌾 PanganWatch: Governed Food Price Intelligence

> [`/panganwatch`](https://github.com/MRSandyra/panganwatch)

**Helping people understand food price changes across Indonesia.** PanganWatch brings together public food price, rainfall, and inflation data to explore regional price differences and possible relationships between weather and prices. Its current cloud path is a batch-focused Databricks Lakehouse built from immutable source snapshots.

- Migrated the governed transformation flow to a Databricks Free Edition Lakehouse with Bronze, Silver, and Gold Delta layers, reference mapping, data-quality gates, governance tables, and source-replay reconciliation.
- Validated the migration against a recovered legacy baseline with **zero missing business keys and zero value mismatches**; a nine-task batch job powers a four-page Databricks dashboard.
- Clearly reports missing information instead of hiding it: food price coverage reaches **34 of Indonesia's 38 provinces**, with some periods having much less data.
- Treats patterns between weather and prices as **relationships, not proof of cause and effect**.

**Built with:** Python · Databricks · PySpark · Delta Lake · Unity Catalog · Airflow · dbt · PostgreSQL · MongoDB · Kafka · Docker

---

### 🚨 Fraud & Transaction Risk Monitoring Pipeline

> [`/fraud-transaction-risk-pipeline`](https://github.com/MRSandyra/fraud-transaction-risk-pipeline)

**Making payment risks easier to monitor.** This project simulates a payment monitoring platform using a public dataset of **6.36 million transactions**, including **8,213 labelled fraudulent transactions**.

- Processes transaction activity and prepares dashboards to help identify unusual payment patterns.
- Combines continuously arriving data with scheduled updates and includes checks to improve data consistency.

**Built with:** Kafka · Apache Spark · MinIO · ClickHouse · dbt · Airflow · Superset

---

### 🌫️ Jakarta Air Quality Pipeline

> [`/jakarta-air-quality-pipeline`](./jakarta-air-quality-pipeline)

**Making Jakarta's air quality easier to explore.** This project collects air quality readings and weather information, refreshes them regularly, and brings them together in an interactive dashboard.

- Automates hourly updates and includes checks to reduce errors in the displayed data.
- Found that **only 2 of 27 nearby monitoring stations** were reporting usable readings during the investigation, highlighting why data quality matters before drawing conclusions.

**Built with:** Python · Airflow · dbt · PostgreSQL · Metabase · Docker

---

### 🌐 Wikipedia Edit Stream Analytics

> [`/wikipedia-edit-stream-analytics`](https://github.com/MRSandyra/wikipedia-edit-stream-analytics)

**Tracking how Wikipedia changes in real time.** This application follows public Wikipedia edits and highlights activity that may deserve closer review, such as unusually large deletions or repeated edits.

- Gives edits a **0–100 risk score** using six predefined warning signals; the score is an indicator, not a confirmed finding of vandalism.
- Updates summary data regularly and displays patterns in an interactive dashboard.

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
