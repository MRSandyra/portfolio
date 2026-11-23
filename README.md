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
| [NusaLens](#-nusalens-ai-assistant-for-indonesian-public-data) | Prepares evidence-backed Indonesia briefs with an explicit production-readiness gate | Python, PostgreSQL, pgvector, Ollama |
| [League of Legends Analytics](#-league-of-legends-challenger-rank-analytics) | Freezes time-aware 10- and 20-minute models behind one-time holdout governance | Python, scikit-learn, PyArrow |
| [Lung Disease Detection](#-lung-disease-detection-deep-learning) | Combines chest X-rays and symptoms across 11 lung conditions | TensorFlow, Laravel, Flask |
| [Tokopedia Sentiment Analysis](#-tokopedia-sentiment-analysis-reproducible-ml-case-study) | Evaluates three-class review triage under extreme class imbalance | Python, scikit-learn, TF-IDF |
| [Anime Recommender](#-anime-recommender) | Suggests anime based on user preferences | Python |
| [IndoStock Analyzer](#-indostock-analyzer-exploratory-research-tool) | Produces governed exploratory indicators without investment recommendations | Python, Selenium, yfinance |
| [Cybersecurity Anomaly Detection](#-cybersecurity-anomaly-detection) | Provides security-hardened triage for suspicious web-server requests | PHP, Python, MySQL |
| [Global Income Inequality](#-global-income-inequality-lorenz-curves--gini-coefficients) | Reproduces grouped-data Gini analysis and illustrative scenarios | Python, Pandas, Matplotlib |
| [Capstone: Flight Offer Analysis](#-flight-offer-analysis-capstone) | Audits flight listings for pricing and assortment analysis | Python, Pandas, Matplotlib |
| [FulfillIQ Cloud](#-fulfilliq-cloud-delivery-performance-dashboard) | Reconciles one delivery-SLA definition locally and in AWS with gated evidence | Python, DuckDB, AWS, Terraform |
| [PanganWatch](#-panganwatch-governed-food-price-intelligence) | Governs food-price data and proves a Databricks migration against its legacy baseline | Databricks, PySpark, Delta Lake |
| [Fraud & Transaction Risk Monitoring Pipeline](#-fraud--transaction-risk-monitoring-pipeline) | Demonstrates contract-validated streaming and batch fraud monitoring | Kafka, Spark, dbt, Superset |
| [Jakarta Air Quality Pipeline](#-jakarta-air-quality-pipeline) | Identifies which Jakarta sensors can support maintenance and public-health decisions | Airflow, dbt, PostgreSQL, Metabase |
| [Wikipedia Edit Stream Analytics](#-wikipedia-edit-stream-analytics) | Turns a global edit stream into an explainable moderation queue | Redpanda, DuckDB, Streamlit |

---

## 🤖 Machine Learning & AI

### 🔎 NusaLens: AI Assistant for Indonesian Public Data

> [`/NusaLens`](https://github.com/MRSandyra/NusaLens)

**Helping Indonesian public-policy analysts prepare evidence-backed country briefs.** NusaLens answers questions in Indonesian that combine World Bank indicators with indexed Wikipedia context. It exposes citations and generated SQL, and abstains when the available evidence is insufficient.

- Checked-in evaluation artifacts record an nDCG@10 improvement from **0.390 to 0.630 (+61.5%)**, text-to-SQL execution of **35/35**, routing accuracy of **119/120**, and abstain recall of **15/15**.
- Runs locally without paid AI APIs, using a rule-first router, PostgreSQL/pgvector retrieval, and local Ollama models. Its verification command reports the stored evidence and production gate without silently rerunning or changing the benchmark.
- **Deployment decision:** the production gate remains **NOT_MET** because answer fact coverage is **68.6%**, below the 80% target. Real policy use requires independent citation checks until the synthesis step is replaced or constrained and revalidated without reducing abstain recall.

**Built with:** Python · PostgreSQL · pgvector · Ollama · FastAPI · Streamlit · Docker

---

### 🎮 League of Legends Challenger Rank Analytics

> [`/League of Legends Challenger Rank Analytics`](./League%20of%20Legends%20Challenger%20Rank%20Analytics)

**Building an auditable prediction system instead of optimizing a better-looking score.** This project predicts whether Team 100 will win using only information available at a declared 10- or 20-minute decision point. Development uses an audited **2,000-match Americas cohort**, followed by a strictly later **500-match Europe holdout**.

- The one-time future holdout recorded **0.7844 ROC-AUC** at 10 minutes across 498 eligible matches and **0.8667** at 20 minutes across 404 matches, with bootstrap intervals reported for both.
- On the same 404 matches, performance rose from **0.7218 at 10 minutes to 0.8667 at 20 minutes**: a paired difference of **+0.1449** with a 95% interval of **[+0.0985, +0.1903]**.
- Feature-availability contracts, absolute-time validation, checksum-bound cohort roles, and a one-time release ledger prevent future information and post-holdout tuning. The final results are frozen and guarded by **105 passing tests** plus manifest and artifact audits.
- The final holdout changes time, patch, and region together, so the result supports that specific transfer setting rather than universal generalization.

**Built with:** Python · Pandas · scikit-learn · PyArrow · joblib · uv · pytest · Jupyter · GitHub Actions

---

### 🫁 Lung Disease Detection: Deep Learning

> [`/lung-diagnosis`](https://github.com/MRSandyra/Lung-Diagnosis-Web)

**Exploring how AI can support the analysis of lung conditions.** This web application combines **224x224 chest X-ray images with 13 symptom features** and produces probabilities across **11 lung-related categories**.

- Uses a DenseNet-169 model with a documented **0.7586 AUC** model artifact.
- Includes an interface for uploading images, viewing predictions, and reviewing previous results.

*This is a research/software project, not a substitute for professional medical diagnosis.*

**Built with:** TensorFlow · DenseNet-169 · Laravel · Flask · MySQL/PostgreSQL

---

### 💬 Tokopedia Sentiment Analysis: Reproducible ML Case Study

> [`/Sentiment Analysis Tokopedia`](./Sentiment%20Analysis%20Tokopedia)

**Evaluating Indonesian review triage under realistic imbalance and time shift.** This project uses **65,543 Tokopedia reviews** to compare three-class positive, neutral, and negative sentiment models with temporal, product-disjoint, slice, drift, calibration, and uncertainty checks.

- The selected two-stage balanced Logistic Regression with word and character TF-IDF reached **81.68% accuracy** but only **44.91% macro-F1**, showing why headline accuracy is insufficient on a dataset that is 97.6% positive.
- Negative recall reached **71.43% at 20.24% precision**, while neutral recall reached **69.84% at 7.24% precision**. The recommended precision-constrained triage policy routes only **440 of 8,374** test reviews to candidate queues instead of forcing every prediction into an operational label.
- The older **82.79% binary accuracy** came from global downsampling and vocabulary fitting before splitting; it remains historical exploration and is not comparable with the leakage-aware package benchmark.
- This remains an offline research baseline: labels are rating proxies, rare-class precision is low, and no independently human-labelled gold set exists yet.

**Built with:** Python · Pandas · scikit-learn · TF-IDF · Logistic Regression · uv · pytest

---

### 🎌 Anime Recommender

> [`/anime-recommender`](https://github.com/MRSandyra/anime-recommender)

**Helping viewers discover their next anime.** A recommendation tool that suggests titles based on user preferences and similarities between shows.

**Built with:** Python

---

## 🛠️ Applied Systems & Tools

### 📈 IndoStock Analyzer: Exploratory Research Tool

> [`/IndoStock Analyzer`](./IndoStock%20Analyzer)

**Making exploratory market research traceable without presenting it as investment advice.** IndoStock Analyzer collects attributed news and market data for Indonesian equities and produces bounded **1-5 exploratory indicators**. It explicitly does not issue Buy, Sell, Strong Buy, or Strong Sell recommendations.

- Reports include source URLs, retrieval times, failures, component coverage, data-quality status, and a disclaimer. Stale or invalid components are excluded instead of silently scored as current data.
- The bundled ticker universe is a disclosed **43-symbol curated subset**, not complete IDX coverage. Indonesian-language sentiment remains `UNASSESSED` by default because the original TextBlob analyzer uses an English lexicon.
- Sentiment activation fails closed behind dataset/model provenance, evaluation thresholds, exact artifact hashes, signed multi-reviewer approval, and revocation controls.
- A point-in-time backtest and evidence-audit framework is included, but no licensed historical observation dataset or completed holdout report is bundled, so the project makes **no predictive-performance claim**.

**Built with:** Python · Selenium · yfinance · Pandas · TextBlob · cryptography · pytest

---

### 🛡️ Cybersecurity Anomaly Detection

> [`/Cybersecurity Anomali Detection`](./Cybersecurity%20Anomali%20Detection)

**Providing security-hardened triage for suspicious website activity.** This PHP and Python application analyses Apache/Nginx Combined Log Format files and flags **11 attack or anomaly patterns**, including injection attempts, path traversal, unauthorized-path probing, repeated logins, and high-volume requests.

- Uploads are limited to 5 MB and checked by extension, MIME type, NUL bytes, and log format before each analysis runs in a random private workspace outside the web root.
- Password-reset tokens are random, stored only as SHA-256 digests, expire after one hour, and are consumed atomically; generic responses reduce account enumeration.
- The analyzer bypasses shell command construction, isolates concurrent jobs, restricts CSV downloads to the originating session, and includes PHP security tests plus a Python concurrency integration test.
- Only `public/` is intended as the HTTP document root; analyzer code, database configuration, migrations, tests, and generated files remain outside direct web access.
- Detection remains intentionally regex-based and may miss novel or off-path attacks. It is a **triage tool, not an IDS decision engine**.

**Built with:** PHP · Python · MySQL/MariaDB

---

## 📊 Data Analysis & Visualization

### 🌍 Global Income Inequality: Lorenz Curves & Gini Coefficients

> [`/Global Income Inequality`](./Global%20Income%20Inequality%20-%20Visualizing%20Lorenz%20Curves%20%26%20Gini%20Coefficients)

**Reproducing grouped-data inequality analysis with explicit comparability limits.** This project builds Lorenz curves and grouped Gini coefficients from the World Bank Poverty and Inequality Platform. Tested Python modules generate a deterministic report from a committed **1,600-row sample**, while a separate command refreshes data from a pinned, checksum-verified official source version.

- National distributions must contain exactly 100 percentile rows; incomplete analytical bins are rejected rather than imputed or silently dropped.
- The report distinguishes income from consumption as welfare and avoids implying that mixed welfare definitions or non-harmonized survey years are directly comparable.
- Indonesia 2019 examples add 0.5 PPP dollars per person per day to the bottom 50% or redistribute a static 10% top-decile reduction to the bottom 40%. These are arithmetic illustrations, **not policy forecasts or evidence that one policy is superior**.

**Built with:** Python · Pandas · NumPy · Matplotlib · uv · pytest

---

### ✈️ Flight Offer Analysis Capstone

> [`/Capstone Project - Analisis Data Penerbangan`](./Capstone%20Project%20-%20Analisis%20Data%20Penerbangan)

**Finding defensible pricing and assortment insights in a snapshot of flight offers.** The source rows are **flight offer listings, not ticket sales**. A reproducible pipeline cleans and validates fares, durations, stop patterns, dates, airlines, and routes before generating business-facing outputs.

- Outputs include cleaned and rejected-row data, a quality report, KPI/airline/route/stop tables, three charts, and a dataset-derived executive summary.
- The source and license remain unverified, so the raw dataset is not committed. Arrival-after-departure validation is explicitly not testable without departure dates, years, and time-zone context.
- The refactor corrects legacy notebook risks, including calling listing frequency “ticket sales,” summing repeated group counts, cross-row backfilling, and accepting unvalidated fares and dates.

**Built with:** Python · Pandas · NumPy · Matplotlib · pytest · GitHub Actions

---

## 🔧 Data Engineering

### 🚀 FulfillIQ Cloud: Delivery Performance Dashboard

> [`/fulfilliq`](https://github.com/MRSandyra/fulfilliq)

**Making a delivery-SLA dashboard earn its numbers.** FulfillIQ turns historical Olist data into one-row-per-order events, separates checkout facts from future outcomes, and independently reconciles the same metric contract in local DuckDB and AWS.

- The completed AWS replay reconciled **98,665 unique orders**, **96,384 eligible deliveries**, and **6,466 late deliveries** exactly to the frozen local reference: a **6.7086%** late-delivery rate.
- Deterministic, content-addressed batches make retries safe; all **987 placement** and **964 outcome** batches completed with zero full-run quarantine objects and passed duplicate-retry probes.
- A checked-in synthetic quickstart and local CI tiers verify the core contracts without AWS credentials, Olist downloads, or a cloud account; credentialed cloud reconciliation is explicitly gated and uses a read-only Athena query.
- A checkout-only model is reported honestly as a historical baseline (**ROC-AUC 0.6815**), while the browser receives aggregate data only.
- **Operational recommendation:** use 6.7% as the verified baseline, then prioritize customer regions that combine high order volume with a high late-delivery rate for investigation.

**Built with:** Python · DuckDB · AWS (S3 · Lambda · Athena) · Terraform · React · TypeScript · Vite

---

### 🌾 PanganWatch: Governed Food Price Intelligence

> [`/panganwatch`](https://github.com/MRSandyra/panganwatch)

**Proving whether a provincial food-price comparison deserves to be trusted.** PanganWatch preserves public food-price and weather responses as evidence, applies reference mapping and data-quality gates, and publishes a batch-focused Databricks Lakehouse from immutable source snapshots.

- The Databricks path uses Bronze, Silver, Gold, and governance Delta tables; its nine-task job completed successfully and feeds a four-page published dashboard. The local Docker/Airflow path and Databricks path retain separate, reviewable verification evidence.
- Source replay matched **164,750 price** and **269,496 weather** records with zero missing or unexpected rows; recovered legacy replay also had zero missing keys and zero value mismatches.
- The analysis found major coverage limits: prices cover **34 of 38 provinces**, only **8 of 14 reference commodities**, and January 2026 is nearly empty. Weather-price patterns are reported as correlations, not causation.
- **Policy recommendation:** onboard and validate feeds for the four missing Papua provinces before using provincial rankings for nationwide allocation or intervention decisions; until then, visibly flag and exclude incomplete provinces from nationally comparable rankings.

**Built with:** Python · Databricks · PySpark · Delta Lake · Unity Catalog · Airflow · dbt · PostgreSQL · MongoDB · Kafka · Docker

---

### 🚨 Fraud & Transaction Risk Monitoring Pipeline

> [`/fraud-transaction-risk-pipeline`](https://github.com/MRSandyra/fraud-transaction-risk-pipeline)

**Demonstrating the platform work behind fraud monitoring when bank-grade logs are unavailable.** The well-known public PaySim dataset is used as a stand-in for sensitive real transaction logs; the original work is the dual-path architecture, tested analytical layers, and incident investigation rather than the dataset itself. It is a demonstration platform, not a production fraud model or banking reference architecture.

- Kafka and Spark Structured Streaming validate versioned events, quarantine invalid records, and power the live path, while Airflow runs a separate batch path through a MinIO Bronze/Silver/Gold lake; ClickHouse, dbt, and Superset provide tested serving layers.
- A tracked 10-row demo fixture lets reviewers verify contracts, quarantine, dbt marts, and serving without downloading the 6.36-million-row PaySim dataset. CI checks producer contracts, Spark transformations, dbt mart contracts, and anomaly investigations without starting the Docker stack.
- Streaming telemetry records invalid and duplicate events, freshness, accepted volume, Kafka lag, and pipeline task failures for operational review.
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
- **Moderation recommendation:** send scores **50-69** to the automatic review queue and escalate scores **70+** to a human moderator. Do not auto-revert because the heuristic has not been precision/recall calibrated.

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
