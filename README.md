<div align="center">

# Rizky Sandyra: Data & AI Portfolio

**I build reliable data products that turn complex information into decisions people can act on.**

Data & ML Engineering · Analytics Engineering · Applied AI

[**View portfolio website ↗**](https://mrsandyra.netlify.app/) · [GitHub](https://github.com/MRSandyra)

</div>

---

## Selected work

These projects show the range of my work: applied AI, cloud analytics, governed data platforms, and decision-focused dashboards. Click any image for the project source or case study.

<table>
  <tr>
    <td width="50%">
      <a href="https://github.com/MRSandyra/NusaLens">
        <img src="https://raw.githubusercontent.com/MRSandyra/NusaLens/main/reports/figures/11_ui_sql.png" alt="NusaLens answering an Indonesian public-data question and charting the result" />
      </a>
      <br />
      <sub><b>NusaLens</b>: an Indonesian-language assistant that shows its sources and generated SQL.</sub>
    </td>
    <td width="50%">
      <a href="https://github.com/MRSandyra/fulfilliq">
        <img src="https://raw.githubusercontent.com/MRSandyra/fulfilliq/main/docs/assets/dashboard-overview.png" alt="FulfillIQ delivery performance dashboard with order, eligibility, and late-delivery metrics" />
      </a>
      <br />
      <sub><b>FulfillIQ</b>: a delivery dashboard whose local and AWS results reconcile exactly.</sub>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <a href="./Global%20Income%20Inequality%20-%20Visualizing%20Lorenz%20Curves%20%26%20Gini%20Coefficients">
        <img src="./Global%20Income%20Inequality%20-%20Visualizing%20Lorenz%20Curves%20%26%20Gini%20Coefficients/report/assets/lorenz_idn_2019.png" alt="Lorenz curve for Indonesia in 2019" />
      </a>
      <br />
      <sub><b>Income inequality analysis</b>: reproducible Lorenz curves with clear comparison limits.</sub>
    </td>
    <td width="50%">
      <a href="https://github.com/MRSandyra/fraud-transaction-risk-pipeline">
        <img src="https://raw.githubusercontent.com/MRSandyra/fraud-transaction-risk-pipeline/main/docs/dashboard.png" alt="Fraud monitoring dashboard with fraud rate, alert, and event-volume views" />
      </a>
      <br />
      <sub><b>Fraud monitoring pipeline</b>: streaming and batch data paths feeding an investigation dashboard.</sub>
    </td>
  </tr>
</table>

### NusaLens: AI assistant for Indonesian public data

- **Problem:** Public-policy questions often require both structured indicators and supporting documents, while a language model alone can produce plausible but unsupported answers.
- **Built:** A local assistant that routes Indonesian questions to SQL, document search, or both; every answer can expose its citations, SQL, and confidence.
- **Evidence:** Search quality improved from **0.390 to 0.630 nDCG@10**, SQL execution passed **35/35** cases, and the router passed **119/120**. The answer-writing layer remains below its release target (**68.6% vs 80% fact coverage**), so the project is presented as evaluated research rather than a production-ready policy tool.

[Source & case study](https://github.com/MRSandyra/NusaLens) · [UI preview](https://github.com/MRSandyra/NusaLens/blob/main/reports/figures/11_ui_sql.png)

**Stack:** Python · PostgreSQL · pgvector · Ollama · FastAPI · Streamlit · Docker

---

### FulfillIQ: cloud delivery performance dashboard

- **Problem:** Delivery metrics become hard to trust when local analysis, cloud processing, and dashboard definitions drift apart.
- **Built:** A one-row-per-order data model, replay-safe AWS pipeline, independent DuckDB reference, and browser dashboard that receives aggregate data only.
- **Evidence:** Local and AWS results matched across **98,665 orders** and **96,384 eligible deliveries**, producing the same **6.71% late-delivery rate**. All 987 placement batches and 964 outcome batches completed without full-run quarantine records.

[Source & case study](https://github.com/MRSandyra/fulfilliq) · [Dashboard preview](https://github.com/MRSandyra/fulfilliq/blob/main/docs/assets/dashboard-overview.png) · [Run locally](https://github.com/MRSandyra/fulfilliq#local-quick-start-with-synthetic-data)

**Stack:** Python · DuckDB · AWS · Terraform · React · TypeScript · Vite

---

### PanganWatch: governed food-price intelligence

- **Problem:** Provincial price rankings can mislead decision-makers when source coverage is incomplete or a migration silently changes the numbers.
- **Built:** A Databricks Lakehouse with Bronze, Silver, Gold, and governance layers, plus a separate local Airflow/dbt implementation for comparison.
- **Evidence:** The migration replay matched **164,750 price** and **269,496 weather** records without missing or unexpected rows. The analysis also exposed the real limitation: price coverage reaches **34 of 38 provinces** and **8 of 14 reference commodities**, so incomplete regions should not be ranked as if coverage were national.

[Source & case study](https://github.com/MRSandyra/PanganWatch) · [Key findings](https://github.com/MRSandyra/PanganWatch#2-key-findings) · [Run locally](https://github.com/MRSandyra/PanganWatch#6-quick-start)

**Stack:** Databricks · PySpark · Delta Lake · Airflow · dbt · PostgreSQL · Docker

---

### Anime Recommender: hybrid discovery prototype

- **Problem:** Genre filters alone do not capture the story, themes, and tone that make one title feel similar to another.
- **Built:** A FastAPI and Streamlit prototype that embeds title metadata, retrieves 100 candidates from Qdrant, and reranks them with a **70% content / 30% item-similarity** blend.
- **Evidence:** The checked-in catalog contains **30,157 titles** and the API returns a ranked top 10 after excluding the viewer's selected favorites. No offline relevance benchmark is included yet, so this is described as a functional prototype rather than a validated recommendation model.

[Source](https://github.com/MRSandyra/anime-recommender) · [Recommendation API](https://github.com/MRSandyra/anime-recommender/blob/main/app/main.py) · [Streamlit interface](https://github.com/MRSandyra/anime-recommender/blob/main/frontend/app.py)

**Stack:** Python · Sentence Transformers · Qdrant · scikit-learn · FastAPI · Streamlit

---

## More projects

Each entry leads to the implementation and its reproducible evidence. **Preview** opens a screenshot; **Local app** means the interface is reproducible from source but is not publicly deployed.

### Machine learning & applied AI

| Project | What it demonstrates | Explore |
|---|---|---|
| [League of Legends Challenger Analytics](./League%20of%20Legends%20Challenger%20Rank%20Analytics) | Time-aware win prediction with a one-time future holdout: **0.784 ROC-AUC at 10 minutes** and **0.867 at 20 minutes**. | [Case study](./League%20of%20Legends%20Challenger%20Rank%20Analytics) |
| [Lung Disease Detection](https://github.com/MRSandyra/Lung-Diagnosis-Web) | Research web app combining chest X-rays and symptoms across 11 categories; the stored DenseNet-169 artifact reports **0.7586 AUC**. | [Source / local app](https://github.com/MRSandyra/Lung-Diagnosis-Web) |
| [Tokopedia Sentiment Analysis](./Sentiment%20Analysis%20Tokopedia) | Three-class review triage across **65,543 reviews**; shows why 81.68% accuracy can still mean only **44.91% macro-F1** under heavy imbalance. | [Case study](./Sentiment%20Analysis%20Tokopedia) |
| [IndoStock Analyzer](./IndoStock%20Analyzer) | Traceable exploratory indicators with source, freshness, failure, and coverage reporting, without presenting investment recommendations. | [Case study](./IndoStock%20Analyzer) |

### Analytics & decision support

| Project | What it demonstrates | Explore |
|---|---|---|
| [Global Income Inequality](./Global%20Income%20Inequality%20-%20Visualizing%20Lorenz%20Curves%20%26%20Gini%20Coefficients) | Reproducible Lorenz curves and grouped Gini analysis from a committed **1,600-row sample**, with survey-comparability limits made explicit. | [Case study](./Global%20Income%20Inequality%20-%20Visualizing%20Lorenz%20Curves%20%26%20Gini%20Coefficients) · [Preview](./Global%20Income%20Inequality%20-%20Visualizing%20Lorenz%20Curves%20%26%20Gini%20Coefficients/report/assets/lorenz_idn_2019.png) |
| [Flight Offer Analysis](./Capstone%20Project%20-%20Analisis%20Data%20Penerbangan) | A quality-checked pricing and assortment analysis that correctly treats the source as listings, not ticket sales. | [Case study](./Capstone%20Project%20-%20Analisis%20Data%20Penerbangan) |
| [Cybersecurity Anomaly Detection](./Cybersecurity%20Anomali%20Detection) | Security-hardened log triage for 11 suspicious request patterns, with isolated uploads and safe password-reset handling. | [Source / local app](./Cybersecurity%20Anomali%20Detection) |

### Data engineering

| Project | What it demonstrates | Explore |
|---|---|---|
| [Fraud & Transaction Risk Pipeline](https://github.com/MRSandyra/fraud-transaction-risk-pipeline) | Contract-checked Kafka/Spark and Airflow/dbt paths feeding an investigation dashboard; analysis separates a rate spike from its true volume cause. | [Source](https://github.com/MRSandyra/fraud-transaction-risk-pipeline) · [Preview](https://github.com/MRSandyra/fraud-transaction-risk-pipeline/blob/main/docs/dashboard.png) |
| [Jakarta Air Quality Pipeline](./jakarta-air-quality-pipeline) | Hourly OpenAQ/BMKG pipeline whose quality checks found **25 of 27 nearby stations stale**, turning data quality into a maintenance priority. | [Case study / local app](./jakarta-air-quality-pipeline) |
| [Wikipedia Edit Stream Analytics](https://github.com/MRSandyra/wikipedia-edit-stream-analytics) | Explainable moderation queue using six transparent signals, a 0–100 score, and 60-second dashboard refreshes. | [Source / local app](https://github.com/MRSandyra/wikipedia-edit-stream-analytics) |

---

## Technical toolbox

- **Languages:** Python, SQL, TypeScript, PHP
- **Data & ML:** Pandas, NumPy, scikit-learn, TensorFlow, PyArrow, Sentence Transformers
- **Pipelines & platforms:** Airflow, dbt, Kafka, Spark, Databricks, AWS, Terraform
- **Storage & serving:** PostgreSQL, pgvector, DuckDB, ClickHouse, Delta Lake, FastAPI
- **Interfaces:** React, Streamlit, Superset, Metabase, Plotly, Matplotlib
- **Quality:** pytest, data contracts, holdout design, reconciliation, lineage, CI

---

## Contact

[Portfolio website](https://mrsandyra.netlify.app/) · [GitHub @MRSandyra](https://github.com/MRSandyra)
