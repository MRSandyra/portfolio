"""
Hourly ELT pipeline: pull air quality (OpenAQ) + weather (BMKG) for Jakarta,
land them in Postgres, then transform with dbt into staging/marts models,
gated by dbt tests.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

import sys
sys.path.append("/opt/airflow/extractors")

import openaq_extractor
import bmkg_extractor

DBT_PROJECT_DIR = "/opt/airflow/dbt_project"
DBT_PROFILES_DIR = "/opt/airflow/dbt_project"

default_args = {
    "owner": "iky",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="jakarta_air_quality_pipeline",
    description="Hourly ELT: OpenAQ + BMKG -> Postgres -> dbt marts",
    default_args=default_args,
    schedule_interval="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["data-engineering", "air-quality", "jakarta"],
) as dag:

    extract_openaq = PythonOperator(
        task_id="extract_openaq",
        python_callable=openaq_extractor.run,
    )

    extract_bmkg = PythonOperator(
        task_id="extract_bmkg",
        python_callable=bmkg_extractor.run,
    )

    # packages.yml declares dbt-utils; without `dbt deps` first, dbt cannot
    # even parse the project (the accepted_range tests reference the package).
    dbt_deps = BashOperator(
        task_id="dbt_deps",
        bash_command=f"cd {DBT_PROJECT_DIR} && dbt deps --profiles-dir {DBT_PROFILES_DIR}",
    )

    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command=f"cd {DBT_PROJECT_DIR} && dbt run --profiles-dir {DBT_PROFILES_DIR}",
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command=f"cd {DBT_PROJECT_DIR} && dbt test --profiles-dir {DBT_PROFILES_DIR}",
    )

    [extract_openaq, extract_bmkg] >> dbt_deps >> dbt_run >> dbt_test
