"""
Runs sql/extract_demand.sql against the real BigQuery public dataset and
writes the raw result to data/raw/. Requires GCP application-default
credentials (`gcloud auth application-default login`) with BigQuery read
access — not exercised by the test suite, which runs entirely against the
synthetic fixture in tests/conftest.py.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from config.settings import settings


def run_extraction(project_id: str | None = None, sql_path: str | None = None) -> pd.DataFrame:
    """Execute the extraction query and return the raw result as a
    DataFrame with columns: week_start, store_number, liquor_type,
    bottles_sold. Billing project defaults to the caller's own GCP project
    (settings.gcp_project_id is the *data* project, `bigquery-public-data`,
    which cannot be billed against)."""
    from google.cloud import bigquery

    query = Path(sql_path or settings.sql_path).read_text()
    client = bigquery.Client(project=project_id)
    return client.query(query).to_dataframe()


def extract_and_save(project_id: str | None = None, out_path: str | None = None) -> Path:
    """Run the extraction and write the result to
    `<out_path or settings.raw_data_dir>/demand_raw.parquet`, returning the
    path written."""
    df = run_extraction(project_id=project_id)
    out = Path(out_path or settings.raw_data_dir) / "demand_raw.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="GCP project ID to bill the query against")
    args = parser.parse_args()
    path = extract_and_save(project_id=args.project)
    print(f"Wrote raw extract to {path}")
