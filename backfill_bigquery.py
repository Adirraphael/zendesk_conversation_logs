"""
One-time backfill: uploads the ENTIRE existing master CSV to BigQuery.

Run this once to catch BigQuery up with all the historical rows already
in zendesk_conversations_scored_master.csv (rows that were skipped from
the regular pipeline's BigQuery upload because they were already scored
before BigQuery was added).

Uses WRITE_TRUNCATE — this REPLACES whatever is currently in the BigQuery
table with the full CSV contents, so it's safe to run even after the
pipeline already uploaded a handful of rows today (no duplicates result).

Uploads as JSON rows (load_table_from_json) rather than a pandas dataframe —
this avoids a pyarrow/pandas version incompatibility that causes
"TypeError: expected bytes, NoneType found" with load_table_from_dataframe
on some environments.
"""

import csv
import json
import os
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv()

MASTER_CSV = "zendesk_conversations_scored_master.csv"
GCP_SERVICE_ACCOUNT_JSON = os.environ["GCP_SERVICE_ACCOUNT_JSON"]
BQ_TABLE_REF             = os.environ["BQ_TABLE_REF"]

INT_COLS   = ["ticket_id", "accuracy_helpfulness", "retrieval_quality",
              "conversational_ux", "safety_compliance", "business_outcomes"]
FLOAT_COLS = ["success_rate"]

SCHEMA = [
    bigquery.SchemaField("ticket_id", "INTEGER"),
    bigquery.SchemaField("status", "STRING"),
    bigquery.SchemaField("question_date", "STRING"),
    bigquery.SchemaField("question_time", "STRING"),
    bigquery.SchemaField("answer_time", "STRING"),
    bigquery.SchemaField("question", "STRING"),
    bigquery.SchemaField("category", "STRING"),
    bigquery.SchemaField("bot_answer", "STRING"),
    bigquery.SchemaField("article_titles", "STRING"),
    bigquery.SchemaField("accuracy_helpfulness", "INTEGER"),
    bigquery.SchemaField("retrieval_quality", "INTEGER"),
    bigquery.SchemaField("conversational_ux", "INTEGER"),
    bigquery.SchemaField("safety_compliance", "INTEGER"),
    bigquery.SchemaField("business_outcomes", "INTEGER"),
    bigquery.SchemaField("instructions_involved", "STRING"),
    bigquery.SchemaField("notes", "STRING"),
    bigquery.SchemaField("success_rate", "FLOAT"),
    bigquery.SchemaField("outlier", "STRING"),
    bigquery.SchemaField("responded", "STRING"),
    bigquery.SchemaField("graded_at", "STRING"),
]


def clean_row(row):
    """Convert a csv.DictReader row (all strings) into properly typed JSON values."""
    cleaned = {}
    for key, value in row.items():
        if key is None:
            continue  # guards against stray/unnamed CSV columns
        value = (value or "").strip()
        if key in INT_COLS:
            cleaned[key] = int(value) if value.lstrip("-").isdigit() else None
        elif key in FLOAT_COLS:
            try:
                cleaned[key] = float(value) if value else None
            except ValueError:
                cleaned[key] = None
        else:
            cleaned[key] = value if value != "" else None
    return cleaned


def main():
    print(f"Reading {MASTER_CSV}...")
    with open(MASTER_CSV, newline="", encoding="utf-8-sig") as f:
        raw_rows = list(csv.DictReader(f))
    print(f"  → {len(raw_rows)} total rows found\n")

    if not raw_rows:
        print("Nothing to upload.")
        return

    rows = [clean_row(r) for r in raw_rows]

    # Only keep schema fields that actually exist in this CSV
    csv_columns = set(rows[0].keys())
    schema = [f for f in SCHEMA if f.name in csv_columns]

    gcp_info = json.loads(GCP_SERVICE_ACCOUNT_JSON)
    credentials = service_account.Credentials.from_service_account_info(gcp_info)
    client = bigquery.Client(credentials=credentials, project=gcp_info["project_id"])

    print(f"Uploading {len(rows)} rows to {BQ_TABLE_REF} (replacing existing table contents)...")
    job = client.load_table_from_json(
        rows,
        BQ_TABLE_REF,
        job_config=bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE",
            schema=schema,
        )
    )
    job.result()
    print(f"✅ Uploaded {len(rows)} rows to BigQuery: {BQ_TABLE_REF}")


if __name__ == "__main__":
    main()