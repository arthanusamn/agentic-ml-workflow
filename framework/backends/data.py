"""Data backend — provider-agnostic data loading.

Supports CSV, BigQuery, and GCS. Returns a pandas DataFrame.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import pandas as pd


class DataBackend:
    """Loads data from the configured source into a pandas DataFrame."""

    def __init__(self, config: dict[str, Any]):
        self.type = (config.get("type") or "csv").lower()
        self.config = config

    def load(self) -> pd.DataFrame:
        if self.type == "csv":
            return self._load_csv()
        elif self.type == "bigquery":
            return self._load_bigquery()
        elif self.type == "gcs":
            return self._load_gcs()
        else:
            raise ValueError(f"Unsupported data source: {self.type}")

    def _load_csv(self) -> pd.DataFrame:
        path = self.config.get("path", "")
        if not path:
            raise ValueError("CSV data source requires a 'path' field")
        return pd.read_csv(path)

    def _load_bigquery(self) -> pd.DataFrame:
        location = self.config.get("location", "")
        if not location:
            raise ValueError(
                "BigQuery source requires 'location' (e.g., project.dataset.table)"
            )
        credentials_path = self.config.get("credentials", "")
        # Expose the SQL / table for the Data Agent to configure filtering
        try:
            from google.cloud import bigquery
            if credentials_path:
                client = bigquery.Client.from_service_account_json(credentials_path)
            else:
                client = bigquery.Client()
            sql = self.config.get("sql", f"SELECT * FROM `{location}`")
            df = client.query(sql).to_dataframe()
            return df
        except ImportError:
            raise ImportError("Install google-cloud-bigquery to use BigQuery backend")
        except Exception as e:
            raise RuntimeError(f"BigQuery load failed: {e}")

    def _load_gcs(self) -> pd.DataFrame:
        gcs_path = self.config.get("path", "")
        if not gcs_path:
            raise ValueError("GCS source requires 'path' (e.g., gs://bucket/file.csv)")
        try:
            from google.cloud import storage
            import io
            client = storage.Client()
            parts = gcs_path.replace("gs://", "").split("/", 1)
            bucket = client.bucket(parts[0])
            blob = bucket.blob(parts[1])
            content = blob.download_as_bytes()
            return pd.read_csv(io.BytesIO(content))
        except ImportError:
            raise ImportError("Install google-cloud-storage to use GCS backend")
