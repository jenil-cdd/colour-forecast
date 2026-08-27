"""Pull the duvet panel out of BigQuery into local parquet.

Each query is run once and cached to ``data/raw``. The sales/traffic fact table
is ~7 GB per full scan, so re-running the pipeline should never re-query it
unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import logging

import db_dtypes  # noqa: F401  (registers BigQuery DATE/TIME pandas extension dtypes)
import pandas as pd
from google.cloud import bigquery

from src.config import RAW, Config, ensure_dirs, load_config

log = logging.getLogger(__name__)

SQL_DIR = None  # resolved lazily against config.ROOT


def _sql(name: str, cfg: Config) -> str:
    from src.config import ROOT

    text = (ROOT / "sql" / name).read_text()
    return text.format(
        project=cfg.gcp_project,
        dataset=cfg.dataset,
        mapper=cfg.mapper,
    )


def _client(cfg: Config) -> bigquery.Client:
    return bigquery.Client(project=cfg.gcp_project)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Cast BigQuery DATE extension columns to plain datetime64.

    ``db_dtypes`` writes a ``dbdate`` extension type into the parquet metadata,
    which then fails to load anywhere that has not imported ``db_dtypes`` first.
    Normalising here keeps ``data/raw`` readable by plain pandas.
    """
    for col in df.columns:
        if str(df[col].dtype) in {"dbdate", "dbtime"}:
            df[col] = pd.to_datetime(df[col].astype("datetime64[ns]"))
    return df


def _run(client: bigquery.Client, sql: str, params: list) -> pd.DataFrame:
    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
    df = _normalise(job.to_dataframe())
    log.info("scanned %.2f GB -> %d rows", (job.total_bytes_processed or 0) / 1e9, len(df))
    return df


def extract(cfg: Config, force: bool = False) -> dict[str, pd.DataFrame]:
    ensure_dirs()
    split = cfg.split
    # Extract through today's data, not just the test end: the extra tail is
    # useful for post-hoc validation and costs nothing extra on a full scan.
    hist_start = split.history_start
    hist_end = pd.Timestamp.today().date()

    out: dict[str, pd.DataFrame] = {}
    client = None

    # ---- 1. SKU dimension --------------------------------------------------
    p = RAW / "sku_dim.parquet"
    if p.exists() and not force:
        out["sku_dim"] = pd.read_parquet(p)
    else:
        client = client or _client(cfg)
        out["sku_dim"] = _run(
            client,
            _sql("01_sku_dim.sql", cfg),
            [bigquery.ArrayQueryParameter("programs", "STRING", cfg.all_programs)],
        )
        out["sku_dim"].to_parquet(p, index=False)
    asins = sorted(out["sku_dim"]["child_asin"].dropna().unique().tolist())
    log.info("%d ASINs across programs %s", len(asins), cfg.all_programs)

    date_params = [
        bigquery.ScalarQueryParameter("history_start", "DATE", hist_start),
        bigquery.ScalarQueryParameter("history_end", "DATE", hist_end),
    ]
    asin_param = bigquery.ArrayQueryParameter("asins", "STRING", asins)

    # ---- 2..5 fact tables ---------------------------------------------------
    specs = [
        ("daily_panel", "02_daily_panel.sql", date_params + [asin_param]),
        ("clean_days", "03_clean_days.sql", date_params),
        ("asin_deal_days", "04_asin_deal_days.sql", date_params + [asin_param]),
        ("returns", "05_returns.sql", date_params + [asin_param]),
    ]
    for key, sql_file, params in specs:
        p = RAW / f"{key}.parquet"
        if p.exists() and not force:
            out[key] = pd.read_parquet(p)
            log.info("%s: cached (%d rows)", key, len(out[key]))
            continue
        client = client or _client(cfg)
        df = _run(client, _sql(sql_file, cfg), params)
        df.to_parquet(p, index=False)
        out[key] = df
        log.info("%s: %d rows -> %s", key, len(df), p.name)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-query BigQuery, ignoring cache")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tables = extract(load_config(), force=args.force)
    for k, v in tables.items():
        print(f"{k:16s} rows={len(v):>8,}  cols={len(v.columns)}")


if __name__ == "__main__":
    main()
