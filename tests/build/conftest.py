#!/usr/bin/env python3
"""
Fixtures specific to the build/ test module.

Provides:
  sqlite_db        : SQLite URI pointing to tables `sequence_data` and `static_data`
  sequence_data_pl : entity data as a polars DataFrame
  sequence_data_lf : entity data as a polars LazyFrame
  sequence_data_pd : entity data as a pandas DataFrame
  static_data_pl   : static data as a polars DataFrame
  static_data_lf   : static data as a polars LazyFrame
  static_data_pd   : static data as a pandas DataFrame
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sqlite_db(tmp_path_factory: pytest.TempPathFactory, data_dir: Path) -> str:
    """Return a sqlite:/// URI with tables `sequence_data` and `static_data`.

    Both tables are populated from the committed timestep/ data.  Float64
    start/end values survive the pandas → sqlite → connectorx round-trip
    without any datetime string-parsing.
    """
    db_path = tmp_path_factory.mktemp("sqlite") / "test.db"
    uri = f"sqlite:///{db_path}"

    seq_df = pl.read_parquet(data_dir / "timestep" / "sequence_main.parquet").select(
        ["id", "start", "end", "value", "status", "flag_valid", "duration"]
    )
    sta_df = pl.read_csv(data_dir / "static" / "static.csv")

    conn = sqlite3.connect(db_path)
    seq_df.to_pandas().to_sql("sequence_data", conn, if_exists="replace", index=False)
    sta_df.to_pandas().to_sql("static_data", conn, if_exists="replace", index=False)
    conn.close()

    return uri


# ---------------------------------------------------------------------------
# DataFrames: entity (sequence_data)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sequence_data_pl(data_dir: Path) -> pl.DataFrame:
    """Entity data from timestep/sequence_main.parquet as a polars DataFrame."""
    return pl.read_parquet(data_dir / "timestep" / "sequence_main.parquet").select(
        ["id", "start", "end", "value", "status", "flag_valid", "duration"]
    )


@pytest.fixture(scope="session")
def sequence_data_lf(sequence_data_pl: pl.DataFrame) -> pl.LazyFrame:
    """Entity data as a polars LazyFrame (lazy view of sequence_data_pl)."""
    return sequence_data_pl.lazy()


@pytest.fixture(scope="session")
def sequence_data_pd(sequence_data_pl: pl.DataFrame) -> pd.DataFrame:
    """Entity data as a pandas DataFrame."""
    return sequence_data_pl.to_pandas()


# ---------------------------------------------------------------------------
# DataFrames: static (static_data)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def static_data_pl(data_dir: Path) -> pl.DataFrame:
    """Static data from static/static.csv as a polars DataFrame."""
    return pl.read_csv(data_dir / "static" / "static.csv")


@pytest.fixture(scope="session")
def static_data_lf(static_data_pl: pl.DataFrame) -> pl.LazyFrame:
    """Static data as a polars LazyFrame (lazy view of static_data_pl)."""
    return static_data_pl.lazy()


@pytest.fixture(scope="session")
def static_data_pd(static_data_pl: pl.DataFrame) -> pd.DataFrame:
    """Static data as a pandas DataFrame."""
    return static_data_pl.to_pandas()


# ---------------------------------------------------------------------------
# Grouped parametrized DF fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=["pl", "lf", "pd"], ids=["polars_df", "polars_lf", "pandas_df"])
def sequence_df_fixture(
    request: pytest.FixtureRequest,
    sequence_data_pl: pl.DataFrame,
    sequence_data_lf: pl.LazyFrame,
    sequence_data_pd: pd.DataFrame,
) -> pl.DataFrame | pl.LazyFrame | pd.DataFrame:
    """Entity data parametrized over polars DataFrame, polars LazyFrame and pandas DataFrame."""
    return {"pl": sequence_data_pl, "lf": sequence_data_lf, "pd": sequence_data_pd}[
        request.param
    ]


@pytest.fixture(params=["pl", "lf", "pd"], ids=["polars_df", "polars_lf", "pandas_df"])
def static_df_fixture(
    request: pytest.FixtureRequest,
    static_data_pl: pl.DataFrame,
    static_data_lf: pl.LazyFrame,
    static_data_pd: pd.DataFrame,
) -> pl.DataFrame | pl.LazyFrame | pd.DataFrame:
    """Static data parametrized over polars DataFrame, polars LazyFrame and pandas DataFrame."""
    return {"pl": static_data_pl, "lf": static_data_lf, "pd": static_data_pd}[
        request.param
    ]
