"""Portuguese national holiday and cyclical calendar features."""

from __future__ import annotations

import numpy as np
import pandas as pd


PORTUGAL_HOLIDAYS = {
    # National fixed holidays.
    *(f"{year}-{month_day}" for year in (2024, 2025) for month_day in (
        "01-01", "04-25", "05-01", "06-10", "08-15", "10-05",
        "11-01", "12-01", "12-08", "12-25",
    )),
    # National movable holidays: Good Friday, Easter, and Corpus Christi.
    "2024-03-29", "2024-03-31", "2024-05-30",
    "2025-04-18", "2025-04-20", "2025-06-19",
}


def _datetime_values(df: pd.DataFrame, datetime_col: str) -> pd.Series | None:
    if datetime_col not in df.columns:
        return None
    # API timestamps are UTC; Portuguese calendar semantics use local civil time.
    return pd.to_datetime(df[datetime_col], errors="coerce", utc=True).dt.tz_convert("Europe/Lisbon")


def add_portugal_holidays(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    result = df.copy()
    values = _datetime_values(result, datetime_col)
    if values is None:
        return result
    holiday_dates = {pd.Timestamp(value).date() for value in PORTUGAL_HOLIDAYS}
    result["is_holiday"] = values.dt.date.isin(holiday_dates)
    return result


def add_calendar_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    result = add_portugal_holidays(df, datetime_col)
    values = _datetime_values(result, datetime_col)
    if values is None:
        return result

    result["is_weekend"] = values.dt.dayofweek.ge(5)
    result["is_workday"] = (~result["is_weekend"] & ~result["is_holiday"]).astype(bool)
    result["dayofyear"] = values.dt.dayofyear
    result["weekofyear"] = values.dt.isocalendar().week.astype("Int64")
    result["quarter"] = values.dt.quarter
    result["season"] = values.dt.month.map(
        {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
         6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
    )
    result["is_month_start"] = values.dt.is_month_start
    result["is_month_end"] = values.dt.is_month_end
    result["sin_hour"] = np.sin(2 * np.pi * values.dt.hour / 24)
    result["cos_hour"] = np.cos(2 * np.pi * values.dt.hour / 24)
    result["sin_dayofyear"] = np.sin(2 * np.pi * values.dt.dayofyear / 365.25)
    result["cos_dayofyear"] = np.cos(2 * np.pi * values.dt.dayofyear / 365.25)
    return result
