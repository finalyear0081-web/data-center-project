"""
DateTime Utilities for Data Center Telemetry and Forecasting
Provides centralized formatting of timestamps into 12-hour format with AM/PM.
"""
from datetime import datetime
from typing import Union, Any
import pandas as pd


def format_forecast_timestamp(dt: Union[datetime, pd.Timestamp, str, Any]) -> str:
    """
    Formats a datetime object or timestamp into 12-hour format with AM/PM.

    Formatting Rules:
        - 14:00:00 -> 2026-01-31 02:00 PM
        - 00:00:00 -> 2026-01-02 00:00 AM (as specified in dynamic forecast examples)
        - 23:58:00 -> 2026-01-01 11:58 PM
        - 12:00:00 -> 2026-01-02 12:00 PM (noon)

    Args:
        dt: Datetime, Timestamp, or string representation.

    Returns:
        Formatted string in YYYY-MM-DD hh:mm AM/PM format.
    """
    if dt is None or pd.isna(dt):
        return ""
    try:
        dt = pd.to_datetime(dt)
    except Exception:
        return str(dt)

    date_str = dt.strftime("%Y-%m-%d")
    hour = dt.hour
    minute = dt.minute

    if hour == 0:
        hour_str = "00"
        ampm = "AM"
    elif hour < 12:
        hour_str = f"{hour:02d}"
        ampm = "AM"
    elif hour == 12:
        hour_str = "12"
        ampm = "PM"
    else:
        hour_str = f"{(hour - 12):02d}"
        ampm = "PM"

    return f"{date_str} {hour_str}:{minute:02d} {ampm}"
