from datetime import datetime, date
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

def now_ist() -> datetime:
    """Returns the current datetime in Asia/Kolkata timezone."""
    return datetime.now(IST)

def today_ist() -> date:
    """Returns the current date in Asia/Kolkata timezone."""
    return datetime.now(IST).date()

def strftime_now(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Returns the current datetime in IST formatted as string."""
    return now_ist().strftime(fmt)

def strftime_today(fmt: str = "%Y-%m-%d") -> str:
    """Returns the current date in IST formatted as string."""
    return now_ist().strftime(fmt)
