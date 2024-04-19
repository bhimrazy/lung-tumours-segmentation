from datetime import datetime
from zoneinfo import ZoneInfo


def generate_run_id(zone: ZoneInfo = ZoneInfo("Asia/Kathmandu")) -> str:
    """Generate a unique run ID using current UTC date and time.

    Args:
        zone (ZoneInfo, optional): Timezone information. Defaults to Indian Standard Time.

    Returns:
        str: A unique run ID in the format 'run-YYYY-MM-DD-HH-MM-SS'.
    """
    try:
        current_utc_time = datetime.utcnow().astimezone(zone)
        formatted_time = current_utc_time.strftime("%Y-%m-%d-%H-%M-%S")
        return f"run-{formatted_time}"
    except Exception as e:
        # Handle exceptions gracefully
        print(f"Error generating run ID: {e}")
        return None  # Or raise an exception if appropriate
