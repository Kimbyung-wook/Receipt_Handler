import os
from datetime import datetime
import json

USAGE_LOG = "usage_log.json"
# --- 2. 호출 횟수 관리 (IP별 로그) ---
def get_usage_data():
    if not os.path.exists(USAGE_LOG): return {}
    with open(USAGE_LOG, "r", encoding="utf-8") as f:
        return json.load(f)

def log_api_call(ip):
    today = datetime.now().strftime("%Y-%m-%d")
    data = get_usage_data()
    if today not in data:
        data[today] = {"total": 0, "ips": {}}
    data[today]["total"] += 1
    data[today]["ips"][ip] = data[today]["ips"].get(ip, 0) + 1
    with open(USAGE_LOG, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return data[today]["total"]