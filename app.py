"""
Sentinel AI – Insider Threat Detection System
Role-aware, slow-burn detection, fixed alert thresholds
"""

from flask import Flask, render_template, jsonify, request, session, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os
import json
import random
import hashlib
import uuid
from datetime import datetime, timedelta
from collections import deque, Counter

try:
    import shap
except ImportError:
    shap = None

app = Flask(__name__)
app.secret_key = os.environ.get("SENTINEL_SECRET", "sentinel_ai_production_key_2024")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "isolation_forest_model.joblib")
DATA_PATH  = os.path.join(BASE_DIR, "data", "FINAL_insider_results.csv")
USERS_PATH = os.path.join(BASE_DIR, "data", "analyst_accounts.json")
CASES_PATH = os.path.join(BASE_DIR, "data", "investigation_cases.json")

FEATURE_COLS = [
    "avg_email_size", "total_attachments", "avg_recipients",
    "total_file_actions", "total_to_rem", "total_from_rem",
    "total_uses_rem", "avg_file_size", "total_device_actions",
    "connected_count", "total_logons", "o", "c", "e", "a", "n"
]

FEATURE_LABELS = [
    "Avg Email Size (KB)", "Total Attachments", "Avg Recipients",
    "File Actions", "Data to USB", "Data from USB",
    "USB Usage Count", "Avg File Size (KB)", "Device Events",
    "Network Connections", "Total Logons", "Openness (OCEAN)",
    "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"
]

# ══════════════════════════════════════════════════════════════════════════════
#  ROLE-BASED BEHAVIOUR BASELINES
#  Each role has its own normal activity envelope.
#  Values are (mean, std) per feature — features follow FEATURE_COLS order.
# ══════════════════════════════════════════════════════════════════════════════
ROLE_BASELINES = {
    # [avg_email_size, attachments, recipients, file_actions, to_rem, from_rem,
    #  uses_rem, file_size, device_actions, connections, logons, o,c,e,a,n]
    "Developer": {
        "mean": [150, 1.5, 2.0, 200, 5,  3,  2,  80,  15, 20, 8, 0.5, 0.5, 0.45, 0.5, 0.4],
        "std":  [ 60, 1.2, 1.0,  80, 8,  4,  3,  40,   8, 10, 3, 0.1, 0.1,  0.1, 0.1, 0.1],
        # Developers legitimately download/move a lot of files and connect devices
        "high_tolerance": ["total_file_actions", "avg_file_size", "connected_count"],
    },
    "Finance": {
        "mean": [200, 2.0, 3.0,  80, 2,  1,  1, 120,   6, 10, 7, 0.4, 0.6, 0.35, 0.55, 0.35],
        "std":  [ 70, 1.5, 1.5,  40, 3,  2,  2,  60,   4,  6, 2, 0.1, 0.1,  0.1,  0.1,  0.1],
        # Finance legitimately handles large files (reports, payroll)
        "high_tolerance": ["avg_file_size", "avg_email_size"],
    },
    "HR": {
        "mean": [180, 3.0, 5.0,  50, 1,  1,  1,  60,   5,  8, 7, 0.55, 0.5, 0.55, 0.6, 0.35],
        "std":  [ 60, 2.0, 2.0,  30, 2,  2,  2,  30,   3,  5, 2,  0.1, 0.1,  0.1, 0.1,  0.1],
        # HR legitimately emails many recipients
        "high_tolerance": ["avg_recipients", "total_attachments"],
    },
    "Manager": {
        "mean": [220, 4.0, 8.0,  60, 2,  1,  1,  70,   5, 12, 8, 0.5, 0.55, 0.6, 0.55, 0.3],
        "std":  [ 80, 2.5, 3.0,  35, 3,  2,  2,  35,   4,  8, 2, 0.1,  0.1, 0.1,  0.1, 0.1],
        # Managers legitimately send large emails to many people
        "high_tolerance": ["avg_recipients", "avg_email_size", "total_attachments"],
    },
    "Admin": {
        "mean": [160, 1.5, 2.5, 150, 3,  2,  2,  60,  20, 18, 9, 0.4, 0.55, 0.4, 0.5, 0.35],
        "std":  [ 60, 1.2, 1.5,  60, 4,  3,  3,  30,  10, 10, 3, 0.1,  0.1, 0.1, 0.1,  0.1],
        # Admins legitimately connect many devices and have high logons
        "high_tolerance": ["total_logons", "total_device_actions", "connected_count"],
    },
    "Security Analyst": {
        "mean": [140, 1.0, 2.0, 120, 4,  2,  2,  70,  25, 30, 9, 0.5, 0.6, 0.45, 0.5, 0.4],
        "std":  [ 50, 0.8, 1.0,  50, 5,  3,  3,  35,  12, 15, 3, 0.1, 0.1,  0.1, 0.1, 0.1],
        # Security analysts legitimately access many files and devices
        "high_tolerance": ["total_file_actions", "total_device_actions", "connected_count", "total_logons"],
    },
}
DEFAULT_ROLE_BASELINE = ROLE_BASELINES["Developer"]

# ══════════════════════════════════════════════════════════════════════════════
#  SLOW-BURN TRACKER
#  Tracks the number of anomalous ML signals in the last N events (per user).
#  A single anomaly never escalates to HIGH — only a persistent pattern does.
# ══════════════════════════════════════════════════════════════════════════════
# Window of events to look back over per user
SLOW_BURN_WINDOW = 20

# How many anomalous events in the window before we escalate:
#   >= SLOW_BURN_HIGH_THRESHOLD → HIGH risk
#   >= SLOW_BURN_MED_THRESHOLD  → MEDIUM risk
#   < SLOW_BURN_MED_THRESHOLD   → LOW risk (even if latest event is anomalous)
#
# With a window of 20 and these thresholds:
#   HIGH  requires 75% anomaly rate (15/20 events flagged) → sustained campaign
#   MED   requires 40% anomaly rate (8/20 events flagged) → noticeable pattern
#   LOW   everything below 40% stays green
SLOW_BURN_HIGH_THRESHOLD_PCT = 0.75   # proportion
SLOW_BURN_MED_THRESHOLD_PCT  = 0.40

# Alert-threshold sensitivity adjusts these at runtime (from settings)
SENSITIVITY_MAP = {
    "Low":    (0.85, 0.55),   # harder to trigger (fewer false positives)
    "Medium": (0.75, 0.40),   # balanced
    "High":   (0.60, 0.30),   # more sensitive
}

def generate_synthetic_employee_rows(count: int = 100) -> list[dict]:
    first_names = ["Ava", "Noah", "Liam", "Mia", "Oliver", "Sophia", "Ethan", "Isabella",
                   "Lucas", "Charlotte", "James", "Amelia", "Benjamin", "Harper", "Michael", "Evelyn"]
    last_names = ["Johnson", "Smith", "Lee", "Brown", "Garcia", "Miller", "Davis", "Wilson",
                  "Taylor", "Moore", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin"]
    departments = ["Engineering", "IT", "Finance", "HR", "Security", "Operations",
                   "Sales", "Legal", "Marketing", "Research", "Support", "Admin"]
    teams = ["Alpha", "Beta", "Gamma", "Delta", "Sigma", "Phoenix", "Omega", "Tesla"]
    units = ["Corporate", "Technology", "Risk", "Operations", "Research", "Infrastructure", "Product", "Services"]
    supervisor_names = [
        "Alice Johnson", "Bob Smith", "Carol White", "David Lee", "Emma Brown",
        "Frank Garcia", "Grace Wilson", "Henry Taylor", "Iris Martinez", "Jack Anderson"
    ]
    rows = []
    anomaly_count = max(3, count // 10)
    anomaly_indices = set(random.sample(range(count), anomaly_count))

    for i in range(count):
        first = random.choice(first_names)
        last = random.choice(last_names)
        name = f"{first} {last}"
        user_id = f"user{100 + i}"
        anomaly = "anomaly" if i in anomaly_indices else "normal"
        dept = random.choice(departments)
        row = {
            "user_id": user_id,
            "name": name,
            "email": f"{name.lower().replace(' ', '')}@dtaa.com",
            "role": random.choice(["Employee", "Specialist", "Manager", "Coordinator"]),
            "department_name": dept,
            "team_name": random.choice(teams),
            "functional_unit_name": random.choice(units),
            "supervisor": random.choice(supervisor_names),
            "anomaly": anomaly,
            "reason": "Potential policy review required: unusual activity detected." if anomaly == "anomaly"
                      else "Standard monitoring. No immediate action required.",
        }
        if anomaly == "normal":
            row.update({
                "avg_email_size": random.uniform(80, 320),
                "total_attachments": random.uniform(0, 3),
                "avg_recipients": random.uniform(1, 4),
                "total_file_actions": random.uniform(20, 120),
                "total_to_rem": random.uniform(0, 30),
                "total_from_rem": random.uniform(0, 8),
                "total_uses_rem": random.uniform(0, 5),
                "avg_file_size": random.uniform(20, 140),
                "total_device_actions": random.uniform(5, 22),
                "connected_count": random.uniform(5, 24),
                "total_logons": random.uniform(3, 9),
                "o": random.uniform(0.2, 0.8), "c": random.uniform(0.3, 0.8),
                "e": random.uniform(0.2, 0.8), "a": random.uniform(0.3, 0.8),
                "n": random.uniform(0.1, 0.6),
            })
        else:
            row.update({
                "avg_email_size": random.uniform(320, 800),
                "total_attachments": random.uniform(3, 10),
                "avg_recipients": random.uniform(4, 10),
                "total_file_actions": random.uniform(200, 1200),
                "total_to_rem": random.uniform(100, 800),
                "total_from_rem": random.uniform(50, 350),
                "total_uses_rem": random.uniform(50, 400),
                "avg_file_size": random.uniform(150, 650),
                "total_device_actions": random.uniform(25, 180),
                "connected_count": random.uniform(15, 70),
                "total_logons": random.uniform(8, 18),
                "o": random.uniform(0.2, 0.9), "c": random.uniform(0.2, 0.9),
                "e": random.uniform(0.2, 0.9), "a": random.uniform(0.2, 0.9),
                "n": random.uniform(0.3, 0.9),
            })
        rows.append(row)
    return rows

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
try:
    model = joblib.load(MODEL_PATH)
    print(f"[✓] Isolation Forest loaded: {model.n_estimators} trees, contamination={model.contamination}")
except Exception as e:
    print(f"[✗] Model load failed: {e}")
    model = None

SHAP_EXPLAINER = None
if model is not None and shap is not None:
    try:
        SHAP_EXPLAINER = shap.TreeExplainer(model)
        print("[✓] SHAP TreeExplainer initialized")
    except Exception as e:
        print(f"[⚠] SHAP explainer initialization failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD EMPLOYEE DATASET
# ══════════════════════════════════════════════════════════════════════════════
try:
    df_employees = pd.read_csv(DATA_PATH)
    df_employees = df_employees.dropna(subset=FEATURE_COLS)
    normal_users  = df_employees[df_employees["anomaly"] == "normal"]
    anomaly_users = df_employees[df_employees["anomaly"] == "anomaly"]
    POPULATION_MEAN = normal_users[FEATURE_COLS].mean().values
    POPULATION_STD  = normal_users[FEATURE_COLS].std().values + 1e-9

    if model:
        X_normal  = normal_users[FEATURE_COLS].values
        X_anomaly = anomaly_users[FEATURE_COLS].values
        normal_scores  = model.decision_function(X_normal)
        anomaly_scores = model.decision_function(X_anomaly)

        # The ML score is used only as a signal to the slow-burn tracker —
        # not directly as a risk level.  An event is counted as "anomalous"
        # only when the score falls below the bottom 10% of normal users.
        # This keeps the raw signal rate low so the slow-burn thresholds
        # do the work of escalation.
        ML_ANOMALY_THRESHOLD = np.percentile(normal_scores, 10)   # bottom 10% of normals
        print(f"[✓] ML anomaly signal threshold (p10 of normals): {ML_ANOMALY_THRESHOLD:.4f}")
    else:
        ML_ANOMALY_THRESHOLD = -0.05

except Exception as e:
    print(f"[✗] Dataset load failed: {e}")
    df_employees = pd.DataFrame(generate_synthetic_employee_rows(100))
    normal_users  = df_employees[df_employees["anomaly"] == "normal"]
    POPULATION_MEAN = normal_users[FEATURE_COLS].mean().values
    POPULATION_STD  = normal_users[FEATURE_COLS].std().values + 1e-9
    ML_ANOMALY_THRESHOLD = -0.05
    if model:
        X_normal = normal_users[FEATURE_COLS].values
        normal_scores = model.decision_function(X_normal)
        ML_ANOMALY_THRESHOLD = np.percentile(normal_scores, 10)

# ══════════════════════════════════════════════════════════════════════════════
#  ANALYST ACCOUNT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def load_analyst_accounts() -> dict:
    if os.path.exists(USERS_PATH):
        with open(USERS_PATH) as f:
            return json.load(f)
    defaults = {
        "admin@dtaa.com":    {"password": _hash("Sentinel2024!"), "name": "System Administrator", "role": "admin",    "must_change": False},
        "analyst1@dtaa.com": {"password": _hash("Analyst2024!"),  "name": "Sarah Chen",           "role": "analyst",  "must_change": True},
        "analyst2@dtaa.com": {"password": _hash("Analyst2024!"),  "name": "Marcus Torres",        "role": "analyst",  "must_change": True},
    }
    save_analyst_accounts(defaults)
    return defaults

def save_analyst_accounts(accounts: dict):
    os.makedirs(os.path.dirname(USERS_PATH), exist_ok=True)
    with open(USERS_PATH, "w") as f:
        json.dump(accounts, f, indent=2)

ANALYST_ACCOUNTS = load_analyst_accounts()

# ══════════════════════════════════════════════════════════════════════════════
#  INVESTIGATION CASE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
def load_cases() -> dict:
    if os.path.exists(CASES_PATH):
        with open(CASES_PATH) as f:
            raw_cases = json.load(f)
        for case in raw_cases.values():
            if not isinstance(case.get("notes", []), list):
                case["notes"] = []
            if not isinstance(case.get("events", []), list):
                case["events"] = []
        return raw_cases
    return {}

def save_cases(cases: dict):
    os.makedirs(os.path.dirname(CASES_PATH), exist_ok=True)
    with open(CASES_PATH, "w") as f:
        json.dump(cases, f, indent=2)

INVESTIGATION_CASES = load_cases()

DEFAULT_SETTINGS = {
    "auto_refresh": 18,
    "model_sensitivity": "Medium",
    "alert_threshold": "Medium",
    "warning_template": "Please stop sending external attachments and maintain normal workplace communication. Continued misuse may result in account lockout.",
}
CURRENT_SETTINGS = DEFAULT_SETTINGS.copy()

# ══════════════════════════════════════════════════════════════════════════════
#  RUNTIME STATE
# ══════════════════════════════════════════════════════════════════════════════
USER_HISTORY:  dict[str, deque] = {}   # {user_id: deque(maxlen=SLOW_BURN_WINDOW)}
EVENT_HISTORY: dict[str, deque] = {}   # {user_id: deque(maxlen=10) of event_type strings}
ALERTS:        list[dict] = []
RESOLVED:      set = set()
LOCKED_ACCOUNTS: set = set()
EVENT_LOG:     deque = deque(maxlen=500)
PROCESSED_LOGS: deque = deque(maxlen=500)
THREAT_STATS:  dict = {}

VALID_EMPLOYEE_ROLES = ["Developer", "HR", "Finance", "Manager", "Admin", "Security Analyst"]
ROLE_DEPARTMENT_MAP = {
    "Engineering": "Developer", "IT": "Developer", "Research": "Developer",
    "Finance": "Finance", "HR": "HR", "Security": "Security Analyst",
    "Admin": "Admin", "Operations": "Manager", "Sales": "Manager",
    "Legal": "Manager", "Support": "Manager", "Marketing": "Manager",
}

RAW_LOG_FILES = {
    "login_logs":        os.path.join(BASE_DIR, "data", "login_logs.csv"),
    "email_logs":        os.path.join(BASE_DIR, "data", "email_logs.csv"),
    "file_logs":         os.path.join(BASE_DIR, "data", "file_logs.csv"),
    "device_logs":       os.path.join(BASE_DIR, "data", "device_logs.csv"),
    "decoy_logs":        os.path.join(BASE_DIR, "data", "decoy_logs.csv"),
    "psychometric_logs": os.path.join(BASE_DIR, "data", "psychometric_logs.csv"),
}
RAW_LOG_STORE: dict[str, list[dict]] = {key: [] for key in RAW_LOG_FILES}

# ══════════════════════════════════════════════════════════════════════════════
#  RAW LOG GENERATION
# ══════════════════════════════════════════════════════════════════════════════
EVENT_DISTRIBUTION = {
    "LOGON_SUCCESS": 35, "FILE_ACCESS": 30, "EMAIL_EXTERNAL": 12, "VPN_CONNECTION": 8,
    "LOGON_FAILED": 5, "USB_CONNECTED": 4, "DATA_COPY_USB": 2, "MASS_DOWNLOAD": 2,
    "AFTER_HOURS_ACCESS": 1, "DLP_POLICY_VIOLATION": 0.5,
    "PRIVILEGE_ESCALATION": 0.3, "HONEYPOT_FILE": 0.2,
}
EVENT_POOL = [k for k, w in EVENT_DISTRIBUTION.items() for _ in range(int(w * 10))]

FEATURE_DELTAS = {
    "LOGON_SUCCESS":         {"total_logons": (1, 2)},
    "LOGON_FAILED":          {"total_logons": (1, 3)},
    "FILE_ACCESS":           {"total_file_actions": (5, 20), "avg_file_size": (10, 100)},
    "USB_CONNECTED":         {"connected_count": (1, 2), "total_device_actions": (1, 3)},
    "DATA_COPY_USB":         {"total_to_rem": (100, 800), "total_uses_rem": (100, 600), "avg_file_size": (50, 300)},
    "EMAIL_EXTERNAL":        {"total_attachments": (1, 4), "avg_recipients": (2, 6), "avg_email_size": (50, 400)},
    "MASS_DOWNLOAD":         {"total_file_actions": (200, 1500), "avg_file_size": (100, 500)},
    "DLP_POLICY_VIOLATION":  {"total_to_rem": (500, 2000), "avg_email_size": (200, 800), "total_attachments": (3, 10)},
    "PRIVILEGE_ESCALATION":  {"total_device_actions": (2, 8), "total_logons": (1, 3)},
    "AFTER_HOURS_ACCESS":    {"total_logons": (1, 2)},
    "HONEYPOT_FILE":         {"total_file_actions": (1, 5)},
    "VPN_CONNECTION":        {"connected_count": (1, 2)},
}

def get_user_role(meta: dict) -> str:
    role = meta.get("role")
    if role and role in VALID_EMPLOYEE_ROLES:
        return role
    return ROLE_DEPARTMENT_MAP.get(meta.get("department", ""), "Developer")

def get_role_baseline(role: str) -> dict:
    return ROLE_BASELINES.get(role, DEFAULT_ROLE_BASELINE)

def write_raw_log_csv(source_name: str, record: dict):
    path = RAW_LOG_FILES.get(source_name)
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([record])
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

def append_raw_log(source_name: str, record: dict):
    record = record.copy()
    record["log_type"] = "raw"
    RAW_LOG_STORE.setdefault(source_name, []).append(record)
    write_raw_log_csv(source_name, record)
    return record

def infer_raw_log_source(record: dict) -> str | None:
    if "login_status" in record and "device_name" in record:
        return "login_logs"
    if "recipient_count" in record and "email_id" in record:
        return "email_logs"
    if "action" in record and "file_name" in record:
        return "file_logs"
    if "device_type" in record and "action" in record and "event_id" in record:
        return "device_logs"
    if "decoy_file" in record:
        return "decoy_logs"
    if "stress_level" in record and "job_satisfaction" in record:
        return "psychometric_logs"
    return None

def infer_event_type_from_raw(raw_log: dict) -> str:
    source = raw_log.get("source_type") or infer_raw_log_source(raw_log)
    if source == "login_logs":
        return "LOGON_FAILED" if raw_log.get("login_status") == "FAILED" else "LOGON_SUCCESS"
    if source == "email_logs":
        return "EMAIL_EXTERNAL"
    if source == "file_logs":
        action = raw_log.get("action", "OPEN")
        return "MASS_DOWNLOAD" if action in {"DOWNLOAD", "UPLOAD", "MOVE"} else "FILE_ACCESS"
    if source == "device_logs":
        return "USB_CONNECTED" if raw_log.get("action") in {"CONNECT", "USB_CONNECTED"} else "FILE_ACCESS"
    if source == "decoy_logs":
        return "HONEYPOT_FILE"
    if source == "psychometric_logs":
        return "AFTER_HOURS_ACCESS"
    return "LOGON_SUCCESS"

def _clamp(value: float, minimum: float = 0.05, maximum: float = 0.95) -> float:
    return max(minimum, min(maximum, value))

def _parse_datetime(ts: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts, fmt)
        except Exception:
            continue
    return None

def _filter_records_by_time(records: list[dict], start_ts: str | None, end_ts: str | None) -> list[dict]:
    if not start_ts and not end_ts:
        return records
    start = _parse_datetime(start_ts) if start_ts else None
    end   = _parse_datetime(end_ts)   if end_ts   else None
    if end and start and end < start:
        start, end = end, start
    if end and end.time() == datetime.min.time():
        end = end + timedelta(hours=23, minutes=59, seconds=59)
    filtered = []
    for record in records:
        timestamp = record.get("timestamp") or record.get("date")
        dt = _parse_datetime(str(timestamp or ""))
        if dt is None:
            continue
        if start and dt < start:
            continue
        if end and dt > end:
            continue
        filtered.append(record)
    return filtered

def _matches_query(record: dict, query: str | None) -> bool:
    if not query:
        return True
    query = str(query).lower().strip()
    return any(query in str(value).lower() for value in record.values() if value is not None)

# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION — ROLE-AWARE
# ══════════════════════════════════════════════════════════════════════════════
def engineer_features_from_raw_logs(uid: str) -> list | None:
    """Aggregate raw logs to compute the model feature vector, role-adjusted."""
    login_logs       = [r for r in RAW_LOG_STORE.get("login_logs",        []) if r.get("user_id") == uid]
    email_logs       = [r for r in RAW_LOG_STORE.get("email_logs",        []) if r.get("user_id") == uid]
    file_logs        = [r for r in RAW_LOG_STORE.get("file_logs",         []) if r.get("user_id") == uid]
    device_logs      = [r for r in RAW_LOG_STORE.get("device_logs",       []) if r.get("user_id") == uid]
    decoy_logs       = [r for r in RAW_LOG_STORE.get("decoy_logs",        []) if r.get("user_id") == uid]
    psychometric_logs= [r for r in RAW_LOG_STORE.get("psychometric_logs", []) if r.get("employee_id") == uid or r.get("user_id") == uid]

    if not any([login_logs, email_logs, file_logs, device_logs, decoy_logs, psychometric_logs]):
        return None

    login_count      = len(login_logs)
    email_count      = max(1, len(email_logs))
    email_volume     = sum(float(r.get("recipient_count", 0) or 0) for r in email_logs)
    attachment_count = sum(1 for r in email_logs if bool(r.get("attachment_present")))
    total_file_transfer_MB = sum(float(r.get("file_size_MB", 0) or 0) for r in file_logs
                                 if str(r.get("action", "")).upper() in {"DOWNLOAD", "UPLOAD", "MOVE"})
    usb_insertions   = sum(1 for r in device_logs if str(r.get("device_type", "")).upper() == "USB"
                           and str(r.get("action", "")).upper() in {"CONNECT", "USB_CONNECTED"})
    decoy_access_count = len(decoy_logs)
    total_file_actions = len(file_logs) + sum(1 for r in file_logs
                           if str(r.get("action", "")).upper() in {"DOWNLOAD", "UPLOAD", "MOVE"})
    total_device_actions = len(device_logs)
    connected_count = len({(r.get("ip_address"), r.get("device_name")) for r in login_logs if r.get("ip_address")}) \
                    + len({r.get("device_type") for r in device_logs if r.get("device_type")})
    avg_file_size   = max(1.0, sum(float(r.get("file_size_MB", 0) or 0) for r in file_logs) / max(1, len(file_logs)))
    avg_email_size  = max(5.0, sum(float(r.get("attachment_size_MB", 0) or 0) for r in email_logs) / email_count)
    avg_recipients  = max(1.0, email_volume / email_count)

    stress_average  = 5.0
    satisfaction_average = 5.0
    fatigue_average = 5.0
    if psychometric_logs:
        stress_average       = sum(float(r.get("stress_level",    5) or 5) for r in psychometric_logs) / len(psychometric_logs)
        satisfaction_average = sum(float(r.get("job_satisfaction",5) or 5) for r in psychometric_logs) / len(psychometric_logs)
        fatigue_average      = sum(float(r.get("fatigue_score",   5) or 5) for r in psychometric_logs) / len(psychometric_logs)

    # Get role-aware baseline
    meta = get_user_metadata(uid)
    role = get_user_role(meta)
    rb   = get_role_baseline(role)
    base = rb["mean"].copy()

    features = base.copy()
    features[0]  = max(5,   avg_email_size * 20)
    features[1]  = max(features[1], float(attachment_count) * 1.5)
    features[2]  = max(1.0, avg_recipients)
    features[3]  = max(features[3], float(total_file_actions) * 1.25)
    features[4]  = max(features[4], float(total_file_transfer_MB) * 2.5)
    features[5]  = max(features[5], float(total_file_transfer_MB) * 0.35)
    features[6]  = max(features[6], float(usb_insertions + decoy_access_count))
    features[7]  = max(features[7], avg_file_size * 2.0)
    features[8]  = max(features[8], float(total_device_actions) * 1.5)
    features[9]  = max(features[9], float(connected_count))
    features[10] = max(features[10], float(login_count))
    features[11] = _clamp(0.35 + (satisfaction_average - 5) * 0.06 - (stress_average - 5) * 0.02)
    features[12] = _clamp(0.40 + (stress_average - 5) * 0.04 - (fatigue_average - 5) * 0.03)
    features[13] = _clamp(0.45 + (satisfaction_average - 5) * 0.05 - (stress_average - 5) * 0.02)
    features[14] = _clamp(0.40 + (fatigue_average - 5) * 0.05 - (satisfaction_average - 5) * 0.02)
    features[15] = _clamp(0.50 + (stress_average - 5) * 0.06 - (satisfaction_average - 5) * 0.04)
    return [float(round(x, 4)) for x in features]


def extract_features_from_baseline(uid: str, event_type: str = None) -> list:
    """Fallback: role-aware baseline + event deltas."""
    meta = get_user_metadata(uid)
    role = get_user_role(meta)
    rb   = get_role_baseline(role)

    if not df_employees.empty:
        row = df_employees[df_employees["user_id"] == uid]
        if not row.empty:
            baseline = row.iloc[0][FEATURE_COLS].values.astype(float).tolist()
        else:
            baseline = rb["mean"].copy()
    else:
        baseline = rb["mean"].copy()

    deltas = FEATURE_DELTAS.get(event_type, {})
    for feat, (lo, hi) in deltas.items():
        if feat in FEATURE_COLS:
            idx = FEATURE_COLS.index(feat)
            baseline[idx] += random.uniform(lo, hi)
    return baseline


def extract_features(uid: str, event_type: str = None) -> list:
    engineered = engineer_features_from_raw_logs(uid)
    if engineered is not None:
        return engineered
    return extract_features_from_baseline(uid, event_type)


def compute_shap_values(features: list) -> list:
    if len(features) != len(FEATURE_COLS):
        return [0.0] * len(FEATURE_COLS)
    if model is not None and SHAP_EXPLAINER is not None:
        try:
            df_input = pd.DataFrame([features], columns=FEATURE_COLS)
            shap_values = SHAP_EXPLAINER.shap_values(df_input)
            if isinstance(shap_values, np.ndarray):
                shap_array = shap_values
            elif isinstance(shap_values, list) and len(shap_values) == 1:
                shap_array = np.array(shap_values[0])
            else:
                shap_array = np.array(shap_values).reshape(1, -1)
            return [round(float(v), 4) for v in shap_array.flatten().tolist()[:len(FEATURE_COLS)]]
        except Exception:
            pass
    shap_vals = []
    for i, fval in enumerate(features):
        deviation = (fval - POPULATION_MEAN[i]) / POPULATION_STD[i]
        shap_vals.append(round(deviation, 4))
    return shap_vals

# ══════════════════════════════════════════════════════════════════════════════
#  RISK ENGINE — SLOW-BURN, ROLE-AWARE
# ══════════════════════════════════════════════════════════════════════════════
def is_event_anomalous_for_role(uid: str, features: list, ml_score: float, event_type: str) -> bool:
    """
    Determine whether a single event is anomalous, taking the user's role into account.
    An event is NOT anomalous just because it involves high file volume if the user is a Developer,
    or many recipients if the user is a Manager, etc.
    """
    meta = get_user_metadata(uid)
    role = get_user_role(meta)
    rb   = get_role_baseline(role)
    high_tolerance_features = rb.get("high_tolerance", [])

    # --- Rule: decoy/honeypot access is ALWAYS anomalous regardless of role ---
    if event_type == "HONEYPOT_FILE":
        return True

    # --- ML signal: is the score below our threshold? ---
    ml_says_anomalous = ml_score < ML_ANOMALY_THRESHOLD

    if not ml_says_anomalous:
        return False

    # --- Role check: if the ML flags it but the driver is a tolerated feature, dampen ---
    # Compute per-feature z-scores relative to role baseline
    role_mean = np.array(rb["mean"])
    role_std  = np.array(rb["std"])
    z_scores  = (np.array(features) - role_mean) / (role_std + 1e-9)

    # Find the feature driving the anomaly signal (highest absolute z-score)
    top_driver_idx = int(np.argmax(np.abs(z_scores)))
    top_driver     = FEATURE_COLS[top_driver_idx]

    # If the top driver is a high-tolerance feature for this role, require
    # a much stronger ML signal before flagging it as anomalous
    if top_driver in high_tolerance_features:
        # Only count it if the ML score is really far out (bottom 5% of normals)
        very_strong_threshold = ML_ANOMALY_THRESHOLD * 1.5  # further from normal
        return ml_score < very_strong_threshold

    return True


def map_threat_category(uid: str, features: list, event_type: str, ml_score: float, burn_pct: float) -> str:
    threat_mapping = {
        "HONEYPOT_FILE":         "Honeypot File Access",
        "DATA_COPY_USB":         "Data Exfiltration (USB)",
        "DLP_POLICY_VIOLATION":  "Policy Violation",
        "PRIVILEGE_ESCALATION":  "Privilege Escalation",
        "MASS_DOWNLOAD":         "Bulk Data Download",
        "LOGON_FAILED":          "Unauthorized Access Attempt",
        "AFTER_HOURS_ACCESS":    "After-Hours Activity",
        "VPN_CONNECTION":        "Suspicious VPN Usage",
        "FILE_ACCESS":           "Sensitive File Access",
        "EMAIL_EXTERNAL":        "External Communication",
    }
    if event_type in threat_mapping:
        base = threat_mapping[event_type]
    else:
        usb_usage  = features[4] + features[5] + features[6]
        file_volume = features[3]
        if usb_usage > POPULATION_MEAN[4:7].sum() * 3 or file_volume > POPULATION_MEAN[3] * 6:
            base = "Suspicious Data Transfer"
        else:
            base = "Anomalous Behavior Pattern"

    # Decoy access is always a definitive indicator — tag it as such
    if event_type == "HONEYPOT_FILE":
        return "Honeypot File Access"

    # Slow-burn escalation label
    if burn_pct >= SLOW_BURN_HIGH_THRESHOLD_PCT:
        return f"{base} [Persistent]"
    return base


def risk_engine(uid: str, features: list, event_type: str = "LOGON_SUCCESS") -> tuple[str, float, str]:
    """
    Returns: (risk_level, ml_score, threat_category)

    Risk levels are driven by the slow-burn window, not a single event.
    - A single anomalous event keeps the user at LOW (or MEDIUM at most if already building).
    - Only a sustained pattern of anomalous events over the window escalates to HIGH.
    - Decoy/honeypot access is the single exception: immediate HIGH.
    """
    # ML Score
    if model:
        df_input = pd.DataFrame([features], columns=FEATURE_COLS)
        ml_score = float(model.decision_function(df_input)[0])
    else:
        z_score  = (np.mean(features[:10]) - POPULATION_MEAN[:10].mean()) / (POPULATION_STD[:10].mean() + 1e-9)
        ml_score = -z_score * 0.03

    # Sensitivity adjustment from settings
    sensitivity_adjust = {"Low": -0.01, "Medium": 0.0, "High": 0.01}.get(
        CURRENT_SETTINGS.get("model_sensitivity", "Medium"), 0.0)
    adjusted_ml_score = ml_score + sensitivity_adjust

    # --- Decoy file access: immediate HIGH, skip slow-burn ---
    if event_type == "HONEYPOT_FILE":
        threat_cat = map_threat_category(uid, features, event_type, ml_score, 1.0)
        return "HIGH", round(ml_score, 4), threat_cat

    # --- Slow-burn window ---
    if uid not in USER_HISTORY:
        USER_HISTORY[uid] = deque(maxlen=SLOW_BURN_WINDOW)

    is_anomalous = 1 if is_event_anomalous_for_role(uid, features, adjusted_ml_score, event_type) else 0
    USER_HISTORY[uid].append(is_anomalous)

    window_size = len(USER_HISTORY[uid])
    burn_count  = sum(USER_HISTORY[uid])
    burn_pct    = burn_count / window_size if window_size > 0 else 0

    threat_cat  = map_threat_category(uid, features, event_type, ml_score, burn_pct)

    # Get thresholds from settings
    high_thresh, med_thresh = SENSITIVITY_MAP.get(
        CURRENT_SETTINGS.get("alert_threshold", "Medium"), (0.75, 0.40))

    # Risk tiering purely based on slow-burn pattern
    if burn_pct >= high_thresh:
        risk_level = "HIGH"
    elif burn_pct >= med_thresh:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return risk_level, round(ml_score, 4), threat_cat

# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def sample_users(n: int) -> list[str]:
    if df_employees.empty:
        return [f"user{100 + random.randint(0, 99)}" for _ in range(min(n, 100))]
    available = df_employees["user_id"].tolist()[:100]
    return random.sample(available, min(n, len(available)))

def get_user_metadata(uid: str) -> dict:
    if df_employees.empty:
        return {}
    row = df_employees[df_employees["user_id"] == uid]
    if row.empty:
        return {}
    r    = row.iloc[0]
    dept = r.get("department_name", "Unknown")
    supervisor = r.get("supervisor", "Unknown")
    team = r.get("team_name", "General")
    unit = r.get("functional_unit_name", "Corporate")

    def _fallback(val, choices):
        if not val or str(val) in ("0", "Unknown", "nan", ""):
            return random.choice(choices)
        return val

    dept       = _fallback(dept,       ["Engineering", "IT", "Finance", "HR", "Operations", "Sales", "Security", "Legal"])
    supervisor = _fallback(supervisor, ["Alice Johnson", "Bob Smith", "Carol White", "David Lee", "Emma Brown"])
    team       = _fallback(team,       ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"])
    unit       = _fallback(unit,       ["Corporate", "Subsidiaries", "Operations", "Technology", "Research"])

    return {
        "user_id": uid,
        "name":         r.get("name", uid),
        "email":        r.get("email", f"{uid.lower()}@dtaa.com"),
        "role":         r.get("role", "Employee"),
        "department":   dept,
        "team":         team,
        "business_unit": unit,
        "supervisor":   supervisor,
        "reason":       r.get("reason", "Standard monitoring. No immediate action required."),
        "is_active":    bool(r.get("is_active", True)),
        "actual_label": r.get("anomaly", "normal"),
    }

def generate_random_event_timestamp(after_hours: bool = False) -> datetime:
    now  = datetime.now()
    hour = random.choice([19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]) if after_hours \
           else random.choice([8, 9, 10, 11, 13, 14, 15, 16, 17])
    ts   = now.replace(hour=hour, minute=random.randint(0,59), second=random.randint(0,59), microsecond=0)
    if ts > now:
        ts -= timedelta(days=1)
    return ts

def generate_location(department: str) -> str:
    return f"{random.choice(['New York','London','Dublin','Berlin','Singapore','Sydney','Toronto'])} Office"

def generate_source_ip():
    return f"10.{random.randint(10,99)}.{random.randint(1,254)}.{random.randint(1,254)}"

def generate_hostname(dept: str):
    prefix = dept[:3].upper() if dept != "Unknown" else "WKS"
    return f"{prefix}-PC{random.randint(100,999)}"

def generate_login_log(uid: str, meta: dict, abnormal: bool = False) -> dict:
    status      = "FAILED" if abnormal and random.random() < 0.55 else "SUCCESS"
    after_hours = abnormal or random.random() < 0.12
    timestamp   = generate_random_event_timestamp(after_hours=after_hours)
    return {
        "source_type": "login_logs", "event_id": 4625 if status == "FAILED" else 4624,
        "login_id": str(uuid.uuid4()), "employee_id": uid, "user_id": uid,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "login_status": status, "device_name": generate_hostname(meta.get("department","Unknown")),
        "ip_address": generate_source_ip(), "location": generate_location(meta.get("department","Unknown")),
        "department": meta.get("department"), "role": get_user_role(meta),
    }

def generate_email_log(uid: str, meta: dict, abnormal: bool = False) -> dict:
    role = get_user_role(meta)
    recipient_count = random.randint(3, 8) if role == "Manager" else random.randint(1, 3)
    if abnormal:
        recipient_count = random.randint(6, 15)
    attachment_present  = abnormal or random.random() < 0.35
    attachment_size_MB  = round(random.uniform(0.5, 12.0), 2) if attachment_present else 0.0
    external            = abnormal or random.random() < 0.25
    timestamp           = generate_random_event_timestamp(after_hours=abnormal)
    return {
        "source_type": "email_logs", "email_id": str(uuid.uuid4()),
        "employee_id": uid, "user_id": uid,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "recipient_count": recipient_count, "attachment_present": attachment_present,
        "attachment_size_MB": attachment_size_MB, "external_recipient": external,
        "department": meta.get("department"), "role": role,
    }

def generate_file_log(uid: str, meta: dict, abnormal: bool = False) -> dict:
    role    = get_user_role(meta)
    actions = [("DOWNLOAD","MOVE","UPLOAD") if abnormal else ("OPEN","DOWNLOAD","UPLOAD","MOVE","DELETE")][0]
    action  = random.choice(actions)
    file_choices = {
        "Finance":  ["payroll.xlsx","audit_report.pdf","tax_forms.xlsx"],
        "HR":       ["org_chart.xlsx","employee_records.csv","onboarding.docx"],
        "Developer":["source_code.zip","config.yaml","build_output.tar"],
    }.get(role, ["budget.xlsx","strategy.docx","project_plan.pptx","customer_data.csv"])
    file_name = random.choice(file_choices)
    size      = round(random.uniform(0.5, 120.0), 2) if action in {"OPEN","DELETE"} else round(random.uniform(10.0, 1800.0), 2)
    timestamp = generate_random_event_timestamp(after_hours=abnormal)
    return {
        "source_type": "file_logs", "event_id": 4663,
        "employee_id": uid, "user_id": uid,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "action": action, "file_name": file_name, "file_size_MB": size,
        "source_path": f"/home/{uid}/documents/{file_name}",
        "destination_path": f"/home/{uid}/downloads/{file_name}" if action in {"DOWNLOAD","MOVE","UPLOAD"} else f"/home/{uid}/documents/{file_name}",
        "department": meta.get("department"), "role": role,
    }

def generate_device_log(uid: str, meta: dict, abnormal: bool = False) -> dict:
    device_type = random.choice(["USB","External HDD","Laptop","Mobile"])
    action      = "USB_CONNECTED" if device_type == "USB" else random.choice(["CONNECT","DISCONNECT"])
    if abnormal and device_type != "USB":
        device_type, action = "USB", "USB_CONNECTED"
    timestamp = generate_random_event_timestamp(after_hours=abnormal)
    return {
        "source_type": "device_logs", "event_id": 20001,
        "employee_id": uid, "user_id": uid,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "device_type": device_type, "action": action,
        "department": meta.get("department"), "role": get_user_role(meta),
    }

def generate_decoy_log(uid: str, meta: dict, abnormal: bool = False) -> dict:
    """Decoy/honeypot access — always treated as a definitive threat signal."""
    timestamp = generate_random_event_timestamp(after_hours=abnormal)
    return {
        "source_type": "decoy_logs", "event_id": 90001,
        "employee_id": uid, "user_id": uid,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "decoy_file": random.choice(["decoy_salary.xlsx","decoy_pipeline.docx","decoy_credentials.txt","decoy_plan.pptx"]),
        "action": random.choice(["OPENED","ACCESSED","MODIFIED"]),
        "department": meta.get("department"), "role": get_user_role(meta),
        "level": "Critical",
    }

def generate_psychometric_log(uid: str, meta: dict, abnormal: bool = False) -> dict:
    stress_level = round(random.uniform(2.0, 9.5) if abnormal else random.uniform(3.0, 7.0), 1)
    satisfaction = round(random.uniform(1.5, 5.5) if abnormal else random.uniform(4.0, 8.5), 1)
    fatigue      = round(random.uniform(5.0, 9.5) if abnormal else random.uniform(2.5, 6.5), 1)
    return {
        "source_type": "psychometric_logs",
        "employee_id": uid, "user_id": uid,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "stress_level": stress_level, "job_satisfaction": satisfaction, "fatigue_score": fatigue,
        "department": meta.get("department"), "role": get_user_role(meta),
    }

def choose_raw_log_source(meta: dict, abnormal: bool) -> str:
    role = get_user_role(meta)
    base = {
        "Developer":        ["file_logs","login_logs","email_logs","device_logs","psychometric_logs"],
        "HR":               ["email_logs","login_logs","psychometric_logs","file_logs"],
        "Finance":          ["file_logs","login_logs","email_logs","device_logs"],
        "Manager":          ["email_logs","login_logs","file_logs","psychometric_logs"],
        "Admin":            ["login_logs","file_logs","device_logs","email_logs"],
        "Security Analyst": ["device_logs","login_logs","file_logs","psychometric_logs"],
    }.get(role, ["login_logs","file_logs","email_logs"])
    if abnormal:
        return random.choice(["file_logs","device_logs","decoy_logs","email_logs"])
    return random.choice(base)

def generate_raw_activity(uid: str, meta: dict, abnormal: bool = False) -> dict:
    source = choose_raw_log_source(meta, abnormal)
    generators = {
        "login_logs":        generate_login_log,
        "email_logs":        generate_email_log,
        "file_logs":         generate_file_log,
        "device_logs":       generate_device_log,
        "decoy_logs":        generate_decoy_log,
        "psychometric_logs": generate_psychometric_log,
    }
    raw = generators.get(source, generate_login_log)(uid, meta, abnormal)
    raw["source_type"] = source
    return append_raw_log(source, raw)

def build_processed_log_entry(uid: str, raw_log: dict, features: list, risk_level: str,
                               ml_score: float, threat_cat: str, shap_values: list,
                               meta: dict | None = None) -> dict:
    if meta is None:
        meta = get_user_metadata(uid)
    event_type = infer_event_type_from_raw(raw_log)
    source_type = raw_log.get("source_type", "")
    source_type_map = {
        "login_logs": "Security (Logon)", "email_logs": "ExchangeServer",
        "file_logs": "Security (Object Access)", "device_logs": "DeviceGuard",
        "decoy_logs": "Honeypot", "psychometric_logs": "HR Analytics",
    }
    source_label = raw_log.get("source") or source_type_map.get(source_type) or source_type or "Security"
    action = raw_log.get("action", "")
    login_status = raw_log.get("login_status", "")
    category_map = {
        "login_logs":        f"Logon ({login_status})" if login_status else "Logon",
        "email_logs":        "Mail Flow",
        "file_logs":         f"File {action.title()}" if action else "Object Access",
        "device_logs":       f"Device {action.title()}" if action else "Plug and Play",
        "decoy_logs":        "Honeypot Access",
        "psychometric_logs": "Behavioural Assessment",
    }
    category_label = raw_log.get("category") or category_map.get(source_type) or event_type.replace("_"," ").title()

    # Attach slow-burn context
    burn_history = list(USER_HISTORY.get(uid, []))
    burn_count   = sum(burn_history)
    window_size  = len(burn_history) or 1
    burn_pct     = burn_count / window_size

    return {
        "log_type": "processed",
        "user_id": uid, "employee_id": uid,
        "timestamp": raw_log.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time_short": (raw_log.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"))[11:],
        "event_id": raw_log.get("event_id", 0),
        "source_type": source_type,
        "source": source_label, "category": category_label,
        "level": raw_log.get("level", "Information"),
        "source_ip": raw_log.get("ip_address", raw_log.get("ip", "N/A")),
        "hostname": raw_log.get("device_name", raw_log.get("hostname", "N/A")),
        "role": raw_log.get("role") or get_user_role(meta),
        "department": raw_log.get("department") or meta.get("department"),
        "team": raw_log.get("team") or meta.get("team"),
        "business_unit": raw_log.get("business_unit") or meta.get("business_unit"),
        "supervisor": raw_log.get("supervisor") or meta.get("supervisor"),
        "risk": risk_level,
        "score": round(float(ml_score), 4),
        "threat_category": threat_cat,
        "features": [round(float(x), 4) for x in features],
        "shap_values": shap_values,
        "feature_labels": FEATURE_LABELS,
        "raw_reference": raw_log,
        "event_type": event_type,
        "locked": uid in LOCKED_ACCOUNTS,
        "actual_label": meta.get("actual_label") or meta.get("anomaly"),
        "burn_count": burn_count,
        "burn_pct": round(burn_pct, 3),
        "burn_window": window_size,
    }

def get_realistic_event_sequence(uid: str) -> str:
    if uid not in EVENT_HISTORY:
        EVENT_HISTORY[uid] = deque(maxlen=10)
    recent     = list(EVENT_HISTORY[uid])
    last_event = recent[-1] if recent else None
    if last_event == "LOGON_SUCCESS":
        return random.choice(["FILE_ACCESS", "EMAIL_EXTERNAL"])
    elif last_event == "FILE_ACCESS":
        return random.choice(["EMAIL_EXTERNAL", "MASS_DOWNLOAD", "USB_CONNECTED"])
    elif last_event == "USB_CONNECTED":
        return "DATA_COPY_USB"
    elif last_event == "MASS_DOWNLOAD":
        return random.choice(["EMAIL_EXTERNAL", "DATA_COPY_USB"])
    return random.choice(EVENT_POOL)

# ══════════════════════════════════════════════════════════════════════════════
#  AUTHENTICATION ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    if session.get("logged_in"):
        return redirect(url_for("page_overview"))
    return redirect(url_for("login_page"))

@app.route("/login")
def login_page():
    return render_template("login.html", error=None)

@app.route("/login", methods=["POST"])
def login_submit():
    global ANALYST_ACCOUNTS
    email    = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    account  = ANALYST_ACCOUNTS.get(email)
    if not account or account["password"] != _hash(password):
        return render_template("login.html", error="Invalid credentials")
    session["logged_in"]    = True
    session["analyst_id"]   = email
    session["analyst_name"] = account["name"]
    session["analyst_role"] = account["role"]
    session["must_change"]  = account.get("must_change", False)
    if account.get("must_change"):
        return redirect(url_for("change_password"))
    return redirect(url_for("page_overview"))

@app.route("/change-password", methods=["GET", "POST"])
def change_password():
    if not session.get("logged_in"):
        return redirect(url_for("login_page"))
    if request.method == "POST":
        new_pw  = request.form.get("new_password", "")
        confirm = request.form.get("confirm_password", "")
        if len(new_pw) < 8:
            return render_template("change_password.html", error="Password must be at least 8 characters")
        if new_pw != confirm:
            return render_template("change_password.html", error="Passwords do not match")
        email = session["analyst_id"]
        ANALYST_ACCOUNTS[email]["password"]    = _hash(new_pw)
        ANALYST_ACCOUNTS[email]["must_change"] = False
        save_analyst_accounts(ANALYST_ACCOUNTS)
        session["must_change"] = False
        return redirect(url_for("page_overview"))
    return render_template("change_password.html", error=None)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-PAGE DASHBOARD ROUTES
# ══════════════════════════════════════════════════════════════════════════════
def require_auth():
    if not session.get("logged_in"):
        return redirect(url_for("login_page"))
    if session.get("must_change"):
        return redirect(url_for("change_password"))
    return None

@app.route("/dashboard")
@app.route("/dashboard/overview")
def page_overview():
    redir = require_auth()
    if redir: return redir
    return render_template("overview.html")

@app.route("/dashboard/investigation")
def page_investigation():
    redir = require_auth()
    if redir: return redir
    return render_template("investigation.html")

@app.route("/dashboard/threats")
def page_threats():
    redir = require_auth()
    if redir: return redir
    return render_template("threats.html")

@app.route("/dashboard/users")
def page_users():
    redir = require_auth()
    if redir: return redir
    return render_template("users.html")

@app.route("/dashboard/admin")
def page_admin():
    redir = require_auth()
    if redir: return redir
    if session.get("analyst_role") != "admin":
        return redirect(url_for("page_overview"))
    return render_template("admin.html")

@app.route("/dashboard/settings")
def page_settings():
    redir = require_auth()
    if redir: return redir
    return render_template("settings.html")

@app.route("/dashboard/upload")
def page_upload():
    redir = require_auth()
    if redir: return redir
    return render_template("upload_logs.html")

@app.route("/dashboard/logs")
def page_logs():
    redir = require_auth()
    if redir: return redir
    return render_template("logs.html")

# ══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/telemetry")
def api_telemetry():
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 401

    global ALERTS, THREAT_STATS

    active_users = sample_users(random.randint(4, 8))
    new_processed = []

    for uid in active_users:
        if uid in RESOLVED:
            continue

        meta             = get_user_metadata(uid)
        is_known_anomaly = meta.get("actual_label") == "anomaly" or meta.get("anomaly") == "anomaly"
        raw_log          = generate_raw_activity(uid, meta, is_known_anomaly)
        inferred_event   = infer_event_type_from_raw(raw_log)
        features         = extract_features(uid, inferred_event)
        risk_level, ml_score, threat_cat = risk_engine(uid, features, inferred_event)
        shap_values      = compute_shap_values(features)

        processed_entry = build_processed_log_entry(uid, raw_log, features, risk_level, ml_score, threat_cat, shap_values, meta)
        new_processed.append(processed_entry)
        EVENT_LOG.appendleft(raw_log)
        PROCESSED_LOGS.appendleft(processed_entry)

        if uid not in EVENT_HISTORY:
            EVENT_HISTORY[uid] = deque(maxlen=10)
        EVENT_HISTORY[uid].append(inferred_event)

        if threat_cat != "Normal Business Activity":
            THREAT_STATS[threat_cat] = THREAT_STATS.get(threat_cat, 0) + 1

        # Only alert on MED/HIGH
        if risk_level in ["HIGH", "MEDIUM"]:
            ALERTS = [a for a in ALERTS if a["user_id"] != uid]
            ALERTS.insert(0, processed_entry)

    priority = {"HIGH": 2, "MEDIUM": 1}
    ALERTS.sort(key=lambda x: priority.get(x["risk"], 0), reverse=True)

    combined = sorted(new_processed, key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify({"logs": combined, "alerts": ALERTS[:20]})

@app.route("/api/stats")
def api_stats():
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 401
    high_count = sum(1 for a in ALERTS if a["risk"] == "HIGH")
    med_count  = sum(1 for a in ALERTS if a["risk"] == "MEDIUM")
    threat_breakdown = dict(THREAT_STATS)
    return jsonify({
        "active_alerts":    len(ALERTS),
        "high_risk":        high_count,
        "medium_risk":      med_count,
        "suspicious_signals": sum(1 for e in EVENT_LOG if e.get("risk") in ["HIGH","MEDIUM"]),
        "open_cases":       len(INVESTIGATION_CASES),
        "events_today":     len(EVENT_LOG),
        "resolved":         len(RESOLVED),
        "threat_breakdown": threat_breakdown,
        "top_threats":      sorted(threat_breakdown.items(), key=lambda x: x[1], reverse=True)[:5],
    })

@app.route("/api/resolve", methods=["POST"])
def api_resolve():
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 401
    data   = request.json or {}
    uid    = data.get("user_id")
    action = data.get("action", "resolve")
    if uid:
        global ALERTS
        if action == "resolve":
            ALERTS = [a for a in ALERTS if a["user_id"] != uid]
            RESOLVED.add(uid)
            return jsonify({"status": "resolved", "message": f"Alert for {uid} resolved"})
        if action == "lock_account":
            LOCKED_ACCOUNTS.add(uid)
            for alert in ALERTS:
                if alert["user_id"] == uid:
                    alert["locked"] = True
            return jsonify({"status": "locked", "message": f"Account {uid} locked"})
        if action == "unlock_account":
            LOCKED_ACCOUNTS.discard(uid)
            for alert in ALERTS:
                if alert["user_id"] == uid:
                    alert["locked"] = False
            return jsonify({"status": "unlocked", "message": f"Account {uid} unlocked"})
    return jsonify({"status": "ok"})

@app.route("/api/user/<uid>")
def api_user_detail(uid):
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 401
    meta    = get_user_metadata(uid)
    history = list(USER_HISTORY.get(uid, []))
    recent_events = [e for e in list(EVENT_LOG)[:100] if e.get("user_id") == uid]
    return jsonify({
        **meta,
        "history":      history,
        "burn_rate":    sum(history),
        "burn_pct":     round(sum(history) / max(1, len(history)), 3),
        "burn_window":  len(history),
        "recent_events": recent_events[:10],
        "locked":       uid in LOCKED_ACCOUNTS,
    })

@app.route("/api/employees")
def api_employees_list():
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 401
    employees = []
    if not df_employees.empty:
        for uid in df_employees["user_id"].tolist()[:100]:
            employees.append(get_user_metadata(uid))
    return jsonify({"employees": employees})

@app.route("/api/user/warn", methods=["POST"])
def api_user_warn():
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 401
    data    = request.json or {}
    uid     = data.get("user_id")
    message = data.get("message", "Please stop the behavior or your account may be locked.")
    if not uid:
        return jsonify({"error": "user_id required"}), 400
    meta = get_user_metadata(uid)
    warning_event = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event_id": 99999, "source": "Sentinel Mailer", "category": "User Warning",
        "level": "Information", "source_ip": "N/A", "hostname": "SENTINEL-ADMIN",
        "user_id": uid, "email": meta.get("email",""),
        "role": meta.get("role","Employee"), "department": meta.get("department","Unknown"),
        "team": meta.get("team","General"), "business_unit": meta.get("business_unit","Corporate"),
        "supervisor": meta.get("supervisor","Unknown"),
        "risk": "LOW", "score": 0.0, "threat_category": "Manager Warning",
        "features": [], "shap_values": [], "feature_labels": FEATURE_LABELS,
        "burn_rate": sum(USER_HISTORY.get(uid, [])),
        "event_type": "MANAGER_WARN", "warning_message": message,
    }
    EVENT_LOG.appendleft(warning_event)
    return jsonify({"status": "ok", "message": "Warning sent."})

@app.route("/api/case/create", methods=["POST"])
def api_case_create():
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 401
    data    = request.json or {}
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    meta    = get_user_metadata(user_id)
    case_id = str(uuid.uuid4())[:8]
    notes   = data.get("notes", [])
    if not isinstance(notes, list):
        notes = [notes] if notes else []
    events  = data.get("events", [])
    if not isinstance(events, list):
        events = [events] if events else []
    INVESTIGATION_CASES[case_id] = {
        "case_id": case_id, "user_id": user_id,
        "user_email":   meta.get("email",""), "department":  meta.get("department",""),
        "supervisor":   meta.get("supervisor",""), "team": meta.get("team",""),
        "business_unit": meta.get("business_unit",""), "analyst": session["analyst_name"],
        "created": datetime.now().isoformat(), "status": "Open",
        "severity": data.get("severity","Medium"),
        "reason":   data.get("reason","Manual investigation from alert"),
        "notes": notes, "events": events,
    }
    save_cases(INVESTIGATION_CASES)
    return jsonify({"status": "ok", "case_id": case_id})

@app.route("/api/case/<case_id>/note", methods=["POST"])
def api_case_add_note(case_id):
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 401
    note = (request.json or {}).get("note","")
    if case_id in INVESTIGATION_CASES:
        notes = INVESTIGATION_CASES[case_id].get("notes", [])
        if not isinstance(notes, list):
            notes = []
            INVESTIGATION_CASES[case_id]["notes"] = notes
        notes.append({"analyst": session["analyst_name"], "timestamp": datetime.now().isoformat(), "text": note})
        save_cases(INVESTIGATION_CASES)
        return jsonify({"status": "ok"})
    return jsonify({"error": "case not found"}), 404

@app.route("/api/case/<case_id>/status", methods=["POST"])
def api_case_update_status(case_id):
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 401
    status = (request.json or {}).get("status")
    if case_id in INVESTIGATION_CASES and status:
        INVESTIGATION_CASES[case_id]["status"]       = status
        INVESTIGATION_CASES[case_id]["last_updated"] = datetime.now().isoformat()
        save_cases(INVESTIGATION_CASES)
        return jsonify({"status": "ok"})
    return jsonify({"error": "invalid request"}), 400

@app.route("/api/cases")
def api_cases_list():
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"cases": list(INVESTIGATION_CASES.values())})

@app.route("/api/admin/users")
def api_admin_list_users():
    if not session.get("logged_in") or session.get("analyst_role") != "admin":
        return jsonify({"error": "forbidden"}), 403
    users = [{"email": k, "name": v["name"], "role": v["role"]} for k, v in ANALYST_ACCOUNTS.items()]
    return jsonify({"users": users})

@app.route("/api/admin/user/add", methods=["POST"])
def api_admin_add_user():
    if not session.get("logged_in") or session.get("analyst_role") != "admin":
        return jsonify({"error": "forbidden"}), 403
    data   = request.json or {}
    email  = data.get("email","").strip().lower()
    name   = data.get("name","New User")
    role   = data.get("role","analyst")
    temp_pw = data.get("temp_password","TempPass2024!")
    if not email:
        return jsonify({"error": "Email required"}), 400
    ANALYST_ACCOUNTS[email] = {"password": _hash(temp_pw), "name": name, "role": role, "must_change": True}
    save_analyst_accounts(ANALYST_ACCOUNTS)
    return jsonify({"status": "ok", "email": email, "temp_password": temp_pw})

@app.route("/api/admin/user/disable", methods=["POST"])
def api_admin_disable_user():
    if not session.get("logged_in") or session.get("analyst_role") != "admin":
        return jsonify({"error": "forbidden"}), 403
    email = (request.json or {}).get("email")
    if email and email in ANALYST_ACCOUNTS:
        del ANALYST_ACCOUNTS[email]
        save_analyst_accounts(ANALYST_ACCOUNTS)
        return jsonify({"status": "ok"})
    return jsonify({"error": "user not found"}), 404

@app.route("/api/logs/list")
def api_logs_list():
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 401
    log_type      = request.args.get("type","processed")
    source_filter = request.args.get("source")
    start_date    = request.args.get("start_date")
    end_date      = request.args.get("end_date")
    query         = request.args.get("q","")
    if log_type == "raw":
        for src, path in RAW_LOG_FILES.items():
            if (not RAW_LOG_STORE.get(src)) and os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    RAW_LOG_STORE[src] = df.to_dict("records")
                except Exception:
                    RAW_LOG_STORE[src] = []
        raw_logs = {
            src: [r for r in _filter_records_by_time(rows, start_date, end_date) if _matches_query(r, query)]
            for src, rows in RAW_LOG_STORE.items()
        }
        if source_filter and source_filter in raw_logs:
            raw_logs = {source_filter: raw_logs[source_filter]}
        log_count = sum(len(v) for v in raw_logs.values())
        return jsonify({"timestamp": datetime.now().isoformat(), "count": log_count,
                        "type": log_type, "source": source_filter,
                        "start_date": start_date, "end_date": end_date, "logs": raw_logs})
    else:
        if len(PROCESSED_LOGS) == 0:
            api_telemetry()
        processed = _filter_records_by_time(list(PROCESSED_LOGS), start_date, end_date)
        if query:
            processed = [r for r in processed if _matches_query(r, query)]
        return jsonify({"timestamp": datetime.now().isoformat(), "count": len(processed),
                        "type": log_type, "source": source_filter,
                        "start_date": start_date, "end_date": end_date, "logs": processed})

@app.route("/api/logs/download")
def api_logs_download():
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 401
    return api_logs_list()

@app.route("/api/settings")
def api_settings_get():
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 401
    return jsonify(CURRENT_SETTINGS)

@app.route("/api/settings", methods=["POST"])
def api_settings_update():
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 401
    data = request.json or {}
    if "auto_refresh" in data:
        try:
            CURRENT_SETTINGS["auto_refresh"] = max(5, min(120, int(data["auto_refresh"])))
        except Exception:
            pass
    if "model_sensitivity" in data and data["model_sensitivity"] in {"Low","Medium","High"}:
        CURRENT_SETTINGS["model_sensitivity"] = data["model_sensitivity"]
    if "alert_threshold" in data and data["alert_threshold"] in {"Low","Medium","High"}:
        CURRENT_SETTINGS["alert_threshold"] = data["alert_threshold"]
    if "warning_template" in data:
        CURRENT_SETTINGS["warning_template"] = data["warning_template"]
    return jsonify({"status": "ok", "settings": CURRENT_SETTINGS})

@app.route("/api/logs/clear", methods=["POST"])
def api_logs_clear():
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 401
    global EVENT_LOG, ALERTS
    EVENT_LOG.clear()
    PROCESSED_LOGS.clear()
    for k in RAW_LOG_STORE.keys():
        RAW_LOG_STORE[k].clear()
    ALERTS.clear()
    return jsonify({"status": "ok", "message": "Logs cleared"})

@app.route("/api/logs/upload", methods=["POST"])
def api_logs_upload():
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 401
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    try:
        if file.filename.endswith(".json"):
            data = json.loads(file.read())
            logs = data.get("logs",[]) if isinstance(data, dict) else data
        elif file.filename.endswith(".csv"):
            import io
            df   = pd.read_csv(io.StringIO(file.read().decode("utf-8")))
            logs = df.to_dict("records")
        else:
            return jsonify({"error": "Unsupported file type. Use JSON or CSV."}), 400
        if not isinstance(logs, list):
            return jsonify({"error": "Invalid log format"}), 400
        analyzed = []
        for raw in logs[:100]:
            uid    = raw.get("user_id") or raw.get("employee_id") or f"USR-{random.randint(1000,9999)}"
            source = infer_raw_log_source(raw) or raw.get("source_type") or "login_logs"
            raw_record = {**raw, "user_id": uid, "employee_id": raw.get("employee_id", uid)}
            RAW_LOG_STORE.setdefault(source, []).append(raw_record)
            features      = engineer_features_from_raw_logs(uid) or extract_features_from_baseline(uid)
            inferred_event = infer_event_type_from_raw(raw_record)
            risk_level, ml_score, threat_cat = risk_engine(uid, features, inferred_event)
            shap_values   = compute_shap_values(features)
            meta          = get_user_metadata(uid)
            processed     = build_processed_log_entry(uid, raw_record, features, risk_level, ml_score, threat_cat, shap_values, meta)
            PROCESSED_LOGS.appendleft(processed)
            EVENT_LOG.appendleft(raw_record)
            analyzed.append(processed)
        case_id = request.form.get("case_id") or None
        if case_id and case_id in INVESTIGATION_CASES:
            INVESTIGATION_CASES[case_id].setdefault("events", []).extend(analyzed)
            save_cases(INVESTIGATION_CASES)
        return jsonify({"status": "ok", "count": len(analyzed), "results": analyzed, "case_id": case_id})
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)