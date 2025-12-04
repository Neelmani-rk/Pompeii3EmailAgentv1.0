import os
import re 
from dotenv import load_dotenv
load_dotenv()

# --- Gmail API Configuration ---
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/gmail.labels',
    'https://www.googleapis.com/auth/gmail.compose'
]

SERVICE_ACCOUNT_FILE = os.environ.get("SERVICE_ACCOUNT_FILE", "service-account.json")  # For Google Cloud services
TOKEN_FILE = "TOKEN.json"                      # For Gmail OAuth token
CREDENTIALS_FILE = "CREDENTIAL.json"           # Potentially for other credentials if used

# --- API Credentials (from Environment Variables) ---
# ShipStation
SHIPSTATION_API_KEY = os.environ.get("SHIPSTATION_API_KEY")
SHIPSTATION_API_SECRET = os.environ.get("SHIPSTATION_API_SECRET")

# FedEx
FEDEX_CLIENT_ID = os.environ.get("FEDEX_CLIENT_ID")
FEDEX_CLIENT_SECRET = os.environ.get("FEDEX_CLIENT_SECRET")

# Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- Google Cloud Storage Configuration ---
BUCKET_NAME = "dotted-electron-447414-m1-history-ids" # For email history, LLM payloads, etc.
RESPONSES_BUCKET_NAME = "dotted-electron-447414-m1-responses" # For storing generated responses
VECTOR_DB_BUCKET = "dotted-electron-447414-m1-vector-db" # For storing the vector database
VECTOR_DB_PREFIX = "vector_db_export/" # Prefix for vector DB files in GCS

# --- Local File/Directory Names ---
FILE_NAME = "email_history.json"  # Name for email history file (if used locally)
CHROMA_DIR = "chroma_db"          # Local directory for ChromaDB if persisted locally
LOCAL_DB_DIR = "/tmp/chroma_db"   # Local directory in Cloud Function's /tmp space for GCS copy
SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID", "17VTMM7zhM_qSklirkPZxLcp5sTWJn0b7W07WSODhlIk")

# --- Application Behavior Configuration ---
RATE_PER_MINUTE = int(os.environ.get("RATE_PER_MINUTE", "12")) # Email processing rate limit

HISTORY_ID_BUCKET = "dotted-electron-447414-m1-history-ids"
HISTORY_ID_OBJECT = "last_history_id.json"

# --- Email Quality Validation Configuration ---
ENABLE_QUALITY_VALIDATION = os.environ.get("ENABLE_QUALITY_VALIDATION", "true").lower() == "true"
ENABLE_ANALYTICS = os.environ.get("ENABLE_ANALYTICS", "true").lower() == "true"
MIN_QUALITY_THRESHOLD = float(os.environ.get("MIN_QUALITY_THRESHOLD", "0.5"))
MIN_HELPFULNESS_THRESHOLD = float(os.environ.get("MIN_HELPFULNESS_THRESHOLD", "0.6"))

# --- Logging Configuration ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
ENABLE_DEBUG_LOGGING = os.environ.get("ENABLE_DEBUG_LOGGING", "false").lower() == "true"

# --- Performance Configuration ---
MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "5"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "30"))
RETRY_ATTEMPTS = int(os.environ.get("RETRY_ATTEMPTS", "3"))

# --- Security Configuration ---
ENABLE_INPUT_VALIDATION = os.environ.get("ENABLE_INPUT_VALIDATION", "true").lower() == "true"
SANITIZE_API_RESPONSES = os.environ.get("SANITIZE_API_RESPONSES", "true").lower() == "true"


