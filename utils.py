import logging
import re
from google.oauth2 import service_account
import gspread


from config import (
    SHIPSTATION_API_KEY, SHIPSTATION_API_SECRET,
    FEDEX_CLIENT_ID, FEDEX_CLIENT_SECRET,
    GEMINI_API_KEY,
    SERVICE_ACCOUNT_FILE, SPREADSHEET_ID, 
)


def setup_logging():
    """
    Configure logging for the application.
    Sets up a logger named "customer_service" with a console handler.
    """
    logger = logging.getLogger("customer_service")
    if logger.handlers:
        logger.handlers = []  # Clear existing handlers to avoid duplicates
    logger.setLevel(logging.DEBUG) # Set overall logger level

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) # Set console handler level (can be different)

    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    
    logger.info("Logging initialized in utils.setup_logging")
    return logger

def clean_extracted_text(text: str) -> str:
    """Clean text extracted from images by removing CSS styling and HTML-like content."""
    if not text:
        return ""
    # Remove CSS style blocks
    text = re.sub(r'\s*\.\w+\s*\{[^}]*\}', ' ', text)
    # Remove HTML-like attributes
    text = re.sub(r'\s*\w+-\w+\s*:\s*[^;]+;', ' ', text)
    # Remove other styling indicators
    text = re.sub(r'width:\s*\d+%', ' ', text)
    text = re.sub(r'height:\s*\d+px', ' ', text)
    text = re.sub(r'display:\s*flex', ' ', text)
    text = re.sub(r'flex-direction:\s*\w+', ' ', text)
    text = re.sub(r'justify-content:\s*\w+', ' ', text)
    text = re.sub(r'align-items:\s*\w+', ' ', text)
    text = re.sub(r'text-align:\s*\w+', ' ', text)
    text = re.sub(r'padding:\s*\d+px', ' ', text)
    text = re.sub(r'border:\s*\d+px\s+\w+\s+\w+', ' ', text)
    text = re.sub(r'border-collapse:\s*\w+', ' ', text)
    # Remove excessive whitespace (multiple spaces, newlines, tabs)
    text = re.sub(r'\s+', ' ', text)
    # Trim leading/trailing whitespace
    text = text.strip()
    return text

def extract_email_address(raw_email: str) -> str:
    """
    Extracts the email address from a string that might include a name
    (e.g., "John Doe <john@example.com>" -> "john@example.com").
    """
    if not raw_email:
        return ""
    # Use email.utils.parseaddr for robust parsing
    # from email.utils import parseaddr # Consider moving to top if widely used
    # name, email = parseaddr(raw_email)
    # if email:
    #     return email
    
    # Original regex-like approach if parseaddr is not preferred here
    match = re.search(r'<([^>]+)>', raw_email)
    if match:
        return match.group(1).strip()
    return raw_email.strip() # Fallback if no <...> format

def extract_sender_name(sender_string: str) -> str:
    """Extract the name part from a sender string formatted as 'Name <email>'."""
    if not sender_string:
        return ""
   
    
    # Original regex-like approach
    match = re.match(r'([^<]+)<.*>', sender_string)
    if match:
        return match.group(1).strip()
    return sender_string.strip() # Fallback if no <...> format

def separate_email_parts(email_text: str) -> tuple[str, str]:
    """
    Separates the current message from the previous conversation in an email body.
    Returns a tuple (current_message, previous_conversation).
    """
    # Return default values if input is None or too short
    if not email_text or len(email_text) < 10:
        return email_text or "", ""

    # Clean the email of HTML/CSS and other formatting issues first
    # Remove HTML/CSS content (simple version, might need more robust HTML parsing for complex emails)
    email_text_cleaned = re.sub(r'#yiv\d+\s+[^{]*{[^}]*}', ' ', email_text) # Remove specific Outlook/Yahoo styles
    email_text_cleaned = re.sub(r'<style[^>]*>.*?</style>', ' ', email_text_cleaned, flags=re.DOTALL | re.IGNORECASE) # Remove style blocks
    email_text_cleaned = re.sub(r'<[^>]+>', ' ', email_text_cleaned) # Remove all HTML tags

    # Normalize line breaks and whitespace
    email_text_cleaned = email_text_cleaned.replace('\r\n', '\n').replace('\r', '\n')
    email_text_cleaned = re.sub(r'[ \t]+', ' ', email_text_cleaned) # Replace multiple spaces/tabs with single space
    email_text_cleaned = re.sub(r'\n\s*\n', '\n\n', email_text_cleaned) # Normalize multiple newlines to double newlines

    # Define patterns to identify forwarded content and quoted replies
    # Patterns are ordered from more specific/reliable to more general
    separator_patterns = [
        re.compile(r'^-{5,}\s*Forwarded message\s*-{5,}$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^\s*_{30,}\s*$', re.MULTILINE), # Horizontal line often used as separator
        re.compile(r'^\s*From:\s*.*$', re.IGNORECASE | re.MULTILINE), # "From: " line often starts a forwarded section
        re.compile(r'^\s*Sent:\s*.*$', re.IGNORECASE | re.MULTILINE), # "Sent: " line
        re.compile(r'^\s*Date:\s*.*$', re.IGNORECASE | re.MULTILINE), # "Date: " line (less reliable on its own)
        re.compile(r'^\s*Subject:\s*.*$', re.IGNORECASE | re.MULTILINE), # "Subject: " line
        re.compile(r'^\s*To:\s*.*$', re.IGNORECASE | re.MULTILINE), # "To: " line
        re.compile(r'On\s+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\w+\s+\d{1,2},?\s+\d{4}\s+at\s+\d{1,2}:\d{2}\s+(?:AM|PM)?(?:,\s*)?.*wrote:$', re.IGNORECASE), # "On Tue, May 27, 2025 at 10:00 AM, John Doe <john@example.com> wrote:"
        re.compile(r'On\s+\w+\s+\d{1,2},?\s+\d{4}(?:\s+at\s+\d{1,2}:\d{2}\s*(?:AM|PM)?)?,\s*.*wrote:$', re.IGNORECASE), # More general date format
        re.compile(r'.*\bwrote:\s*$', re.MULTILINE), # Lines ending with "wrote:" (generic)
        re.compile(r'Begin forwarded message:', re.IGNORECASE),
        re.compile(r'Sent from my .*', re.IGNORECASE), # "Sent from my iPhone/Android"
        re.compile(r'-{2,}\s*Original Message\s*-{2,}', re.IGNORECASE),
        
        # Walmart-specific separator patterns from your original code
        re.compile(r'Customer Message:', re.IGNORECASE),
        re.compile(r'What you need to do now:', re.IGNORECASE),
        re.compile(r'Thank you,\s+Your Walmart Customer Care Team', re.IGNORECASE),
        re.compile(r'For this type of email content', re.IGNORECASE),
        re.compile(r'Reason for Contact:', re.IGNORECASE)
    ]

    separator_pos = -1

    # Find the earliest separator
    for pattern in separator_patterns:
        match = pattern.search(email_text_cleaned)
        if match:
            if separator_pos == -1 or match.start() < separator_pos:
                separator_pos = match.start()

    # Also check for email quote markers (lines starting with >)
    # We want the start of the *block* of quoted text
    quote_block_match = re.search(r'(?:\n|^)>\s', email_text_cleaned) # Look for newline or start of string followed by "> "
    if quote_block_match:
        quote_start_pos = quote_block_match.start()
        if separator_pos == -1 or quote_start_pos < separator_pos:
            separator_pos = quote_start_pos
    
    current_message = email_text_cleaned
    previous_conversation = ""

    if separator_pos != -1:
        current_message = email_text_cleaned[:separator_pos].strip()
        previous_conversation = email_text_cleaned[separator_pos:].strip()

    # Final cleanup of current message: remove residual common reply headers if they are at the very beginning
    # This is a bit aggressive, ensure patterns are specific enough
    leading_header_patterns = [
        re.compile(r'^\s*On\s+.*wrote:\s*', re.IGNORECASE),
        # Add other very specific leading headers if needed
    ]
    for pat in leading_header_patterns:
        current_message = pat.sub('', current_message).strip()

    # Clean up previous conversation: remove quote markers and excess newlines
    if previous_conversation:
        previous_conversation = re.sub(r'^>\s?', '', previous_conversation, flags=re.MULTILINE) # Remove leading "> "
        previous_conversation = re.sub(r'\n\s*\n', '\n\n', previous_conversation).strip() # Normalize multiple newlines

    # Remove invisible formatting characters and normalize whitespace again after splitting
    current_message = re.sub(r'[\u034f\u200b\u200c\u200d\u2060\ufeff\u00ad]', '', current_message)
    current_message = re.sub(r'\s+', ' ', current_message).strip()
    
    if previous_conversation:
        previous_conversation = re.sub(r'[\u034f\u200b\u200c\u200d\u2060\ufeff\u00ad]', '', previous_conversation)
        previous_conversation = re.sub(r'\s+', ' ', previous_conversation).strip()

    return current_message, previous_conversation


def extract_order_number_from_subject(subject: str, logger: logging.Logger = None) -> list[str]:
    """Extract order numbers (including 15-digit POs) from the subject line."""
    if not subject:
        return []
        
    order_numbers = []
    # Combined patterns for regular order numbers (4-12 digits) and POs (15 digits)
    # Prioritize more specific patterns
    patterns = [
        r'order\s*number\s*#?\s*(\d{4,15})',      # "order number #12345" or "order number #PO123..."
        r'order\s*id\s*#?\s*(\d{4,15})',          # "order id #12345"
        r'pompeii3\s*order\s*#?\s*(\d{4,15})',    # "pompeii3 order #12345"
        r're:\s*order\s*#?\s*(\d{4,15})',         # "re: order #12345"
        r'regarding\s*order\s*#?\s*(\d{4,15})',   # "regarding order #12345"
        r'order\s*#?\s*(\d{4,15})',               # "order #12345" (most general for order keyword)
        
        r'purchase\s*order\s*#?\s*(\d{15})',      # "purchase order #PO123..." (specific for 15-digit POs)
        r'po\s*number\s*#?\s*(\d{15})',           # "PO number #PO123..."
        r'po\s*#?\s*(\d{15})',                    # "PO #PO123..." (most general for PO keyword)
        r're:\s*po\s*#?\s*(\d{15})',              # "re: PO #PO123..."
        r'regarding\s*po\s*#?\s*(\d{15})',        # "regarding PO #PO123..."

        r'confirmation\s*#?\s*(\d{4,15})',       # "confirmation #12345"
        r'reference\s*#?\s*(\d{4,15})',          # "reference #12345"
        r'ref\s*#?\s*(\d{4,15})',                 # "ref #12345"
        r'inquiry\s*about\s*(?:order|po)\s*#?\s*(\d{4,15})',
        r'tracking\s+for\s+(?:order|po)?\s*#?\s*(\d{4,15})',
        r'status\s+(?:of|for)\s+(?:order|po)?\s*#?\s*(\d{4,15})',
        r'update\s+(?:on|for)\s+(?:order|po)?\s*#?\s*(\d{4,15})',
        r'#\s*(\d{4,15})\b',                      # Just # followed by digits (word boundary)
        r'\b(\d{4,15})\b'                         # Last resort: any 4-15 digit number if others fail
                                                 
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, subject, re.IGNORECASE)
        for match in matches:
            order_num_str = match.group(1).strip()
            if order_num_str.isdigit():
                num_len = len(order_num_str)
                # Validate length for typical orders or 15-digit POs
                if (4 <= num_len <= 12) or (num_len == 15):
                    order_numbers.append(order_num_str)
                    if logger:
                        type_label = "Purchase Order" if num_len == 15 else "Order Number"
                        logger.info(f"Found {type_label} '{order_num_str}' in subject using pattern: {pattern}")
    
    # Remove duplicates while preserving order of first appearance
    seen = set()
    unique_order_numbers = [x for x in order_numbers if not (x in seen or seen.add(x))]
    
    if logger and unique_order_numbers:
        logger.info(f"Unique order numbers extracted from subject: {unique_order_numbers}")
    elif logger:
        logger.info("No order numbers extracted from subject.")
        
    return unique_order_numbers



def initialize_google_sheets_client(logger):
    """Initializes Google Sheets client and fetches the first sheet dynamically."""
    try:
        logger.info("✅ Initializing Google Sheets client...")
        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
        client = gspread.authorize(creds)
        # Open spreadsheet using its ID
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
        # Get the first sheet dynamically
        sheet = spreadsheet.get_worksheet(0)  # Index 0 = first sheet
        if sheet:
            logger.info(f"📂 Accessing Google Sheet ID: {SPREADSHEET_ID}, Auto-Selected Sheet: {sheet.title}")
            return sheet
        else:
            logger.error(f"❌ No worksheets found in Sheet ID '{SPREADSHEET_ID}'.")
            raise ValueError("No worksheets available.")
    except Exception as e:
        logger.error(f"❌ Error initializing Google Sheets: {e}")
        raise

def detect_automated_mail_spreadsheet(logger):
    """Fetches and parses the automated email data from the first available Google Sheet."""
    logger.debug("Attempting to fetch automated email data from Google Sheet...")
    try:
        # Initialize Google Sheets client and get first sheet
        sheet = initialize_google_sheets_client(logger)
        # Fetch all records
        data = sheet.get_all_records()  # Use get_all_records for structured data
        logger.debug(f"✅ Successfully fetched {len(data)} rows from Sheet.")
    except Exception as e:
        logger.error(f"❌ Error fetching automated email data: {str(e)}")
        raise  # Re-raise the error to handle it in calling functions
    domain_email_map = {}
    # **Fix for inconsistent column names: Normalize headers**
    if data:
        raw_headers = list(data[0].keys())  # Get first row headers
        normalized_headers = {header.strip().lower(): header for header in raw_headers}
        domain_header = normalized_headers.get("domain name")
        emails_header = normalized_headers.get("email address")
        if not domain_header or not emails_header:
            logger.error("❌ Column headers not found in Google Sheet. Check sheet format.")
            raise ValueError("Missing expected column headers in sheet.")
    for row in data:
        domain_raw = row.get(domain_header)
        emails_raw = row.get(emails_header)
        if domain_raw is None or emails_raw is None:
            logger.warning(f"⚠️ Skipping row due to missing '{domain_header}' or '{emails_header}': {row}")
            continue
        # Extract and clean domain names
        domain = str(domain_raw).strip().lstrip("@").lower()
        emails = [email.strip().lower() for email in str(emails_raw).split(",") if email.strip()]
        if domain and emails:
            if domain not in domain_email_map:
                domain_email_map[domain] = []
            domain_email_map[domain].extend(emails)
            domain_email_map[domain] = list(set(domain_email_map[domain]))  # Remove duplicates
    logger.info(f"✅ Parsed {len(domain_email_map)} unique domains from {len(data)} rows.")
    return domain_email_map

def extract_email(raw_email):
    """Extracts the actual email address from a string containing a name + email in brackets."""
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', raw_email)
    return match.group(0).lower().strip() if match else None

def is_automated_reply(raw_email, logger):
    """Checks if an email address is likely automated using Google Sheets and regex patterns."""
    try:
        extracted_email = extract_email(raw_email)
        if not extracted_email:
            logger.warning(f"⚠️ Unable to extract valid email from '{raw_email}'. Assuming not automated.")
            return False
        email_username, email_domain = extracted_email.split("@")
        logger.info(f"🔍 Checking if '{extracted_email}' (user: '{email_username}', domain: '{email_domain}') is automated.")
        # Fetch latest data from Google Sheets
        try:
            domain_email_map = detect_automated_mail_spreadsheet(logger)  # Fetch Google Sheets data
        except Exception as e:
            logger.error(f"❌ Failed to fetch/parse automated email data: {e}. Proceeding without Sheet data.")
            domain_email_map = {}
        # Step 1: Check if domain exists in the Google Sheet
        if email_domain in domain_email_map:
            logger.debug(f"✅ Domain '{email_domain}' found in Google Sheets. Checking associated emails.")
            registered_emails = domain_email_map[email_domain]
            # Ensure registered_emails is a list
            if isinstance(registered_emails, str):
                registered_emails = [e.strip() for e in registered_emails.split(",")]
            if extracted_email in registered_emails:
                logger.info(f"✅ MATCH: Email '{extracted_email}' found in Google Sheet. Marked as automated.")
                return True
        # Step 2: Check regex patterns for common automated usernames
        automated_usernames = [
            r'^(mailer-daemon|postmaster|bounce|autorespond|automatic|notification|notifications|notify|noreply|no-reply|'
            r'donotreply|do-not-reply|auto|alert|alerts|info|system|admin|administrator|service|support|help|'
            r'customerservice|customercare|billing|account|accounts|marketing|sales|newsletter|confirm|confirmation|'
            r'verify|verification|security|contact)(\d+)?$'
        ]
        if any(re.match(pattern, email_username, re.IGNORECASE) for pattern in automated_usernames):
            logger.info(f"✅ MATCH: Email username '{email_username}' is a common automated pattern. Marked as automated.")
            return True
        logger.info(f"❌ NO MATCH: Email '{extracted_email}' did not match Google Sheets or common automated usernames. Assuming external customer.")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error in is_automated_reply for '{raw_email}': {e}")
        logger.warning("⚠️ Defaulting to NOT automated due to unexpected error.")
        return False