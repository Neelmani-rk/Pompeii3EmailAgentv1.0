import base64
import json
import logging
import os
import re
import time # Used in get_gmail_service for potential token refresh delays
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any, Optional, Set

# Google API related imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow # For initial auth if token is invalid
from google.auth.transport.requests import Request
from googleapiclient.discovery import build, Resource # type: ignore
from google.cloud import storage

# Imports from other project modules
from config import (
    SCOPES, TOKEN_FILE, BUCKET_NAME, # BUCKET_NAME is used as default for token & processed logs
    SERVICE_ACCOUNT_FILE,GEMINI_API_KEY # For storage client if needed directly here, though usually passed
)
# For functions within this module that call entity extraction or API clients:
from entity_extractor import ExtractedEntity, EntityExtractor, extract_entities_from_image_text,extract_text_from_image_with_gemini, extract_entities_from_image_bytes
from utils import separate_email_parts
# --- Gmail Service Initialization ---
def get_gmail_service(logger: Optional[logging.Logger] = None) -> Optional[Resource]:

    log = logger or logging.getLogger(__name__)
    log.info("Initializing Gmail service...")
    creds: Optional[Credentials] = None
    token_bucket_name = os.environ.get("TOKEN_BUCKET", BUCKET_NAME) # Use general BUCKET_NAME as default
    token_blob_name = os.environ.get("TOKEN_BLOB", "gmail_token.json") # Standardized token file name

    # Try loading token from GCS first if in a GCP environment
    if os.environ.get("GCP_PROJECT") or os.environ.get("FUNCTION_TARGET"): # Check common GCP env vars
        try:
            storage_client = storage.Client() # Uses ambient credentials in GCP
            bucket = storage_client.bucket(token_bucket_name)
            token_blob = bucket.blob(token_blob_name)
            if token_blob.exists():
                log.info(f"Loading token from Cloud Storage: gs://{token_bucket_name}/{token_blob_name}")
                token_content = token_blob.download_as_text()
                creds = Credentials.from_authorized_user_info(json.loads(token_content), SCOPES)
            else:
                log.info(f"Token file not found in GCS: gs://{token_bucket_name}/{token_blob_name}")
        except Exception as e:
            log.warning(f"Failed to load token from Cloud Storage (gs://{token_bucket_name}/{token_blob_name}): {e}", exc_info=True)

    # Fall back to local file if GCS loading failed or not in GCP env
    if not creds and os.path.exists(TOKEN_FILE):
        log.info(f"Loading token from local file: {TOKEN_FILE}")
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception as e:
            log.error(f"Failed to load token from local file {TOKEN_FILE}: {e}", exc_info=True)

    # Refresh or initiate flow if credentials are not valid
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            log.info("Refreshing expired Gmail token.")
            try:
                creds.refresh(Request())
            except Exception as e:
                log.error(f"Failed to refresh Gmail token: {e}", exc_info=True)

                return None
        else:

            log.warning("No valid credentials. THIS USUALLY REQUIRES MANUAL AUTHENTICATION (run locally first).")

            log.error("Cannot perform InstalledAppFlow in a non-interactive environment like Cloud Functions if token is missing/invalid and not refreshable.")
            return None # Cannot proceed without valid creds

        # Save the (potentially refreshed) credentials
        try:
            token_json_str = creds.to_json()
            # Save to GCS
            if os.environ.get("GCP_PROJECT") or os.environ.get("FUNCTION_TARGET"):
                try:
                    storage_client = storage.Client()
                    bucket = storage_client.bucket(token_bucket_name)
                    token_blob = bucket.blob(token_blob_name)
                    token_blob.upload_from_string(token_json_str)
                    log.info(f"Saved refreshed token to Cloud Storage: gs://{token_bucket_name}/{token_blob_name}")
                except Exception as e:
                    log.warning(f"Failed to save refreshed token to Cloud Storage: {e}", exc_info=True)
            # Save locally (e.g., for local development or if GCS fails)
            with open(TOKEN_FILE, "w") as token_f:
                token_f.write(token_json_str)
            log.info(f"Saved/refreshed token to local file: {TOKEN_FILE}")

        except Exception as e:
            log.warning(f"Error saving token after refresh: {e}", exc_info=True)
    
    if creds and creds.valid:
        try:
            service = build('gmail', 'v1', credentials=creds, cache_discovery=False) # Disable cache in serverless
            log.info("Gmail service initialized successfully.")
            return service
        except Exception as e:
            log.error(f"Failed to build Gmail service: {e}", exc_info=True)
    
    log.error("Could not obtain valid Gmail credentials.")
    return None


# --- Email Content Extraction ---
def extract_email_body(message: Dict[str, Any], logger: Optional[logging.Logger] = None) -> str:

    log = logger or logging.getLogger(__name__)
    body_str = ""
    
    if 'payload' not in message:
        log.warning("No payload in message to extract body from.")
        return body_str

    payload = message['payload']
    
    # Recursive function to find the best text part
    def find_text_parts(parts: List[Dict[str, Any]], current_best_text: str = "", preferred_type: str = 'text/plain') -> str:
        text_content = current_best_text
        found_preferred = False # Flag to check if preferred type was found at current level
        
        for part in parts:
            mime_type = part.get('mimeType', '').lower()
            if mime_type == preferred_type:
                if 'body' in part and 'data' in part['body']:
                    try:
                        part_data = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='replace')
                        # If preferred is text/plain, we take it and might stop unless it's short
                        if preferred_type == 'text/plain':
                            return part_data # Prioritize first found plain text
                        text_content = part_data # For html, store it but continue searching for plain in this part list
                        found_preferred = True
                    except Exception as e:
                        log.error(f"Error decoding part data for MIME type {mime_type}: {e}")
                # If preferred is found, no need to look for it in sub-parts of *this current part list* for *this specific type*
                # but we might still prefer a text/plain from a parallel part over an HTML from current.
                # This logic is tricky with nested multiparts.
                # A simpler rule: if we find plain text, we usually prefer it.

            elif mime_type == 'text/html' and not text_content and preferred_type != 'text/html': # Only if no plain text found yet
                if 'body' in part and 'data' in part['body']:
                    try:
                        html_data = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='replace')
                        # Basic HTML to text conversion (more robust library like beautifulsoup4 might be better)
                        text_from_html = re.sub(r'<style[^>]*>.*?</style>', '', html_data, flags=re.DOTALL | re.IGNORECASE)
                        text_from_html = re.sub(r'<script[^>]*>.*?</script>', '', text_from_html, flags=re.DOTALL | re.IGNORECASE)
                        text_from_html = re.sub(r'<[^>]+>', ' ', text_from_html)
                        text_from_html = re.sub(r'\s+', ' ', text_from_html).strip()
                        text_content = text_from_html # Store it, but plain text is still preferred
                    except Exception as e:
                        log.error(f"Error decoding/parsing HTML part: {e}")
            
            elif mime_type.startswith('multipart/') and 'parts' in part:
                # Recursively search in sub-parts. If a plain text is found deeper, it should propagate up.
                nested_text = find_text_parts(part['parts'], text_content if preferred_type == 'text/html' else "", preferred_type)
                if preferred_type == 'text/plain' and nested_text: # If plain text found in subparts, it's good

                    return nested_text 
                elif nested_text: # For HTML or if no plain text yet
                    text_content = nested_text


        return text_content

    if 'parts' in payload:
        log.debug(f"Processing multipart message with {len(payload['parts'])} parts.")
        # First, try to get 'text/plain'
        body_str = find_text_parts(payload['parts'], preferred_type='text/plain')
        # If no plain text, try to get 'text/html' (converted)
        if not body_str:
            log.debug("No 'text/plain' part found, looking for 'text/html'.")
            body_str = find_text_parts(payload['parts'], preferred_type='text/html')
            if body_str:
                 body_str = re.sub(r'<style[^>]*>.*?</style>', '', body_str, flags=re.DOTALL | re.IGNORECASE)
                 body_str = re.sub(r'<script[^>]*>.*?</script>', '', body_str, flags=re.DOTALL | re.IGNORECASE)
                 body_str = re.sub(r'<[^>]+>', ' ', body_str)
                 body_str = re.sub(r'\s+', ' ', body_str).strip()


    elif 'body' in payload and 'data' in payload['body']:
        log.debug(f"Processing single part message with MIME type: {payload.get('mimeType')}")
        mime_type = payload.get('mimeType', '').lower()
        try:
            body_data = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='replace')
            if mime_type == 'text/plain':
                body_str = body_data
            elif mime_type == 'text/html':
                # Basic HTML to text conversion
                body_str = re.sub(r'<style[^>]*>.*?</style>', '', body_data, flags=re.DOTALL | re.IGNORECASE)
                body_str = re.sub(r'<script[^>]*>.*?</script>', '', body_str, flags=re.DOTALL | re.IGNORECASE)
                body_str = re.sub(r'<[^>]+>', ' ', body_str) # Remove tags
                body_str = re.sub(r'\s+', ' ', body_str).strip() # Normalize whitespace
            else:
                log.warning(f"Single part message is neither text/plain nor text/html: {mime_type}")
        except Exception as e:
            log.error(f"Error decoding single message body part: {e}")

    if body_str:
        preview = body_str[:200].replace("\n", " ")
        log.info(
            f"Extracted email body (length: {len(body_str)}). Preview: '{preview}...'"
        )
    else:
        log.warning("Could not extract email body.")

        
    return body_str.strip()


# --- Image Processing in Emails ---
def process_image_attachments(
    payload: Dict[str, Any],
    message_id: str,
    service: Resource,
    email_data: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    depth: int = 0
):
    """
    Processes image attachments in an email payload using Gemini 3 Pro for OCR and entity extraction.
    """
    log = logger or logging.getLogger(__name__)
    
    if depth > 3:  # Prevent infinite recursion
        log.warning(f"Maximum recursion depth reached for message {message_id}. Stopping image processing.")
        return
    
    image_entities_found_this_call = []
    
    # Case 1: Direct image attachment
    if payload.get('filename') and payload.get('body', {}).get('attachmentId'):
        filename = payload['filename']
        mime_type = payload.get('mimeType', '')
        if mime_type.startswith('image/'):
            try:
                attachment_id = payload['body']['attachmentId']
                attachment = service.users().messages().attachments().get(
                    userId='me', messageId=message_id, id=attachment_id
                ).execute()
                
                # Decode image data
                image_data = base64.urlsafe_b64decode(attachment['data'])
                
                # Extract entities directly from image using Gemini
                image_entities = extract_entities_from_image_bytes(
                    image_data, GEMINI_API_KEY, log
                )
                
                for entity in image_entities:
                    image_entities_found_this_call.append({
                        'type': entity.type,
                        'value': entity.value,
                        'confidence': entity.confidence,
                        'source': f'attachment:{filename}'
                    })
                
                log.info(f"Extracted {len(image_entities)} entities from image attachment: {filename}")
                
            except Exception as e:
                log.error(f"Failed to process image attachment {filename}: {e}")
    
    # Case 2: HTML content with embedded base64 images
    elif payload.get('mimeType') == 'text/html' and 'body' in payload and 'data' in payload['body']:
        try:
            html_content = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='replace')
            
            # Find base64 embedded images
            b64_image_pattern = r'data:image/(?P<type>jpeg|jpg|png|gif|webp);base64,(?P<data>[^"\'>\s]+)'
            b64_image_matches = re.finditer(b64_image_pattern, html_content, re.IGNORECASE)
            
            for i, match in enumerate(b64_image_matches):
                img_type = match.group('type')
                img_b64_data = match.group('data')
                filename = f"embedded_image_{i+1}.{img_type}"
                
                try:
                    # Decode base64 image data
                    image_bytes = base64.b64decode(img_b64_data)
                    
                    # Extract entities using Gemini
                    image_entities = extract_entities_from_image_bytes(
                        image_bytes, GEMINI_API_KEY, log
                    )
                    
                    for entity in image_entities:
                        image_entities_found_this_call.append({
                            'type': entity.type,
                            'value': entity.value,
                            'confidence': entity.confidence,
                            'source': f'embedded_image:{filename}'
                        })
                    
                    log.info(f"Extracted {len(image_entities)} entities from embedded image: {filename}")
                    
                except Exception as e:
                    log.error(f"Error processing embedded image {filename}: {e}")
                    
        except Exception as e:
            log.error(f"Error processing HTML content for embedded images in message {message_id}: {e}")
    
    # Case 3: Multipart content - recurse through parts
    if 'parts' in payload and payload['parts']:
        for sub_part in payload['parts']:
            process_image_attachments(sub_part, message_id, service, email_data, log, depth + 1)
    
    # Add entities found at this level to email_data
    if image_entities_found_this_call:
        if 'entities' not in email_data:
            email_data['entities'] = []
        
        # Deduplication
        existing_entity_tuples = set(
            (e.get('type'), e.get('value').lower() if isinstance(e.get('value'), str) else e.get('value'))
            for e in email_data['entities']
        )
        
        for new_entity_dict in image_entities_found_this_call:
            new_entity_tuple = (
                new_entity_dict['type'],
                new_entity_dict['value'].lower() if isinstance(new_entity_dict['value'], str) else new_entity_dict['value']
            )
            if new_entity_tuple not in existing_entity_tuples:
                email_data['entities'].append(new_entity_dict)
                existing_entity_tuples.add(new_entity_tuple)
                log.info(f"Added entity from image processing: {new_entity_dict}")
            else:
                log.debug(f"Skipping duplicate image entity: {new_entity_dict}")


def process_thread_images(
    thread_id: str,
    current_message_id: str,
    service: Resource,
    email_data: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    max_messages_to_check: int = 5
):
    """
    Processes previous messages in a thread for images and their entities using Gemini 3 Pro.
    Also extracts text entities from previous messages.
    """
    log = logger or logging.getLogger(__name__)
    log.info(f"Processing thread {thread_id} for images and text entities in previous messages (up to {max_messages_to_check}).")
    
    try:
        thread_data = service.users().threads().get(userId='me', id=thread_id, format='full').execute()
        messages_in_thread = thread_data.get('messages', [])
        
        if not messages_in_thread:
            log.info(f"No messages found in thread {thread_id}.")
            return
        
        # Sort messages by internalDate
        messages_in_thread.sort(key=lambda m: int(m.get('internalDate', 0)))
        
        current_msg_index = -1
        for i, msg_info in enumerate(messages_in_thread):
            if msg_info['id'] == current_message_id:
                current_msg_index = i
                break
        
        if current_msg_index == -1:
            log.warning(f"Current message {current_message_id} not found in thread {thread_id}.")
            return
        
        # Process previous messages
        start_index = max(0, current_msg_index - max_messages_to_check)
        messages_to_scan = messages_in_thread[start_index:current_msg_index]
        
        log.info(f"Found {len(messages_to_scan)} previous message(s) in thread {thread_id} to scan.")
        
        entity_extractor = EntityExtractor(api_key=GEMINI_API_KEY, model_name="gemini-3-pro-preview")
        
        for prev_msg_full_data in messages_to_scan:
            prev_msg_id = prev_msg_full_data['id']
            log.info(f"Scanning previous message {prev_msg_id} in thread {thread_id}.")
            
            # 1. Process text content
            prev_msg_body = extract_email_body(prev_msg_full_data, log)
            if prev_msg_body and len(prev_msg_body.strip()) > 10:
                text_entities = entity_extractor.extract_entities(prev_msg_body)
                if text_entities:
                    if 'entities' not in email_data:
                        email_data['entities'] = []
                    
                    existing_entity_tuples = set(
                        (e.get('type'), e.get('value').lower() if isinstance(e.get('value'), str) else e.get('value'))
                        for e in email_data['entities']
                    )
                    
                    for entity in text_entities:
                        new_entity_tuple = (entity.type, entity.value.lower() if isinstance(entity.value, str) else entity.value)
                        if new_entity_tuple not in existing_entity_tuples:
                            email_data['entities'].append({
                                'type': entity.type,
                                'value': entity.value,
                                'confidence': entity.confidence,
                                'source': f'previous_message_text:{prev_msg_id}'
                            })
                            existing_entity_tuples.add(new_entity_tuple)
                            log.info(f"Added text entity from prev_msg {prev_msg_id}: {entity.type} = '{entity.value}'")
            
            # 2. Process image attachments using Gemini
            if 'payload' in prev_msg_full_data:
                process_image_attachments(prev_msg_full_data['payload'], prev_msg_id, service, email_data, log)
                
    except Exception as e:
        log.error(f"Error processing thread images/text for thread {thread_id}: {e}", exc_info=True)



# --- Email Fetching & History ---
def fetch_emails_since_history_id(service: Resource, history_id: str, logger: Optional[logging.Logger] = None) -> List[str]:
    """Fetches new email IDs since a given history ID."""
    log = logger or logging.getLogger(__name__)
    log.info(f"Fetching emails since history ID: {history_id}")
    new_email_ids: Set[str] = set()
    try:
        history_response = service.users().history().list(
            userId="me",
            startHistoryId=history_id,
            historyTypes=['messageAdded'], # Only interested in new messages
            maxResults=500 # Adjust as needed, but new emails are usually few per notification
        ).execute()

        if 'history' in history_response:
            for record in history_response['history']:
                if 'messagesAdded' in record:
                    for msg_added_item in record['messagesAdded']:
                        if 'message' in msg_added_item and 'id' in msg_added_item['message']:
                            new_email_ids.add(msg_added_item['message']['id'])
            log.info(f"Found {len(new_email_ids)} new email IDs via history: {new_email_ids}")
        else:
            log.info(f"No 'history' field in response for history ID {history_id}. Might be no new messages.")
        
        # It's possible 'nextPageToken' exists if there's more history, but for Pub/Sub triggers,
        # usually, one history_id change covers a small number of events.
        # If you expect many emails per trigger, pagination here would be needed.

    except Exception as e:
        log.error(f"Error fetching emails since history ID {history_id}: {e}", exc_info=True)
    
    return list(new_email_ids)


def fetch_recent_emails(
    service: Resource, 
    days: int = 1, 
    logger: Optional[logging.Logger] = None,
    max_results: int = 15 # Limit the number of recent emails fetched
) -> List[str]:
    """Fetches recent email IDs as a fallback if history ID fails or isn't provided."""
    log = logger or logging.getLogger(__name__)
    query_date = (datetime.now() - timedelta(days=days)).strftime('%Y/%m/%d')
    query = f"after:{query_date} is:unread in:inbox" # More specific query: unread in inbox
    log.info(f"Fetching recent emails with query: '{query}', max_results={max_results}")
    email_ids: List[str] = []
    try:
        response = service.users().messages().list(
            userId='me', q=query, maxResults=max_results, includeSpamTrash=False
        ).execute()
        if 'messages' in response:
            email_ids = [msg['id'] for msg in response['messages']]
            log.info(f"Found {len(email_ids)} recent email IDs: {email_ids}")
        else:
            log.info("No recent emails found matching the query.")
    except Exception as e:
        log.error(f"Error fetching recent emails: {e}", exc_info=True)
    return email_ids


# --- Draft Management ---
def draft_exists_for_thread(service: Resource, thread_id: str, logger: Optional[logging.Logger] = None) -> bool:
    """Checks if a draft already exists for the given thread ID."""
    log = logger or logging.getLogger(__name__)
    if not thread_id:
        log.warning("Cannot check for existing drafts - thread_id is empty.")
        return False
    try:
        drafts_response = service.users().drafts().list(userId='me').execute()
        if 'drafts' in drafts_response:
            for draft_summary in drafts_response['drafts']:
                draft_id = draft_summary.get('id')
                if draft_id:
                    # Fetch the full draft to get its threadId
                    try:
                        draft_full = service.users().drafts().get(userId='me', id=draft_id).execute()
                        if draft_full.get('message', {}).get('threadId') == thread_id:
                            log.info(f"Draft {draft_id} already exists for thread {thread_id}.")
                            return True
                    except Exception as e_draft:
                        log.warning(f"Could not retrieve full draft {draft_id} to check thread: {e_draft}")
            log.debug(f"No existing draft found for thread {thread_id} after checking {len(drafts_response['drafts'])} drafts.")
        else:
            log.debug(f"No drafts found in the account for thread {thread_id}.")
        return False
    except Exception as e:
        log.error(f"Error listing drafts while checking for thread {thread_id}: {e}", exc_info=True)
        return False # Fail safe: assume no draft to avoid missing a reply


def save_response_as_draft(
    service: Resource, 
    thread_id: str, 
    to_email: str, 
    subject: str, 
    response_text: str, 
    logger: Optional[logging.Logger] = None,
    storage_client: Optional[storage.Client] = None # For draft locking
) -> Optional[str]:
    """Saves the response as a Gmail draft within the specified thread."""
    log = logger or logging.getLogger(__name__)
    
    # --- Draft Locking (Optional but good for concurrent environments) ---
    # This is a simplified lock. For true distributed locking, use a proper service.
    lock_blob = None
    draft_lock_acquired = False
    if storage_client:
        lock_bucket_name = BUCKET_NAME # Or a dedicated locks bucket
        lock_blob_name = f"draft_locks/{thread_id}.lock"
        try:
            lock_bucket = storage_client.bucket(lock_bucket_name)
            lock_blob = lock_bucket.blob(lock_blob_name)
            
            if lock_blob.exists():
                # Check lock age - if too old, maybe stale?
                lock_meta = lock_blob.reload() # Get metadata like timeCreated
                lock_creation_time = lock_blob.time_created
                if lock_creation_time and (datetime.now(lock_creation_time.tzinfo) - lock_creation_time > timedelta(minutes=5)):
                    log.warning(f"Stale draft lock found for thread {thread_id} (older than 5 mins). Proceeding cautiously.")
                    # Potentially delete stale lock here, but be very careful
                else:
                    log.info(f"Draft creation potentially in progress by another instance for thread {thread_id} (lock exists). Aborting draft creation.")
                    return "DRAFT_LOCK_EXISTS" # Indicate lock conflict
            
            lock_blob.upload_from_string(json.dumps({"timestamp": datetime.now().isoformat()}))
            draft_lock_acquired = True
            log.info(f"Acquired draft lock for thread {thread_id}.")

        except Exception as lock_e:
            log.warning(f"Error during draft lock acquisition for thread {thread_id}: {lock_e}", exc_info=True)
            # Proceed without lock if acquisition failed, but log it.
    
    # Re-check if draft exists *after* attempting to acquire lock
    if draft_exists_for_thread(service, thread_id, log):
        log.info(f"Draft already exists for thread {thread_id} (checked after lock attempt). Not creating another.")
        if draft_lock_acquired and lock_blob: # Release lock if we acquired it but won't create draft
            try: lock_blob.delete()
            except Exception: pass
        return "EXISTING_DRAFT_FOUND_LATE"

    try:
        message = MIMEMultipart()
        message['to'] = to_email
        # Ensure subject is a reply if not already
        if not subject.lower().startswith('re:'):
            message['subject'] = f"Re: {subject}"
        else:
            message['subject'] = subject
        
        message.attach(MIMEText(response_text, 'plain')) # Assuming plain text response
        
        raw_message_bytes = message.as_bytes()
        raw_message_b64 = base64.urlsafe_b64encode(raw_message_bytes).decode('utf-8')

        draft_body = {
            'message': {
                'raw': raw_message_b64,
                'threadId': thread_id
            }
        }
        created_draft = service.users().drafts().create(userId='me', body=draft_body).execute()
        draft_id = created_draft.get('id')
        log.info(f"Successfully created draft ID {draft_id} for thread {thread_id}, To: {to_email}")
        return draft_id
    except Exception as e:
        log.error(f"Error creating Gmail draft for thread {thread_id}, To: {to_email}: {e}", exc_info=True)
        return None
    finally:
        # Release lock
        if draft_lock_acquired and lock_blob:
            try:
                lock_blob.delete()
                log.info(f"Released draft lock for thread {thread_id}.")
            except Exception as unlock_e:
                log.warning(f"Error releasing draft lock for thread {thread_id}: {unlock_e}", exc_info=True)


# --- Processed Email Tracking (using GCS) ---
def _get_storage_client(logger: logging.Logger) -> Optional[storage.Client]:
    """Helper to get a GCS client, preferring service account if specified."""
    try:
        # In GCP environment (like Cloud Functions), storage.Client() usually works with ambient creds.
        # For local testing, ensure GOOGLE_APPLICATION_CREDENTIALS is set or provide SA key.
        if SERVICE_ACCOUNT_FILE and os.path.exists(SERVICE_ACCOUNT_FILE) and not os.environ.get("FUNCTION_TARGET"):
             # Use SA file explicitly for local dev if present and not in a function
            return storage.Client.from_service_account_json(SERVICE_ACCOUNT_FILE)
        return storage.Client() # Standard client
    except Exception as e:
        logger.error(f"Failed to initialize Google Cloud Storage client: {e}")
        return None

def check_processed_in_storage(message_id: str, storage_client: Optional[storage.Client], logger: Optional[logging.Logger] = None) -> bool:
    """Checks if a message_id is in the list of processed emails in GCS."""
    log = logger or logging.getLogger(__name__)
    if not message_id:
        log.warning("Cannot check processed status: message_id is empty.")
        return False
    if not storage_client:
        log.error("Storage client not available for check_processed_in_storage.")
        return False # Cannot check, assume not processed to be safe

    tracking_blob_name = os.environ.get("PROCESSED_EMAILS_BLOB", "processed_emails/tracked_emails.json")
    try:
        bucket = storage_client.bucket(BUCKET_NAME) # BUCKET_NAME from config
        blob = bucket.blob(tracking_blob_name)
        if blob.exists():
            tracking_data_str = blob.download_as_text()
            tracking_data = json.loads(tracking_data_str)
            processed_ids = tracking_data.get("processed_message_ids", [])
            if message_id in processed_ids:
                log.info(f"Message {message_id} found in GCS processed list.")
                return True
        log.debug(f"Message {message_id} not found in GCS processed list (or list doesn't exist).")
        return False
    except Exception as e:
        log.error(f"Error checking processed emails in GCS for {message_id}: {e}", exc_info=True)
        return False # Fail safe: assume not processed

def mark_as_processed(message_id: str, storage_client: Optional[storage.Client], logger: Optional[logging.Logger] = None, max_history: int = 2000) -> bool:
    """Marks a message_id as processed by adding it to a list in GCS."""
    log = logger or logging.getLogger(__name__)
    if not message_id:
        log.warning("Cannot mark as processed: message_id is empty.")
        return False
    if not storage_client:
        log.error("Storage client not available for mark_as_processed.")
        return False

    tracking_blob_name = os.environ.get("PROCESSED_EMAILS_BLOB", "processed_emails/tracked_emails.json")
    try:
        bucket = storage_client.bucket(BUCKET_NAME) # BUCKET_NAME from config
        blob = bucket.blob(tracking_blob_name)
        
        if blob.exists():
            try:
                tracking_data_str = blob.download_as_text()
                tracking_data = json.loads(tracking_data_str)
                if not isinstance(tracking_data.get("processed_message_ids"), list):
                    tracking_data["processed_message_ids"] = [] # Reset if malformed
            except json.JSONDecodeError:
                log.warning(f"Malformed JSON in {tracking_blob_name}, reinitializing.")
                tracking_data = {"processed_message_ids": []}
        else:
            tracking_data = {"processed_message_ids": []}

        if message_id not in tracking_data["processed_message_ids"]:
            tracking_data["processed_message_ids"].append(message_id)
            # Prune old entries to keep the list size manageable
            if len(tracking_data["processed_message_ids"]) > max_history:
                tracking_data["processed_message_ids"] = tracking_data["processed_message_ids"][-max_history:]
            
            blob.upload_from_string(json.dumps(tracking_data))
            log.info(f"Message {message_id} marked as processed in GCS.")
        else:
            log.debug(f"Message {message_id} was already in processed list.")
        return True
    except Exception as e:
        log.error(f"Error marking message {message_id} as processed in GCS: {e}", exc_info=True)
        return False

def has_email_been_processed(
    message_id: str, 
    thread_id: str, 
    service: Resource, 
    storage_client: Optional[storage.Client], 
    logger: Optional[logging.Logger] = None
) -> bool:
    """Comprehensive check if an email has been processed (GCS log or existing draft)."""
    log = logger or logging.getLogger(__name__)
    
    if storage_client and check_processed_in_storage(message_id, storage_client, log):
        log.info(f"Deduplication: Message {message_id} already marked as processed in GCS.")
        return True
        
    if draft_exists_for_thread(service, thread_id, log):
        log.info(f"Deduplication: Draft already exists for thread {thread_id} (message {message_id}).")
        # If draft exists but not marked in GCS, mark it now for future GCS checks
        if storage_client:
            mark_as_processed(message_id, storage_client, log)
        return True
        
    log.info(f"Message {message_id} (thread {thread_id}) has not been processed yet.")
    return False

def extract_email_body_with_history(message: Dict[str, Any], logger: Optional[logging.Logger] = None) -> tuple[str, str]:
    """
    Extract email body and separate current message from conversation history
    Returns: (current_message, full_conversation_text)
    """
    log = logger or logging.getLogger(__name__)
    
    try:
        # Extract the full email body using existing function
        full_body = extract_email_body(message, logger)
        
        if not full_body:
            log.warning("No email body found")
            return "", ""
        
        # Use improved separation logic from utils.py
        current_message, previous_conversation = separate_email_parts(full_body)
        
        # Log the separation results
        log.debug(f"Current message length: {len(current_message)}")
        log.debug(f"Previous conversation length: {len(previous_conversation)}")
        
        # Return current message and full body for conversation processing
        return current_message, full_body
        
    except Exception as e:
        log.error(f"Error extracting email body with history: {e}", exc_info=True)
        return "", ""
