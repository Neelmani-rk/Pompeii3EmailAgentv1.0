import base64
import json
import logging
import time
import traceback
from datetime import datetime
import sys
import os
from typing import List, Dict, Any, Optional, Set

# Google Cloud specific imports for Cloud Functions
import functions_framework
from google.cloud import storage

# Add this import to your existing gmail_helpers imports
from gmail_helpers import extract_email_body_with_history

# Imports from other project modules
try:
    from conversation_processor import ConversationProcessor
    from conversation_storage import ConversationStorage
    from config import (
        SERVICE_ACCOUNT_FILE, BUCKET_NAME, RESPONSES_BUCKET_NAME, GEMINI_API_KEY,
        RATE_PER_MINUTE, VECTOR_DB_BUCKET, VECTOR_DB_PREFIX, LOCAL_DB_DIR, HISTORY_ID_BUCKET, HISTORY_ID_OBJECT
    )
    from utils import (
        setup_logging, extract_email_address, extract_order_number_from_subject,
        separate_email_parts, is_automated_reply
    )
    from entity_extractor import EntityExtractor, ExtractedEntity, extract_entities_from_image_bytes, extract_text_from_image_with_gemini
    from api_clients import (
        analyze_email_sentiment, detect_intent_with_gemini, call_gemini_api
    )
    from order_processing import process_order_tracking_info
    from gmail_helpers import (
        get_gmail_service, extract_email_body, process_image_attachments,
        process_thread_images, save_response_as_draft,
        fetch_emails_since_history_id, fetch_recent_emails,
        check_processed_in_storage, mark_as_processed, draft_exists_for_thread,
        has_email_been_processed, extract_email_body_with_history
    )
    from response_generation import generate_llm_payload, generate_response
    from vector_db import VectorKnowledgeBase, initialize_vector_db_from_gcs  # CRITICAL: This was missing
    from api_clients import ShipStationAPI
    # Import validation and analytics modules
    from input_validation import validate_email_data, InputValidator
    from email_quality_validator import validate_draft_quality
    from response_analytics import log_response_analytics
except ImportError as import_error:
    print(f"Import error: {import_error}")
    print("Attempting fallback imports...")
    try:
        from config import (
            SERVICE_ACCOUNT_FILE, BUCKET_NAME, RESPONSES_BUCKET_NAME, GEMINI_API_KEY,
            RATE_PER_MINUTE, VECTOR_DB_BUCKET, VECTOR_DB_PREFIX, LOCAL_DB_DIR, HISTORY_ID_BUCKET, HISTORY_ID_OBJECT
        )
        from utils import (
            setup_logging, extract_email_address, extract_order_number_from_subject, 
            separate_email_parts, is_automated_reply
        )
        from entity_extractor import EntityExtractor, ExtractedEntity, extract_entities_from_image_bytes, extract_text_from_image_with_gemini
        from api_clients import (
            analyze_email_sentiment, detect_intent_with_gemini, call_gemini_api
        )
        from order_processing import process_order_tracking_info
        from gmail_helpers import (
            get_gmail_service, extract_email_body, process_image_attachments,
            process_thread_images, save_response_as_draft,
            fetch_emails_since_history_id, fetch_recent_emails,
            check_processed_in_storage, mark_as_processed, draft_exists_for_thread,
            has_email_been_processed, extract_email_body_with_history

        )
        from response_generation import generate_llm_payload, generate_response
        from vector_db import VectorKnowledgeBase, initialize_vector_db_from_gcs  # CRITICAL: Add this here too
        from api_clients import ShipStationAPI
        # Import validation and analytics modules
        from input_validation import validate_email_data, InputValidator
        from email_quality_validator import validate_draft_quality
        from response_analytics import log_response_analytics
    except ImportError as e2:
        print(f"Fallback import also failed: {e2}")
        print("Please ensure all required modules are in the same directory or properly installed.")
        sys.exit(1)

def save_history_id_to_gcs_json(storage_client, bucket_name, object_name, history_id, logger=None):

    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        content = json.dumps({"last_history_id": history_id})
        blob.upload_from_string(content, content_type='application/json')
        if logger:
            logger.info(f"Saved historyId {history_id} to gs://{bucket_name}/{object_name} in JSON format.")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Failed to save historyId to GCS: {e}")
        return False

def load_history_id_from_gcs_json(storage_client, bucket_name, object_name, logger=None):
    """
    Load the last processed Gmail historyId from a Cloud Storage object in JSON format.
    """
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        if not blob.exists():
            if logger:
                logger.warning(f"No historyId found at gs://{bucket_name}/{object_name}. Starting fresh.")
            return None
        content = blob.download_as_text()
        data = json.loads(content)
        history_id = data.get("last_history_id")
        if logger:
            logger.info(f"Loaded historyId {history_id} from gs://{bucket_name}/{object_name}")
        return history_id
    except Exception as e:
        if logger:
            logger.error(f"Failed to load historyId from GCS: {e}")
        return None

def process_email(message_id: str, service: Any, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """
    Enhanced process_email function with improved customer name handling and conversation history processing.
    """
    try:
        logger.info(f"Starting full processing for message ID: {message_id}")
        
        # Get message details (full format)
        message = service.users().messages().get(userId='me', id=message_id, format='full').execute()
        headers = {header['name']: header['value'] for header in message.get('payload', {}).get('headers', [])}
        
        sender_email_raw = headers.get('From', '')
        
        # Enhanced sender name extraction (around line 85)
        sender_name_header = next((h['value'] for h in message.get('payload', {}).get('headers', []) if h['name'].lower() == 'x-sender-name'), None)
        if sender_name_header:
            sender_name = sender_name_header
        elif '<' in sender_email_raw:
            # Extract name from "Name <email>" format
            sender_name = sender_email_raw.split('<')[0].strip().strip('"\'')
        else:
            # Fallback to email username
            sender_name = sender_email_raw.split('@')[0] if '@' in sender_email_raw else sender_email_raw

        # Clean and validate sender name
        sender_name = sender_name.strip()
        if len(sender_name) < 2 or sender_name.lower() in ['no', 'reply', 'noreply']:
            sender_name = None
        
        sender_email_clean = extract_email_address(sender_email_raw)
        subject = headers.get('Subject', 'No Subject')
        date_str = headers.get('Date', '')
        thread_id = message.get('threadId', '')
        
        logger.info(f"Email From: {sender_email_clean} (Name: {sender_name}), Subject: '{subject}', Thread ID: {thread_id}")
        
        if is_automated_reply(sender_email_raw, logger):
            logger.warning(f"Skipped automated/internal email from: {sender_email_raw}")
            return None
        
        # Extract message body with conversation history
        current_message, full_conversation = extract_email_body_with_history(message, logger)
        
        if not current_message or len(current_message.strip()) < 10:
            logger.warning(f"Email body is too short or empty for message ID {message_id}. Skipping.")
            return None
        
        # Initialize conversation processor
        storage_client = storage.Client()
        
        # Process conversation history if it exists and is substantial
        structured_conversation = {}
        if full_conversation and len(full_conversation) > len(current_message) + 50:
            try:
                # Import here to avoid circular imports
                from conversation_processor import ConversationProcessor
                
                conversation_processor = ConversationProcessor(GEMINI_API_KEY, storage_client)
                structured_conversation = conversation_processor.process_conversation_history(
                    full_conversation, thread_id
                )
                logger.info(f"Processed conversation history for thread {thread_id}")
            except Exception as e:
                logger.error(f"Error processing conversation history: {e}")
                structured_conversation = {}
        
        email_data: Dict[str, Any] = {
            'message_id': message_id,
            'thread_id': thread_id,
            'sender_name': sender_name,
            'sender': sender_email_clean,
            'subject': subject,
            'date': date_str,
            'body': full_conversation,
            'current_message': current_message,
            'conversation_history': structured_conversation,  # New field
            'entities': [],
        }
        
        # Separate current message from thread history for NLP processing
        text_for_nlp = f"{subject}\n\n{current_message}".strip()
        
        # Sentiment analysis
        sentiment = analyze_email_sentiment(text_for_nlp, logger)
        email_data['sentiment'] = sentiment if sentiment else {'score': 0.0, 'magnitude': 0.0}
        logger.info(f"Sentiment: {email_data['sentiment']}")
        
        # Intent detection
        intents = detect_intent_with_gemini(text_for_nlp, logger)
        email_data['intent'] = intents if intents else \
            {'primary': {'type': 'general-query', 'confidence': 0.5}, 'all_intents': []}
        logger.info(f"Detected Intent: {email_data['intent'].get('primary')}")
        
        # Entity extraction
        entity_extractor = EntityExtractor(api_key=GEMINI_API_KEY, model_name="gemini-3-pro-preview")
        extracted_entities_obj = entity_extractor.extract_entities(text_for_nlp)
        
        # Add order numbers from subject
        subject_order_numbers = extract_order_number_from_subject(subject, logger)
        current_entity_values = {e.value for e in extracted_entities_obj if e.type in ["ORDER_NUMBER", "PURCHASE_ORDER_NUMBER"]}
        
        for son in subject_order_numbers:
            if son not in current_entity_values:
                entity_type = "PURCHASE_ORDER_NUMBER" if len(son) == 15 else "ORDER_NUMBER"
                extracted_entities_obj.append(ExtractedEntity(type=entity_type, value=son, confidence=0.95))
                logger.info(f"Added {entity_type} '{son}' from subject to entities.")
        
        # Add customer name as entity if not already present (enhanced)
        customer_name_entities = [e for e in extracted_entities_obj if e.type == "CUSTOMER_NAME"]
        if not customer_name_entities and sender_name and len(sender_name) > 2:
            # Additional validation for customer name
            if not sender_name.isdigit() and '@' not in sender_name:
                extracted_entities_obj.append(ExtractedEntity(type="CUSTOMER_NAME", value=sender_name, confidence=0.8))
                logger.info(f"Added customer name '{sender_name}' from sender to entities.")
        
        email_data['entities'] = [
            {'type': e.type, 'value': e.value, 'confidence': e.confidence, 'source': 'text'}
            for e in extracted_entities_obj
        ]
        
        logger.info(f"Extracted {len(email_data['entities'])} initial text entities.")
        
        # Process image attachments
        if 'payload' in message:
            process_image_attachments(message['payload'], message_id, service, email_data, logger)
        
        # Process thread images
        if thread_id:
            process_thread_images(thread_id, message_id, service, email_data, logger)
        
        logger.info(f"Total entities after image processing: {len(email_data['entities'])}")
        
        # Enhanced order processing with customer name fallback
        process_order_tracking_info(email_data, logger)
        
        logger.info(f"Email processing complete for message ID: {message_id}. Final email_data keys: {list(email_data.keys())}")
        
        return email_data
        
    except Exception as e:
        logger.error(f"Error during core processing of email {message_id}: {e}", exc_info=True)
        return None


# --- Main Orchestration for Responding ---
def process_and_respond_to_email(
    message_id: str,
    service: Any,
    vector_db: Optional[VectorKnowledgeBase],
    storage_client: storage.Client,
    logger: logging.Logger
) -> bool:
    """
    Orchestrates the full lifecycle of processing an email and saving a draft response.
    Includes deduplication checks.
    Returns True if processed (or already processed/skipped), False on critical error.
    """
    processing_start_time = time.time()
    start_datetime = datetime.now()
    logger.info(f"PROCESS_AND_RESPOND START TIME: {start_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} for Msg: {message_id}")

    try:
        # --- Deduplication and Pre-checks ---
        try:
            message_minimal = service.users().messages().get(userId='me', id=message_id, format='minimal').execute()
            thread_id = message_minimal.get('threadId')
            if not thread_id:
                logger.error(f"Could not get thread ID for message {message_id}. Cannot proceed.")
                return False
        except Exception as e_minimal:
            logger.error(f"Failed to fetch minimal message data for {message_id}: {e_minimal}. Cannot proceed.")
            return False

        logger.info(f"Message {message_id} belongs to Thread ID: {thread_id}.")

        if has_email_been_processed(message_id, thread_id, service, storage_client, logger):
            logger.info(f"Deduplication: Message {message_id} / Thread {thread_id} already handled. Skipping.")
            return True

        # --- Core Email Processing ---
        email_data = process_email(message_id, service, logger)
        if not email_data:
            logger.warning(f"No data extracted from email {message_id} or it was skipped. Not generating response.")
            mark_as_processed(message_id, storage_client, logger)
            return True
        
        # --- Input Validation ---
        try:
            validated_email_data = validate_email_data(email_data, logger)
            email_data.update(validated_email_data)
            logger.info(f"Email data validated successfully for message {message_id}")
        except Exception as validation_error:
            logger.error(f"Email data validation failed for {message_id}: {validation_error}")
            # Continue with original data but log the issue
            pass

        # --- Response Generation ---
        logger.info(f"Generating LLM payload for message {message_id}")
        llm_payload = generate_llm_payload(email_data, logger)

        logger.info(f"Generating initial response template for message {message_id}")
        initial_response = generate_response(llm_payload, logger)

        logger.info(f"Calling Gemini API to refine response for message {message_id}")
        final_response = call_gemini_api(
            initial_response, llm_payload, GEMINI_API_KEY, vector_db, logger
        )
        
        # --- Email Quality Validation ---
        draft_approved = True
        quality_metrics = {}
        improvement_suggestions = []
        
        try:
            # Prepare customer context for validation
            customer_context = {
                'sender': email_data.get('sender', ''),
                'order_number': next(
                    (entity['value'] for entity in email_data.get('entities', []) 
                     if entity['type'] in ['ORDER_NUMBER', 'PURCHASE_ORDER_NUMBER']), 
                    None
                ),
                'intent': email_data.get('intent', {})
            }
            
            # Validate draft quality
            original_query = email_data.get('current_message', '')
            draft_approved, validation_results = validate_draft_quality(
                final_response, original_query, customer_context, logger
            )
            
            quality_metrics = validation_results.get('quality_metrics', {})
            improvement_suggestions = validation_results.get('improvement_suggestions', [])
            
            logger.info(f"Draft quality validation completed for {message_id}. Approved: {draft_approved}")
            if not draft_approved:
                logger.warning(f"Draft quality below threshold for {message_id}. Suggestions: {improvement_suggestions[:3]}")
            
        except Exception as quality_error:
            logger.error(f"Draft quality validation failed for {message_id}: {quality_error}")
            # Default to approving the draft if validation fails
            draft_approved = True

        # --- Save Draft ---
        sender_email_clean = email_data.get('sender', '')
        subject = email_data.get('subject', 'Re: Your Inquiry')

        if thread_id and sender_email_clean:
            if draft_exists_for_thread(service, thread_id, logger):
                logger.info(f"DEDUPLICATION (Final Check): Draft already exists for thread {thread_id}. Not creating another.")
                mark_as_processed(message_id, storage_client, logger)
                return True

            logger.info(f"Attempting to save draft for thread {thread_id}, To: {sender_email_clean}")
            draft_id_or_status = save_response_as_draft(
                service, thread_id, sender_email_clean, subject, final_response, logger, storage_client
            )

            if draft_id_or_status and draft_id_or_status not in ["EXISTING_DRAFT_FOUND_LATE", "DRAFT_LOCK_EXISTS"]:
                logger.info(f"Draft successfully created/managed for message {message_id}. Draft ID/Status: {draft_id_or_status}")
                mark_as_processed(message_id, storage_client, logger)
                
                # --- Log Analytics ---
                try:
                    response_time = time.time() - processing_start_time
                    log_response_analytics(
                        response_time=response_time,
                        quality_metrics=quality_metrics,
                        customer_context=customer_context if 'customer_context' in locals() else {'sender': email_data.get('sender', '')},
                        draft_approved=draft_approved,
                        logger=logger
                    )
                    logger.info(f"Analytics logged for message {message_id}")
                except Exception as analytics_error:
                    logger.error(f"Failed to log analytics for {message_id}: {analytics_error}")

                try:
                    username = sender_email_clean.split('@')[0] if '@' in sender_email_clean else sender_email_clean
                    ts = datetime.now().strftime("%Y%m%d%H%M%S")
                    # Ensure directories exist if saving locally, GCS handles paths
                    if BUCKET_NAME: # Assuming GCS usage
                        payload_filename = f"llm_payloads/{username}_{message_id}_{ts}.json"
                        response_filename = f"generated_responses/{username}_{message_id}_{ts}.txt"
                        
                        bucket_payloads = storage_client.bucket(BUCKET_NAME)
                        blob_payload = bucket_payloads.blob(payload_filename)
                        blob_payload.upload_from_string(json.dumps(llm_payload, indent=2, default=str))
                        
                        bucket_responses = storage_client.bucket(RESPONSES_BUCKET_NAME)
                        blob_response = bucket_responses.blob(response_filename)
                        blob_response.upload_from_string(final_response)
                        logger.info(f"Saved LLM payload to GCS BUCKET_NAME/{payload_filename} and response to GCS RESPONSES_BUCKET_NAME/{response_filename}")
                except Exception as storage_log_e:
                    logger.error(f"Error saving payload/response to GCS for {message_id}: {storage_log_e}")
            elif draft_id_or_status in ["EXISTING_DRAFT_FOUND_LATE", "DRAFT_LOCK_EXISTS"]:
                 logger.info(f"Draft not created for {message_id} due to existing draft or lock: {draft_id_or_status}")
                 mark_as_processed(message_id, storage_client, logger)
            else:
                logger.error(f"Failed to save draft for message {message_id}. Status: {draft_id_or_status}")
                return False
        else:
            logger.error(f"Cannot save draft for message {message_id}: Missing thread_id ('{thread_id}') or sender_email ('{sender_email_clean}').")
            return False

        return True

    except Exception as e:
        logger.error(f"Unhandled error in process_and_respond_to_email for {message_id}: {e}", exc_info=True)
        logger.error(traceback.format_exc())
        return False
    finally:
        processing_end_time = time.time()
        end_datetime = datetime.now()
        execution_time = processing_end_time - processing_start_time
        logger.info(f"PROCESS_AND_RESPOND END TIME: {end_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} for Msg: {message_id}")
        logger.info(f"PROCESS_AND_RESPOND EXECUTION TIME: {execution_time:.3f} seconds for Msg: {message_id}")

        seconds_per_email = 60.0 / RATE_PER_MINUTE
        if execution_time < seconds_per_email:
            sleep_time = seconds_per_email - execution_time
            if sleep_time > 0:
                 logger.info(f"Rate limiting: Sleeping for {sleep_time:.2f} seconds after processing {message_id}.")
                 time.sleep(sleep_time)


# --- Cloud Function Entry Points ---

@functions_framework.http # type: ignore
def http_endpoint(request: Any) -> Any: # type: ignore
    """HTTP endpoint, typically for health checks or manual triggers."""
    logger = logging.getLogger("customer_service_http")
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)
    
    logger.info(f"HTTP endpoint called. Request method: {request.method}, Path: {request.path}")
    return "Customer Service AI (HTTP) is alive.", 200

@functions_framework.cloud_event
def pubsub_handler(cloud_event: Any) -> Dict[str, Any]:
    """
    Pub/Sub triggered Cloud Function.
    Processes new Gmail messages based on Pub/Sub notifications.
    """
    logger = setup_logging()
    logger.info(f"Pub/Sub event received. Event ID: {cloud_event}, Type: {cloud_event}")
    
    function_start_time = datetime.now()
    processed_count = 0
    total_to_process = 0
    errors_occurred = False

    try:
        # Initialize Gmail service
        service = get_gmail_service(logger)
        if not service:
            logger.critical("Failed to initialize Gmail service. Cannot proceed.")
            return {"status": "critical_error", "message": "Gmail service init failed"}

        # Initialize GCS Storage Client
        storage_client = storage.Client()

        # Initialize Vector Database from GCS
        logger.info(f"Initializing vector database from GCS: gs://{VECTOR_DB_BUCKET}/{VECTOR_DB_PREFIX} to {LOCAL_DB_DIR}")
        try:
            vector_db = initialize_vector_db_from_gcs(
                VECTOR_DB_BUCKET, VECTOR_DB_PREFIX, LOCAL_DB_DIR, logger
            )
            if not vector_db or not vector_db.is_ready():
                logger.error("Vector database initialization failed. Responses will not use knowledge base augmentation.")
                vector_db = None
        except Exception as e:
            logger.warning(f"Vector database initialization failed: {e}. Proceeding without knowledge base augmentation.")
            vector_db = None

        # Load last processed historyId from GCS
        last_history_id = load_history_id_from_gcs_json(
            storage_client, HISTORY_ID_BUCKET, HISTORY_ID_OBJECT, logger
        )

        # Decode Pub/Sub message and extract new historyId
        message_ids_to_process: List[str] = []
        processed_ids_in_this_run: Set[str] = set()
        new_history_id = None

        if cloud_event.data and "message" in cloud_event.data:
            pubsub_message_data_encoded = cloud_event.data["message"].get("data")
            if pubsub_message_data_encoded:
                try:
                    decoded_data_str = base64.b64decode(pubsub_message_data_encoded).decode("utf-8")
                    logger.info(f"Decoded Pub/Sub message data: {decoded_data_str}")
                    gmail_notification = json.loads(decoded_data_str)
                    new_history_id = gmail_notification.get('historyId')

                    if new_history_id and last_history_id:
                        logger.info(f"Processing Gmail history from {last_history_id} to {new_history_id}")
                        ids_from_history = fetch_emails_since_history_id(
                            service, str(last_history_id), logger
                        )
                        for msg_id in ids_from_history:
                            if msg_id not in processed_ids_in_this_run:
                                message_ids_to_process.append(msg_id)
                                processed_ids_in_this_run.add(msg_id)
                        logger.info(f"Found {len(ids_from_history)} message IDs from history. Unique to process: {len(message_ids_to_process)}")

                    elif new_history_id:
                        logger.warning("No previous historyId found. Using new_history_id as start.")
                        ids_from_history = fetch_emails_since_history_id(
                            service, str(new_history_id), logger
                        )
                        for msg_id in ids_from_history:
                            if msg_id not in processed_ids_in_this_run:
                                message_ids_to_process.append(msg_id)
                                processed_ids_in_this_run.add(msg_id)
                        logger.info(f"Processed {len(ids_from_history)} message IDs from new historyId only.")

                    else:
                        logger.warning("No 'historyId' found in Pub/Sub message. Fallback to recent emails.")
                        recent_ids = fetch_recent_emails(service, days=1, logger=logger, max_results=10)
                        for msg_id in recent_ids:
                            if msg_id not in processed_ids_in_this_run:
                                message_ids_to_process.append(msg_id)
                                processed_ids_in_this_run.add(msg_id)
                        logger.info(f"Fallback: Found {len(recent_ids)} recent emails. Unique to process: {len(message_ids_to_process)}")

                except Exception as e_decode:
                    logger.error(f"Error decoding Pub/Sub message: {e_decode}", exc_info=True)
                    errors_occurred = True
            else:
                logger.warning("No 'data' field in Pub/Sub message.message.")
        else:
            logger.warning("No 'message' or 'data' in Pub/Sub cloud_event. Cannot determine emails to process.")

        total_to_process = len(message_ids_to_process)
        if total_to_process > 0:
            logger.info(f"Will attempt to process {total_to_process} email(s): {message_ids_to_process}")

            for msg_id_to_process in message_ids_to_process:
                logger.info(f"--- Processing email ID: {msg_id_to_process} ---")
                if process_and_respond_to_email(msg_id_to_process, service, vector_db, storage_client, logger):
                    processed_count += 1
                else:
                    errors_occurred = True
                logger.info(f"--- Finished processing email ID: {msg_id_to_process} ---")

            # Save the new historyId to GCS after successful processing
            if new_history_id:
                save_history_id_to_gcs_json(
                    storage_client, HISTORY_ID_BUCKET, HISTORY_ID_OBJECT, new_history_id, logger
                )
        else:
            logger.info("No new email messages found to process in this invocation.")

        function_duration = (datetime.now() - function_start_time).total_seconds()
        logger.info(f"Pub/Sub handler finished. Processed: {processed_count}/{total_to_process}. Duration: {function_duration:.2f}s. Errors: {errors_occurred}")

        if errors_occurred and total_to_process > 0:
            return {"status": "partial_completion", "processed": processed_count, "total_queried": total_to_process, "duration_seconds": function_duration}
        elif total_to_process == 0 and not errors_occurred:
            return {"status": "success_no_new_messages", "processed": 0, "total_queried": 0, "duration_seconds": function_duration}
        elif not errors_occurred and processed_count == total_to_process:
            return {"status": "success_all_processed", "processed": processed_count, "total_queried": total_to_process, "duration_seconds": function_duration}
        else:
            return {"status": "unknown_completion_state", "processed": processed_count, "total_queried": total_to_process, "duration_seconds": function_duration}

    except Exception as e_main:
        logger.critical(f"Critical unhandled error in pubsub_handler: {e_main}", exc_info=True)
        function_duration = (datetime.now() - function_start_time).total_seconds()
        return {"status": "critical_failure", "message": str(e_main), "processed": processed_count, "total_queried": total_to_process, "duration_seconds": function_duration}


# =============================================================================
# Flask Application for Cloud Run
# =============================================================================

try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

if FLASK_AVAILABLE:
    app = Flask(__name__)

    @app.route('/health', methods=['GET'])
    def health_check():
        """
        Health check endpoint for Cloud Run and load balancers.
        Returns system status and component health.
        """
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": {}
        }
        
        # Check Gmail service
        try:
            service = get_gmail_service()
            if service:
                health_status["components"]["gmail"] = {"status": "healthy"}
            else:
                health_status["components"]["gmail"] = {"status": "degraded", "message": "Service not initialized"}
        except Exception as e:
            health_status["components"]["gmail"] = {"status": "unhealthy", "message": str(e)}
            health_status["status"] = "degraded"
        
        # Check GCS connectivity
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(BUCKET_NAME)
            if bucket.exists():
                health_status["components"]["gcs"] = {"status": "healthy"}
            else:
                health_status["components"]["gcs"] = {"status": "degraded", "message": "Bucket not found"}
        except Exception as e:
            health_status["components"]["gcs"] = {"status": "unhealthy", "message": str(e)}
            health_status["status"] = "degraded"
        
        # Check Gemini API key
        if GEMINI_API_KEY:
            health_status["components"]["gemini_api"] = {"status": "configured"}
        else:
            health_status["components"]["gemini_api"] = {"status": "not_configured"}
            health_status["status"] = "degraded"
        
        status_code = 200 if health_status["status"] == "healthy" else 503
        return jsonify(health_status), status_code

    @app.route('/ready', methods=['GET'])
    def readiness_check():
        """Readiness probe for Kubernetes/Cloud Run."""
        return jsonify({"status": "ready", "timestamp": datetime.now().isoformat()}), 200

    @app.route('/live', methods=['GET'])
    def liveness_check():
        """Liveness probe for Kubernetes/Cloud Run."""
        return jsonify({"status": "alive", "timestamp": datetime.now().isoformat()}), 200

    @app.route('/', methods=['GET'])
    def index():
        """Root endpoint with API information."""
        return jsonify({
            "service": "Email Agent",
            "version": "1.0.0",
            "description": "AI-powered customer service email automation for Pompeii3",
            "endpoints": {
                "/health": "Health check endpoint",
                "/ready": "Readiness probe",
                "/live": "Liveness probe",
                "/process": "Process emails (POST)"
            }
        }), 200

    @app.route('/process', methods=['POST'])
    def process_emails_endpoint():
        """
        HTTP endpoint to trigger email processing.
        Can be called manually or by a scheduler.
        """
        logger = setup_logging()
        logger.info("HTTP /process endpoint called")
        
        try:
            # Initialize services
            service = get_gmail_service(logger)
            if not service:
                return jsonify({"error": "Failed to initialize Gmail service"}), 500
            
            storage_client = storage.Client()
            
            # Initialize vector DB with error handling
            vector_db = None
            try:
                vector_db = initialize_vector_db_from_gcs(
                    bucket_name=VECTOR_DB_BUCKET,
                    prefix=VECTOR_DB_PREFIX,
                    local_dir=LOCAL_DB_DIR,
                    logger=logger
                )
            except Exception as e:
                logger.warning(f"Vector DB initialization failed: {e}")
            
            # Fetch recent emails
            recent_ids = fetch_recent_emails(service, days=1, logger=logger, max_results=10)
            
            processed_count = 0
            errors = []
            
            for msg_id in recent_ids:
                try:
                    if process_and_respond_to_email(msg_id, service, vector_db, storage_client, logger):
                        processed_count += 1
                except Exception as e:
                    errors.append({"message_id": msg_id, "error": str(e)})
                    logger.error(f"Error processing {msg_id}: {e}")
            
            return jsonify({
                "status": "completed",
                "processed": processed_count,
                "total": len(recent_ids),
                "errors": errors if errors else None
            }), 200
            
        except Exception as e:
            logger.error(f"Error in /process endpoint: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error"}), 500


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Starting Email Agent service...")
    
    if FLASK_AVAILABLE:
        # Get port from environment variable (Cloud Run sets this)
        port = int(os.environ.get("PORT", 8080))
        logger.info(f"Starting Flask server on port {port}")
        
        # Run Flask app
        app.run(
            host="0.0.0.0",
            port=port,
            debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true"
        )
    else:
        logger.error("Flask not available. Install Flask to run as HTTP service.")
        logger.info("This service is designed to run as a Cloud Function or Cloud Run service.")
