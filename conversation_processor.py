import json
import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from google.cloud import storage
import google.generativeai as genai
from bs4 import BeautifulSoup

class ConversationProcessor:
    def __init__(self, gemini_api_key: str, storage_client: storage.Client):
        self.gemini_api_key = gemini_api_key
        self.storage_client = storage_client
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-3-pro-preview-preview')
        self.logger = logging.getLogger(__name__)
        
    def process_conversation_history(self, email_text: str, thread_id: str) -> Dict[str, Any]:
        """Process email conversation history into structured JSON format"""
        
        if not email_text or not thread_id:
            self.logger.warning("Empty email text or thread_id provided")
            return self._create_fallback_structure()
        
        try:
            # Clean HTML tags while preserving structure
            cleaned_text = self._clean_html_preserve_structure(email_text)
            
            if not cleaned_text or len(cleaned_text.strip()) < 20:
                self.logger.warning("Cleaned text is too short for meaningful conversation processing")
                return self._create_fallback_structure()
            
            # Use Gemini 2.5 Pro to structure the conversation
            structured_conversation = self._structure_with_gemini(cleaned_text)
            
            # Save structured conversation to GCS
            self._save_conversation_to_gcs(structured_conversation, thread_id)
            
            return structured_conversation
            
        except Exception as e:
            self.logger.error(f"Error in process_conversation_history: {e}", exc_info=True)
            return self._create_fallback_structure()
    
    def _clean_html_preserve_structure(self, html_text: str) -> str:
        """Clean HTML tags while maintaining email structure"""
        if not html_text:
            return ""
            
        try:
            # Use BeautifulSoup for robust HTML cleaning
            soup = BeautifulSoup(html_text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Remove other unwanted elements
            for element in soup(["meta", "link", "title"]):
                element.decompose()
            
            # Get text and preserve line breaks
            text = soup.get_text()
            
            # Clean up whitespace while preserving structure
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            
            # Remove excessive newlines (more than 2)
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Error cleaning HTML: {e}")
            # Fallback to simple regex cleaning
            text = re.sub(r'<[^>]+>', '', html_text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
    
    def _structure_with_gemini(self, conversation_text: str) -> Dict[str, Any]:
        """Use Gemini 2.5 Pro to structure conversation into JSON"""
        
        system_prompt = """
        You are an expert email conversation parser. Parse the given email thread into a structured JSON format.
        
        Requirements:
        1. Identify each individual email in the conversation
        2. Determine if each email is from customer or support team (use email domains, signatures, and content patterns)
        3. Extract timestamp, sender, and content for each email
        4. Order emails chronologically (oldest first)
        5. Clean any remaining HTML artifacts
        6. Preserve the actual content while removing formatting
        7. Handle forwarded messages and replies appropriately
        8. Extract meaningful subject lines when available
        
        Support team indicators:
        - Emails from @pompeii3.com, @shipstation.com, @fedex.com, @ups.com
        - Professional signatures with company information
        - Tracking notifications and order confirmations
        
        Customer indicators:
        - Personal email domains (@gmail.com, @yahoo.com, @outlook.com, etc.)
        - Informal language and personal inquiries
        - Questions about orders, products, or services
        
        Return a JSON structure with this schema:
        {
            "conversation_history": [
                {
                    "timestamp": "ISO format timestamp or extracted date",
                    "sender_type": "customer" or "support",
                    "sender_name": "Name of sender",
                    "sender_email": "email@domain.com",
                    "content": "Clean email content without HTML",
                    "message_type": "original" or "reply" or "forward",
                    "subject": "Email subject if available"
                }
            ],
            "conversation_summary": {
                "total_messages": number,
                "customer_messages": number,
                "support_messages": number,
                "date_range": "start_date to end_date",
                "thread_topic": "Main topic of conversation"
            }
        }
        """
        
        user_prompt = f"""
        Parse this email conversation thread:
        
        {conversation_text}
        
        Please structure this into the required JSON format, ensuring chronological order and proper sender identification.
        Pay special attention to email signatures, domains, and content patterns to correctly identify customer vs support messages.
        """
        
        try:
            response = self.model.generate_content(
                [system_prompt, user_prompt],
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    max_output_tokens=4000,
                    top_p=0.8
                )
            )
            
            if response and response.text:
                parsed_data = json.loads(response.text)
                
                # Validate the structure
                if self._validate_conversation_structure(parsed_data):
                    return parsed_data
                else:
                    self.logger.warning("Invalid conversation structure returned by Gemini")
                    return self._create_fallback_structure()
            else:
                self.logger.error("Empty response from Gemini API")
                return self._create_fallback_structure()
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error from Gemini response: {e}")
            return self._create_fallback_structure()
        except Exception as e:
            self.logger.error(f"Error processing conversation with Gemini: {e}")
            return self._create_fallback_structure()
    
    def _validate_conversation_structure(self, data: Dict[str, Any]) -> bool:
        """Validate the structure of conversation data"""
        try:
            if not isinstance(data, dict):
                return False
            
            if 'conversation_history' not in data or not isinstance(data['conversation_history'], list):
                return False
            
            if 'conversation_summary' not in data or not isinstance(data['conversation_summary'], dict):
                return False
            
            # Validate each conversation entry
            for entry in data['conversation_history']:
                if not isinstance(entry, dict):
                    return False
                
                required_fields = ['sender_type', 'content']
                if not all(field in entry for field in required_fields):
                    return False
                
                if entry['sender_type'] not in ['customer', 'support']:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating conversation structure: {e}")
            return False
    
    def _create_fallback_structure(self) -> Dict[str, Any]:
        """Create a fallback structure when processing fails"""
        return {
            "conversation_history": [],
            "conversation_summary": {
                "total_messages": 0,
                "customer_messages": 0,
                "support_messages": 0,
                "date_range": "unknown",
                "thread_topic": "unknown"
            },
            "processing_status": "failed",
            "timestamp": datetime.now().isoformat()
        }
    
    def _save_conversation_to_gcs(self, conversation_data: Dict[str, Any], thread_id: str):
        """Save structured conversation to GCS"""
        try:
            bucket = self.storage_client.bucket("dotted-electron-447414-m1-context-email")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_name = f"structured_conversations/{thread_id}_{timestamp}.json"
            
            # Add metadata
            conversation_data['saved_timestamp'] = datetime.now().isoformat()
            conversation_data['thread_id'] = thread_id
            
            blob = bucket.blob(blob_name)
            blob.upload_from_string(
                json.dumps(conversation_data, indent=2, default=str),
                content_type='application/json'
            )
            
            self.logger.info(f"Saved structured conversation to GCS: {blob_name}")
            
        except Exception as e:
            self.logger.error(f"Error saving conversation to GCS: {e}")
    
    def retrieve_conversation_history(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the latest structured conversation for a thread"""
        try:
            bucket = self.storage_client.bucket("dotted-electron-447414-m1-context-email")
            blobs = bucket.list_blobs(prefix=f"structured_conversations/{thread_id}_")
            
            # Get the most recent conversation file
            latest_blob = None
            for blob in blobs:
                if latest_blob is None or blob.time_created > latest_blob.time_created:
                    latest_blob = blob
            
            if latest_blob:
                content = latest_blob.download_as_text()
                return json.loads(content)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving conversation history: {e}")
            return None
