from google.cloud import storage
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

class ConversationStorage:
    def __init__(self, storage_client: storage.Client):
        self.storage_client = storage_client
        self.responses_bucket = "dotted-electron-447414-m1-responses"
        self.payloads_bucket = "dotted-electron-447414-m1-history-ids"
        self.context_bucket = "dotted-electron-447414-m1-context-email"
        self.logger = logging.getLogger(__name__)
    
    def save_structured_conversation(self, conversation_data: Dict[str, Any], thread_id: str) -> bool:
        """Save structured conversation to the payloads bucket"""
        try:
            bucket = self.storage_client.bucket(self.payloads_bucket)
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
            
            self.logger.info(f"Saved structured conversation to payloads bucket: {blob_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving structured conversation: {e}")
            return False
    
    def retrieve_conversation_history(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the latest structured conversation for a thread"""
        try:
            bucket = self.storage_client.bucket(self.payloads_bucket)
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
    
    def list_conversations_for_thread(self, thread_id: str) -> List[Dict[str, Any]]:
        """List all conversation files for a specific thread"""
        try:
            bucket = self.storage_client.bucket(self.payloads_bucket)
            blobs = bucket.list_blobs(prefix=f"structured_conversations/{thread_id}_")
            
            conversations = []
            for blob in blobs:
                try:
                    content = blob.download_as_text()
                    conversation_data = json.loads(content)
                    conversation_data['blob_name'] = blob.name
                    conversation_data['created_time'] = blob.time_created.isoformat()
                    conversations.append(conversation_data)
                except Exception as e:
                    self.logger.warning(f"Error loading conversation from {blob.name}: {e}")
            
            # Sort by creation time, newest first
            conversations.sort(key=lambda x: x.get('created_time', ''), reverse=True)
            return conversations
            
        except Exception as e:
            self.logger.error(f"Error listing conversations for thread {thread_id}: {e}")
            return []
    
    def cleanup_old_conversations(self, days_to_keep: int = 30) -> int:
        """Clean up old conversation files"""
        try:
            bucket = self.storage_client.bucket(self.payloads_bucket)
            blobs = bucket.list_blobs(prefix="structured_conversations/")
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            deleted_count = 0
            
            for blob in blobs:
                if blob.time_created < cutoff_date:
                    try:
                        blob.delete()
                        deleted_count += 1
                        self.logger.info(f"Deleted old conversation: {blob.name}")
                    except Exception as e:
                        self.logger.warning(f"Error deleting {blob.name}: {e}")
            
            self.logger.info(f"Cleaned up {deleted_count} old conversation files")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return 0
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about stored conversations"""
        try:
            bucket = self.storage_client.bucket(self.payloads_bucket)
            blobs = bucket.list_blobs(prefix="structured_conversations/")
            
            stats = {
                'total_conversations': 0,
                'unique_threads': set(),
                'oldest_conversation': None,
                'newest_conversation': None,
                'total_size_bytes': 0
            }
            
            for blob in blobs:
                stats['total_conversations'] += 1
                stats['total_size_bytes'] += blob.size
                
                # Extract thread_id from blob name
                thread_id = blob.name.split('/')[-1].split('_')[0]
                stats['unique_threads'].add(thread_id)
                
                if stats['oldest_conversation'] is None or blob.time_created < stats['oldest_conversation']:
                    stats['oldest_conversation'] = blob.time_created
                
                if stats['newest_conversation'] is None or blob.time_created > stats['newest_conversation']:
                    stats['newest_conversation'] = blob.time_created
            
            stats['unique_threads'] = len(stats['unique_threads'])
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting conversation stats: {e}")
            return {}
