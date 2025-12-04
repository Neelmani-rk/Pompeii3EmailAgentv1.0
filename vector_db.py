import logging
import os
import time
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions
from google.cloud import storage

# Constants for GCS bucket and vector DB directory
VECTOR_DB_BUCKET = "dotted-electron-447414-m1-vector-db"
VECTOR_DB_PREFIX = "vector_db_export/"  # GCS folder containing chroma.sqlite3 and .bin files
LOCAL_VECTOR_DB_DIR = "./local_vector_db/vector_db_export"  # Local folder to store downloaded vector DB

class VectorKnowledgeBase:
    def __init__(self, persist_dir: Optional[str] = None, collection_name: str = "pompeii3_docs", logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.collection_name = collection_name
        self.is_initialized = False
        self.collection: Optional[chromadb.api.models.Collection.Collection] = None

        try:
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            self.logger.info("Using DefaultEmbeddingFunction for ChromaDB.")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding function: {e}", exc_info=True)
            self.embedding_function = None
            return

        try:
            if persist_dir and os.path.exists(persist_dir):
                self.logger.info(f"Initializing persistent ChromaDB client from directory: {persist_dir}")
                self.client = chromadb.PersistentClient(path=persist_dir)
            elif persist_dir:
                self.logger.info(f"Initializing persistent ChromaDB client. Directory will be created: {persist_dir}")
                os.makedirs(persist_dir, exist_ok=True)
                self.client = chromadb.PersistentClient(path=persist_dir)
            else:
                self.logger.info("Initializing in-memory ChromaDB client.")
                self.client = chromadb.Client()

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            self.is_initialized = True
            self.logger.info(f"Successfully loaded/created ChromaDB collection '{self.collection_name}'. Count: {self.collection.count()}")

        except Exception as e:
            self.logger.error(f"Error initializing ChromaDB client or collection '{self.collection_name}': {e}", exc_info=True)
            self.is_initialized = False

    def is_ready(self) -> bool:
        return self.is_initialized and self.collection is not None

    def search(self, query_text: str, n_results: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.is_ready() or self.collection is None:
            self.logger.warning("Vector database is not ready for search.")
            return []
        if not query_text:
            self.logger.warning("Search query is empty.")
            return []

        try:
            self.logger.info(f"Performing vector search for query: '{query_text[:100]}...' (n_results={n_results}, filter={filter_metadata})")
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=filter_metadata,
                include=['documents', 'metadatas', 'distances']
            )

            formatted_results: List[Dict[str, Any]] = []
            if results and results.get('documents') and results.get('metadatas') and results.get('distances'):
                docs = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]

                for doc, meta, dist in zip(docs, metadatas, distances):
                    relevance = 1.0 - dist if dist is not None else 0.0
                    formatted_results.append({
                        'text': doc,
                        'source': meta.get('source', 'Unknown'),
                        'context': meta.get('paragraph_context', ''),
                        'relevance': relevance,
                        'metadata': meta
                    })
            self.logger.info(f"Vector search for '{query_text[:50]}...' returned {len(formatted_results)} results.")
            return formatted_results
        except Exception as e:
            self.logger.error(f"Error during vector search: {e}", exc_info=True)
            return []

def initialize_vector_db_from_gcs(bucket_name=VECTOR_DB_BUCKET, prefix=VECTOR_DB_PREFIX, local_dir=LOCAL_VECTOR_DB_DIR, logger=None):
    try:
        if logger:
            logger.info(f"Loading vector database from gs://{bucket_name}/{prefix}")
        os.makedirs(local_dir, exist_ok=True)

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)

        db_files_found = False
        for blob in blobs:
            relative_path = os.path.relpath(blob.name, prefix)
            local_file_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            blob.download_to_filename(local_file_path)
            db_files_found = True
            if logger:
                logger.debug(f"Downloaded {blob.name} to {local_file_path}")

        if not db_files_found:
            if logger:
                logger.error(f"No vector database files found in gs://{bucket_name}/{prefix}")
            return None

        if logger:
            logger.info(f"Vector database downloaded to {local_dir}")

        vector_db = VectorKnowledgeBase(persist_dir=local_dir, logger=logger)
        if vector_db.is_ready():
            if logger:
                logger.info("Vector database successfully initialized from GCS")
            return vector_db
        else:
            if logger:
                logger.error("Vector database initialization failed after downloading from GCS")
            return None

    except Exception as e:
        if logger:
            logger.error(f"Error initializing vector database from GCS: {str(e)}")
            logger.debug("Traceback:", exc_info=True)
        return None
