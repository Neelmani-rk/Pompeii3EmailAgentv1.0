# Pompeii3 Email Agent Documentation

## 1. System Overview
The Pompeii3 Email Agent is an AI-powered customer service automation system designed to handle email inquiries for a jewelry e-commerce brand. It processes incoming emails, understands customer intent, retrieves relevant order and product information, and generates empathetic, accurate responses. The system is built on Google Cloud Platform (GCP) and leverages Gemini AI models for natural language understanding and generation.

## 2. Core Workflow
The system follows a linear processing pipeline for each incoming email:

1.  **Trigger & Ingestion:**
    *   **Pub/Sub Trigger:** Listens for Gmail push notifications via Google Cloud Pub/Sub (`main.py`).
    *   **Email Fetching:** Retrieves new emails using the Gmail API (`gmail_helpers.py`).
    *   **Deduplication:** Checks if the email or thread has already been processed to prevent duplicate responses.

2.  **Processing & Analysis:**
    *   **Content Extraction:** Parses email body, separating current message from thread history (`utils.py`, `gmail_helpers.py`).
    *   **Conversation History:** Structures the entire email thread for context (`conversation_processor.py`).
    *   **Intent Detection:** Classifies the customer's intent (e.g., "Order Status", "Return Request") using Gemini AI (`api_clients.py`).
    *   **Entity Extraction:** Identifies key entities like Order Numbers, Tracking Numbers, and Customer Names from text and images (`entity_extractor.py`).
    *   **Sentiment Analysis:** Analyzes the emotional tone of the email (`api_clients.py`).

3.  **Data Retrieval:**
    *   **Order Lookup:** Fetches order details from ShipStation and Pompeii3 APIs using extracted entities (`order_processing.py`, `api_clients.py`).
    *   **Tracking Info:** Retrieves real-time tracking updates from FedEx/ShipStation APIs.
    *   **Knowledge Base:** Searches a vector database (ChromaDB) for relevant policy or product information (`vector_db.py`).

4.  **Response Generation:**
    *   **Payload Construction:** Aggregates all gathered context (intent, entities, order data, history) into a structured payload (`response_generation.py`).
    *   **Draft Generation:** Uses Gemini AI with intent-specific system prompts to generate a personalized response draft.
    *   **Quality Validation:** Evaluates the generated draft against quality standards (helpfulness, empathy, accuracy) (`email_quality_validator.py`).

5.  **Action & Analytics:**
    *   **Draft Saving:** Saves the approved response as a Gmail draft for human review (`gmail_helpers.py`).
    *   **Analytics Logging:** Logs performance metrics, quality scores, and knowledge gaps to Google Sheets (`response_analytics.py`).
    *   **State Management:** Marks the email as processed in Cloud Storage to maintain state (`conversation_storage.py`).

## 3. Component Breakdown

### Core Logic
*   **`main.py`**: The entry point for the Cloud Function. Orchestrates the entire flow from Pub/Sub event to draft creation. Handles initialization of services and high-level error catching.
*   **`config.py`**: Central configuration file managing environment variables, API keys, bucket names, and system constants.
*   **`utils.py`**: Utility functions for text cleaning, email parsing, logging setup, and regex pattern matching.

### Email & Gmail Integration
*   **`gmail_helpers.py`**: Handles all interactions with the Gmail API. Includes functions for fetching messages, parsing MIME types, handling attachments, and creating drafts.
*   **`email_quality_validator.py`**: An AI-driven module that critiques generated drafts. It checks for tone, accuracy, and completeness, providing improvement suggestions and tracking "knowledge gaps" where the AI lacked info.

### Processing & Logic
*   **`conversation_processor.py`**: Uses Gemini to parse unstructured email threads into a structured JSON format, distinguishing between customer and support messages to provide clean context to the LLM.
*   **`entity_extractor.py`**: A robust extraction engine using both Regex and Gemini AI to find business entities (Order #s, Tracking #s) in text and image attachments (OCR).
*   **`order_processing.py`**: The logic layer that coordinates data fetching. It decides which API (ShipStation vs. Pompeii3) to call based on the extracted entities and merges the data into a unified order object.

### Storage & Database
*   **`conversation_storage.py`**: Manages persistence of structured conversation histories in Google Cloud Storage (GCS).
*   **`vector_db.py`**: Manages the ChromaDB vector database. Handles initialization from GCS, embedding generation, and semantic search for retrieving knowledge base articles.
*   **`api_clients.py`**: Contains client classes for external APIs:
    *   `ShipStationAPI`: For order and shipment data.
    *   `FedExAPI`: For detailed tracking events.
    *   `Gemini`: For intent detection and sentiment analysis.
    *   `Pompeii3`: For internal order status and PO lookups.

### Response & Analytics
*   **`response_generation.py`**: Constructs the prompt payload for the LLM and manages the generation call. Contains the library of intent-specific system prompts (e.g., for "Returns", "Shipping").
*   **`response_analytics.py`**: Tracks system performance. Logs metrics like response time, sentiment scores, and quality validation results to Google Sheets for reporting.
*   **`performance_monitor.py`**: A background monitoring tool that tracks system resource usage (CPU, memory) and operation latencies to identify bottlenecks.

### Validation & Deployment
*   **`input_validation.py`**: Provides strict validation for all inputs (email addresses, order numbers, API keys) to ensure security and data integrity.
*   **`error_handler.py`**: A centralized error handling system with retry logic, circuit breakers for external APIs, and fallback mechanisms to ensure system resilience.
*   **`deploy.py`**: Automation script for deploying the system to Google App Engine, including environment validation and secret management.
*   **`test_sheets_connection.py`**: A utility script to verify connectivity and permissions for Google Sheets integration.

## 4. Key Features
*   **Multi-Modal Analysis:** Can read and extract information from both text and image attachments (e.g., screenshots of shipping labels).
*   **Context-Aware:** Understands the full conversation history, not just the latest email.
*   **Hybrid Search:** Combines specific order lookups (SQL-like) with semantic knowledge base search (Vector) for comprehensive answers.
*   **Quality Assurance:** Includes a self-correction loop where an "AI Critic" validates responses before they are saved.
*   **Resilience:** Implements circuit breakers and fallbacks (e.g., if Gemini is down, it generates a template response).