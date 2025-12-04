import json
import logging
from dataclasses import dataclass
from typing import List, Set, Tuple, Optional
import google.generativeai as genai
# Assuming utils.py contains clean_extracted_text
# If utils.py is in a different package structure, adjust the import path accordingly.
from utils import clean_extracted_text
from config import GEMINI_API_KEY
@dataclass
class ExtractedEntity:
    type: str
    value: str
    confidence: float

class EntityExtractor:
    def __init__(self, api_key: str, model_name: str = "gemini-3-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Define the entity types we want to extract with descriptions
        self.entity_types = {
            "ORDER_NUMBER": "Order numbers, order IDs, confirmation numbers (4-12 alphanumeric characters)",
            "PURCHASE_ORDER_NUMBER": "Purchase order numbers, PO numbers (10-20 alphanumeric characters)",
            "TRACKING_NUMBER": "Shipping tracking numbers from any carrier (10-34 characters)",
            "PRODUCT_IDENTIFIER": "SKU codes, item numbers, model numbers, product codes",
            "CUSTOMER_ACCOUNT": "Customer account numbers, user IDs (6-20 alphanumeric characters)",
            "PAYMENT_METHOD": "Last 4 digits of credit cards, payment method identifiers",
            "COUPON_CODE": "Discount codes, promo codes, coupon codes",
            "LOYALTY_POINTS": "Loyalty points, reward points (numeric values)",
            "DELIVERY_CARRIER": "Shipping carriers (FedEx, UPS, USPS, DHL, Amazon Logistics, etc.)",
            "QUANTITY": "Product quantities, item counts",
            "REFUND_EXCHANGE_REASON": "Reasons for returns, refunds, or exchanges"
        }

    def _create_extraction_prompt(self, text: str) -> str:
        """Create a structured prompt for Gemini to extract entities"""
        
        entity_descriptions = "\n".join([
            f"- {entity_type}: {description}" 
            for entity_type, description in self.entity_types.items()
        ])
        
        prompt = f"""
You are an expert entity extraction system. Extract business-relevant entities from the following text.

ENTITY TYPES TO EXTRACT:
{entity_descriptions}

EXTRACTION RULES:
1. Only extract entities that clearly match the defined types
2. Ensure extracted values are clean and properly formatted
3. For ORDER_NUMBER: Extract 4-12 character alphanumeric codes
4. For PURCHASE_ORDER_NUMBER: Extract 10-20 character codes, often labeled as "PO"
5. For TRACKING_NUMBER: Extract 10-34 character tracking codes from any carrier
6. For PRODUCT_IDENTIFIER: Extract SKU, item numbers, model numbers
7. For QUANTITY: Extract only numeric values representing item counts
8. For LOYALTY_POINTS: Extract only the numeric point values
9. For PAYMENT_METHOD: Extract only last 4 digits of payment cards
10. For DELIVERY_CARRIER: Extract carrier names (FedEx, UPS, USPS, DHL, etc.)
11. For REFUND_EXCHANGE_REASON: Extract the reason text for returns/exchanges

CONFIDENCE SCORING:
- 0.95: Very confident, clear pattern match
- 0.85: Confident, good contextual match
- 0.75: Moderately confident, some ambiguity
- 0.65: Low confidence, uncertain match

OUTPUT FORMAT:
Return a JSON array of objects with this exact structure:
[
  {{
    "type": "ENTITY_TYPE",
    "value": "extracted_value",
    "confidence": 0.95
  }}
]

If no entities are found, return an empty array: []

TEXT TO ANALYZE:
{text}

RESPONSE (JSON only):
"""
        return prompt

    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract all specified entities from the given text using Gemini AI."""
        entities: List[ExtractedEntity] = []
        if not text or not text.strip():
            return entities

        try:
            # Create the extraction prompt
            prompt = self._create_extraction_prompt(text)
            
            # Generate response from Gemini
            response = self.model.generate_content(prompt)
            
            if not response.text:
                return entities
            
            # Parse the JSON response
            entities_data = self._parse_gemini_response(response.text)
            
            # Convert to ExtractedEntity objects
            for entity_dict in entities_data:
                if self._validate_entity(entity_dict):
                    entities.append(ExtractedEntity(
                        type=entity_dict["type"],
                        value=entity_dict["value"],
                        confidence=entity_dict["confidence"]
                    ))
            
            # Deduplicate entities
            entities = self._deduplicate_entities(entities)
            
        except Exception as e:
            # Log error but don't raise to maintain original function behavior
            print(f"Error during Gemini entity extraction: {str(e)}")
            
        return entities

    def _parse_gemini_response(self, response_text: str) -> List[dict]:
        """Parse JSON response from Gemini"""
        try:
            # Clean the response text - remove any markdown formatting
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # Parse JSON
            entities_data = json.loads(cleaned_response)
            
            if not isinstance(entities_data, list):
                entities_data = [entities_data] if entities_data else []
            
            return entities_data
            
        except json.JSONDecodeError:
            return []

    def _validate_entity(self, entity_dict: dict) -> bool:
        """Validate entity dictionary structure and content"""
        required_fields = ["type", "value", "confidence"]
        
        # Check required fields
        if not all(field in entity_dict for field in required_fields):
            return False
        
        # Check entity type is valid
        if entity_dict["type"] not in self.entity_types:
            return False
        
        # Check confidence is valid
        confidence = entity_dict["confidence"]
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            return False
        
        # Check value is not empty
        if not entity_dict["value"] or not str(entity_dict["value"]).strip():
            return False
        
        # Type-specific validation
        entity_type = entity_dict["type"]
        value = str(entity_dict["value"]).strip()
        
        if entity_type == "ORDER_NUMBER" and not (4 <= len(value) <= 12):
            return False
        elif entity_type == "PURCHASE_ORDER_NUMBER" and not (10 <= len(value) <= 20):
            return False
        elif entity_type == "TRACKING_NUMBER" and not (10 <= len(value) <= 34):
            return False
        elif entity_type == "PAYMENT_METHOD" and len(value) != 4:
            return False
        elif entity_type in ["QUANTITY", "LOYALTY_POINTS"]:
            try:
                int(value)
            except ValueError:
                return False
        
        return True

    def _extract_pattern(self, text: str, patterns: List[str], entity_type: str, entities: List[ExtractedEntity]) -> None:
        """
        Legacy method maintained for compatibility - now uses Gemini internally
        This method is kept to maintain the original interface but redirects to Gemini extraction
        """
        # This method is now a no-op since we use Gemini for all extraction
        # Keeping it for backward compatibility
        pass

    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities, keeping the one with highest confidence."""
        if not entities:
            return []
        
        # Group by type and value (case-insensitive)
        entity_groups = {}
        for entity in entities:
            key = (entity.type, entity.value.lower())
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)
        
        # Keep the highest confidence entity from each group
        unique_entities = []
        for group in entity_groups.values():
            best_entity = max(group, key=lambda e: e.confidence)
            unique_entities.append(best_entity)
        
        return unique_entities


def extract_entities_from_image_text(text: str, logger: logging.Logger = None, api_key: str = GEMINI_API_KEY, model_name: str = "gemini-3-pro") -> List[ExtractedEntity]:
    """
    Extracts business-relevant entities from text obtained via OCR from an image using Gemini AI.
    
    Args:
        text: OCR'd text from image
        logger: Optional logger for debugging
        api_key: Google AI API key (required for Gemini)
        model_name: Gemini model to use
    """
    if not text:
        return []
    
    if not api_key:
        if logger:
            logger.error("API key is required for Gemini entity extraction")
        return []
    
    if logger:
        logger.info(f"Starting entity extraction from image text using Gemini (length: {len(text)} chars).")

    # Clean the OCR'd text first
    cleaned_text = clean_extracted_text(text)
    if logger:
        if cleaned_text != text:
            logger.info(f"Cleaned image text. Original length: {len(text)}, Cleaned length: {len(cleaned_text)}.")
            logger.debug(f"Original text preview (first 100): {text[:100]}...")
            logger.debug(f"Cleaned text preview (first 100): {cleaned_text[:100]}...")
        else:
            logger.info("Image text did not require further cleaning by clean_extracted_text.")

    if not cleaned_text.strip():
        if logger:
            logger.info("Cleaned image text is empty. No entities to extract.")
        return []

    # Use Gemini-powered EntityExtractor
    entity_extractor = EntityExtractor(api_key, model_name)
    
    # Extract entities using Gemini
    all_extracted_entities = entity_extractor.extract_entities(cleaned_text)
    
    # Filter for relevant entity types
    relevant_entity_types = [
        "ORDER_NUMBER", "TRACKING_NUMBER", "PRODUCT_IDENTIFIER",
        "PURCHASE_ORDER_NUMBER", "QUANTITY", "DELIVERY_CARRIER", 
        "REFUND_EXCHANGE_REASON", "CUSTOMER_ACCOUNT", "PAYMENT_METHOD",
        "COUPON_CODE", "LOYALTY_POINTS"
    ]
    
    filtered_entities = [e for e in all_extracted_entities if e.type in relevant_entity_types]

    if logger:
        logger.info(f"Extracted {len(filtered_entities)} relevant business entities from image text using Gemini.")
        for entity in filtered_entities:
            logger.debug(f"Relevant image entity: Type={entity.type}, Value='{entity.value}', Conf={entity.confidence:.2f}")
            
    return filtered_entities


def extract_entities_with_patterns(
    text: str, 
    patterns: List[str], 
    entity_type: str, 
    logger: logging.Logger = None,
    api_key: str = GEMINI_API_KEY,
    model_name: str = "gemini-3-pro"
) -> List[ExtractedEntity]:
    """
    Extract entities using Gemini AI instead of regex patterns.
    The patterns parameter is kept for compatibility but not used.
    
    Args:
        text: Input text
        patterns: Legacy parameter (not used with Gemini)
        entity_type: Type of entity to extract
        logger: Optional logger
        api_key: Google AI API key (required)
        model_name: Gemini model to use
    """
    entities: List[ExtractedEntity] = []
    if not text or not text.strip():
        return entities
    
    if not api_key:
        if logger:
            logger.error("API key is required for Gemini entity extraction")
        return entities

    try:
        # Create a focused extractor for the specific entity type
        entity_extractor = EntityExtractor(api_key, model_name)
        
        # Filter to only the requested entity type
        original_types = entity_extractor.entity_types.copy()
        if entity_type in original_types:
            entity_extractor.entity_types = {entity_type: original_types[entity_type]}
        else:
            # If entity_type is not in our predefined types, create a generic description
            entity_extractor.entity_types = {entity_type: f"Extract {entity_type} entities from the text"}
        
        # Extract entities using Gemini
        extracted_entities = entity_extractor.extract_entities(text)
        
        # Filter to only the requested type
        entities = [e for e in extracted_entities if e.type == entity_type]
        
        if logger:
            for entity in entities:
                logger.info(f"Gemini extractor found {entity_type}: '{entity.value}' with confidence: {entity.confidence:.2f}")
    
    except Exception as e:
        if logger:
            logger.error(f"Error in Gemini entity extraction: {str(e)}")
    
    return entities


def extract_text_from_image_with_gemini(image_bytes: bytes, api_key: str, logger=None) -> str:
    """
    Extracts all readable text from an image using Gemini 3 Pro.
    Args:
        image_bytes: The image file content in bytes
        api_key: Gemini API key
        logger: Optional logger
    Returns:
        Extracted text as string (empty if none found)
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3-pro")
        
        # Determine MIME type based on image header
        mime_type = "image/jpeg"  # Default
        if image_bytes.startswith(b'\x89PNG'):
            mime_type = "image/png"
        elif image_bytes.startswith(b'GIF'):
            mime_type = "image/gif"
        elif image_bytes.startswith(b'RIFF') and b'WEBP' in image_bytes[:12]:
            mime_type = "image/webp"
        
        prompt = """Extract all readable text from this image. 
        Focus on:
        - Order numbers, tracking numbers, purchase order numbers
        - Product names, SKUs, quantities
        - Customer information, addresses
        - Payment details, shipping information
        - Any business-relevant text
        
        Output only the extracted text, no explanations or formatting."""
        
        response = model.generate_content([
            {"mime_type": mime_type, "data": image_bytes},
            {"text": prompt}
        ])
        
        text = response.text.strip() if response and response.text else ""
        if logger:
            logger.info(f"Gemini OCR extracted text length: {len(text)} characters")
            if text:
                logger.debug(f"OCR text preview: {text[:200]}...")
        return text
        
    except Exception as e:
        if logger:
            logger.error(f"Gemini OCR failed: {e}")
        return ""

def extract_entities_from_image_bytes(image_bytes: bytes, api_key: str, logger=None) -> List[ExtractedEntity]:
    """
    Complete pipeline: Extract text from image using Gemini OCR, then extract entities.
    Args:
        image_bytes: The image file content in bytes
        api_key: Gemini API key
        logger: Optional logger
    Returns:
        List of ExtractedEntity objects
    """
    # Step 1: OCR with Gemini
    ocr_text = extract_text_from_image_with_gemini(image_bytes, api_key, logger)
    if not ocr_text.strip():
        if logger:
            logger.info("No text extracted from image")
        return []
    
    # Step 2: Clean the text
    cleaned_text = clean_extracted_text(ocr_text)
    if logger:
        logger.info(f"Cleaned OCR text length: {len(cleaned_text)} characters")
    
    # Step 3: Extract entities using Gemini
    entity_extractor = EntityExtractor(api_key=api_key, model_name="gemini-3-pro")
    entities = entity_extractor.extract_entities(cleaned_text)
    
    if logger:
        logger.info(f"Extracted {len(entities)} entities from image")
        for entity in entities:
            logger.debug(f"Image entity: {entity.type} = '{entity.value}' (conf: {entity.confidence:.2f})")
    
    return entities
