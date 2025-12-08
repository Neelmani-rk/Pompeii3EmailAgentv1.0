import base64
import json
import logging
import re
import requests
from collections import defaultdict
from typing import List, Dict, Any, Optional
import time
import os
from google.cloud import vision
from google.oauth2 import service_account
from dotenv import load_dotenv
load_dotenv()

from config import (
    SHIPSTATION_API_KEY, SHIPSTATION_API_SECRET,
    FEDEX_CLIENT_ID, FEDEX_CLIENT_SECRET,
    GEMINI_API_KEY,
    SERVICE_ACCOUNT_FILE,  # is used by the fallback detect_intent
)
from utils import clean_extracted_text # Used by extract_text_from_image
from input_validation import InputValidator, sanitize_api_response


# --- ShipStation API Client ---
class ShipStationAPI:
    BASE_URL = "https://ssapi.shipstation.com"

    def __init__(self, api_key: str, api_secret: str, logger: Optional[logging.Logger] = None):
        if not api_key or not api_secret:
            raise ValueError("ShipStation API key and secret are required.")
        
        # Validate API credentials
        self.validator = InputValidator(logger)
        try:
            validated_key = self.validator.validate_api_key(api_key, "ShipStation")
            validated_secret = self.validator.validate_api_key(api_secret, "ShipStation Secret")
        except Exception as e:
            raise ValueError(f"Invalid ShipStation API credentials: {e}")
        
        self.auth_token = base64.b64encode(f"{validated_key}:{validated_secret}".encode()).decode()
        self.headers = {
            'Authorization': f'Basic {self.auth_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        self.logger = logger or logging.getLogger(__name__)

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.BASE_URL}{endpoint}"
        try:
            response = requests.request(method, url, headers=self.headers, timeout=15, **kwargs)
            if response.status_code == 401:
                self.logger.error("ShipStation API Authentication failed. Please verify API credentials.")
                raise Exception("ShipStation authentication failed.")
            response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
            
            # Sanitize API response
            raw_data = response.json()
            sanitized_data = sanitize_api_response(raw_data, self.logger)
            return sanitized_data
        except requests.exceptions.RequestException as e:
            self.logger.error(f"ShipStation API request to {url} failed: {e}")
            raise

    def get_order_by_order_number_exact(self, order_number: str) -> Optional[Dict[str, Any]]:
        endpoint = "/orders"
        params = {
            'orderNumber': order_number,
            'pageSize': 1, # Expecting one order for exact match
            'includeShipmentItems': True
        }
        try:
            response_data = self._request('GET', endpoint, params=params)
            if response_data.get('orders'):
                self.logger.info(f"Found order {order_number} in ShipStation via exact match.")
                return response_data['orders'][0]
            self.logger.warning(f"No orders found with exact order number: {order_number} in ShipStation.")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching exact order {order_number} from ShipStation: {e}")
            return None # Explicitly return None on error

    def get_order_by_order_number(self, order_number: str) -> Optional[Dict[str, Any]]:
        """Attempts to find an order, trying exact match then flexible match."""
        exact_match = self.get_order_by_order_number_exact(order_number)
        if exact_match:
            return exact_match

        self.logger.info(f"Exact match failed for {order_number}, trying flexible search.")
        endpoint = "/orders"
        # Try removing leading zeros if it's a numeric string
        flexible_order_number = order_number.lstrip('0') if order_number.isdigit() else order_number
        
        params = {
            'orderNumber': flexible_order_number,
            'pageSize': 5, # Allow a few results for flexible search if needed
            'includeShipmentItems': True
        }
        try:
            response_data = self._request('GET', endpoint, params=params)
            if response_data.get('orders'):
                if len(response_data['orders']) > 1:
                    self.logger.warning(f"Flexible search for '{flexible_order_number}' returned multiple orders. Using the first one.")
                self.logger.info(f"Found order '{flexible_order_number}' in ShipStation via flexible search.")
                return response_data['orders'][0]
            self.logger.warning(f"No orders found with flexible order number: {flexible_order_number} in ShipStation.")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching flexible order {order_number} from ShipStation: {e}")
            return None

    def get_tracking_numbers(self, order_id: int) -> List[Dict[str, Any]]:
        endpoint = "/shipments"
        params = {'orderId': order_id}
        tracking_numbers_list = []
        try:
            response_data = self._request('GET', endpoint, params=params)
            for shipment in response_data.get('shipments', []):
                tracking_number = shipment.get('trackingNumber')
                if tracking_number:
                    tracking_numbers_list.append({
                        'tracking_number': tracking_number,
                        'carrier': shipment.get('carrierCode'),
                        'service': shipment.get('serviceCode'),
                        'ship_date': shipment.get('shipDate') # Added shipDate
                    })
            self.logger.info(f"Found {len(tracking_numbers_list)} tracking numbers for ShipStation order ID {order_id}.")
            return tracking_numbers_list
        except Exception as e:
            self.logger.error(f"Error fetching tracking for ShipStation order ID {order_id}: {e}")
            return []

    def process_order_data(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transforms raw ShipStation order data into a more structured format."""
        if not order_data:
            return {}
        ship_to = order_data.get('shipTo', {})
        processed = {
            'order_id': order_data.get('orderId'),
            'order_number': order_data.get('orderNumber'),
            'order_date': order_data.get('orderDate'),
            'recipient_name': ship_to.get('name'),
            'order_total': order_data.get('orderTotal'),
            'order_status': order_data.get('orderStatus'),
            'ship_by_date': order_data.get('shipByDate'),
            'customer_email': order_data.get('customerEmail'), # Added customer email
            'shipping_details': {
                'address_line1': ship_to.get('street1'),
                'address_line2': ship_to.get('street2'),
                'city': ship_to.get('city'),
                'state': ship_to.get('state'),
                'postal_code': ship_to.get('postalCode'),
                'country': ship_to.get('country')
            },
            'items': [{
                'item_id': item.get('orderItemId'), # Added item ID
                'item_name': item.get('name'),
                'item_sku': item.get('sku'),
                'quantity': item.get('quantity'),
                'unit_price': item.get('unitPrice'),
                'image_url': item.get('imageUrl') # Added image URL
            } for item in order_data.get('items', []) if item] # Ensure item is not None
        }
        self.logger.debug(f"Processed ShipStation order data for order ID: {processed.get('order_id')}")
        return processed

    def get_orders_by_customer_name(self, customer_name: str, sort_by: str = "orderDate", sort_dir: str = "DESC") -> List[Dict[str, Any]]:
        """
        Fetches orders by customer name when order number is not available.
        Returns a list of orders for the specified customer.
        """
        endpoint = "/orders"
        params = {
            'customerName': customer_name,
            'sortBy': sort_by,
            'sortDir': sort_dir,
            'pageSize': 50,  # Get more results for customer search
            'includeShipmentItems': True
        }
        try:
            response_data = self._request('GET', endpoint, params=params)
            orders = response_data.get('orders', [])
            if orders:
                self.logger.info(f"Found {len(orders)} orders for customer: {customer_name}")
                # Process each order using existing method
                processed_orders = []
                for order in orders:
                    processed_order = self.process_order_data(order)
                    if processed_order:
                        processed_orders.append(processed_order)
                return processed_orders
            else:
                self.logger.warning(f"No orders found for customer name: {customer_name}")
                return []
        except Exception as e:
            self.logger.error(f"Error fetching orders for customer {customer_name}: {e}")
            return []

    def get_order_by_customer_or_number(self, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Attempts to find order by order number first, then by customer name if no order number match.
        This is the main method to call when you have either an order number or customer name.
        """
        # First try to get by order number
        order = self.get_order_by_order_number(identifier)
        if order:
            return order
        
        # If no order found by number, try customer name search
        self.logger.info(f"No order found by number '{identifier}', trying customer name search.")
        customer_orders = self.get_orders_by_customer_name(identifier)
        if customer_orders:
            # Return the most recent order (first in the sorted list)
            self.logger.info(f"Found {len(customer_orders)} orders for customer '{identifier}', returning most recent.")
            return customer_orders[0]
        
        self.logger.warning(f"No orders found by order number or customer name: {identifier}")
        return None


# --- FedEx API Client ---
class FedExAPI:
    SANDBOX_URL = "https://apis-sandbox.fedex.com"
    PROD_URL = "https://apis.fedex.com"

    def __init__(self, client_id: str, client_secret: str, sandbox: bool = True, logger: Optional[logging.Logger] = None):
        if not client_id or not client_secret:
            raise ValueError("FedEx client ID and secret are required.")
        
        # Validate API credentials
        self.validator = InputValidator(logger)
        try:
            validated_id = self.validator.validate_api_key(client_id, "FedEx Client ID")
            validated_secret = self.validator.validate_api_key(client_secret, "FedEx Client Secret")
        except Exception as e:
            raise ValueError(f"Invalid FedEx API credentials: {e}")
        
        self.client_id = validated_id
        self.client_secret = validated_secret
        self.base_url = self.SANDBOX_URL if sandbox else self.PROD_URL
        self.access_token: Optional[str] = None
        self.logger = logger or logging.getLogger(__name__)
        self.token_expiry_time: Optional[float] = None # To manage token refresh

    def _get_access_token(self) -> str:
        """Fetches a new access token from FedEx."""
        endpoint = "/oauth/token"
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        try:
            self.logger.info(f"Requesting new FedEx access token from {self.base_url}{endpoint}")
            response = requests.post(f"{self.base_url}{endpoint}", headers=headers, data=data, timeout=10)
            response.raise_for_status()
            token_data = response.json()
            self.access_token = token_data['access_token']
            # Set token expiry time (e.g., expires_in seconds - a small buffer)
            self.token_expiry_time = time.time() + int(token_data.get('expires_in', 3600)) - 60
            self.logger.info("Successfully obtained new FedEx access token.")
            return self.access_token
        except requests.exceptions.RequestException as e:
            self.logger.error(f"FedEx token request failed: {e}")
            raise
        except KeyError:
            self.logger.error("FedEx token response did not contain 'access_token'.")
            raise Exception("Invalid token response from FedEx.")

    def _ensure_token_valid(self) -> None:
        """Ensures the access token is valid, refreshing if necessary."""
        if not self.access_token or (self.token_expiry_time and time.time() >= self.token_expiry_time):
            self.logger.info("FedEx access token is missing or expired. Refreshing.")
            self._get_access_token()

    def get_tracking_info(self, tracking_numbers: List[str]) -> Optional[Dict[str, Any]]:
        self._ensure_token_valid()
        if not self.access_token: # Should not happen if _ensure_token_valid works
             self.logger.error("FedEx access token unavailable after attempting refresh.")
             return None

        endpoint = "/track/v1/trackingnumbers"
        headers = {
            'Content-Type': 'application/json',
            'X-locale': 'en_US',
            'Authorization': f'Bearer {self.access_token}'
        }
        payload = {
            'includeDetailedScans': True,
            'trackingInfo': [{'trackingNumberInfo': {'trackingNumber': tn}} for tn in tracking_numbers]
        }
        try:
            self.logger.info(f"Requesting FedEx tracking info for: {tracking_numbers}")
            response = requests.post(f"{self.base_url}{endpoint}", headers=headers, json=payload, timeout=15)
            if response.status_code == 401: # Unauthorized, likely token expired despite check
                self.logger.warning("FedEx API returned 401, attempting token refresh and retry.")
                self._get_access_token() # Force refresh
                headers['Authorization'] = f'Bearer {self.access_token}'
                response = requests.post(f"{self.base_url}{endpoint}", headers=headers, json=payload, timeout=15)
            
            response.raise_for_status()
            self.logger.info(f"Successfully retrieved FedEx tracking info for: {tracking_numbers}")
            
            # Sanitize API response
            raw_data = response.json()
            sanitized_data = sanitize_api_response(raw_data, self.logger)
            return sanitized_data
        except requests.exceptions.RequestException as e:
            self.logger.error(f"FedEx tracking request for {tracking_numbers} failed: {e}")
            return None

    def _format_location(self, location: Optional[Dict[str, Any]]) -> Dict[str, str]:
        if not location:
            return {}
        return {
            'city': location.get('city', ''),
            'state_province_code': location.get('stateOrProvinceCode', ''), # Corrected key
            'postal_code': location.get('postalCode', ''),
            'country_code': location.get('countryCode', ''), # Corrected key
            'residential': location.get('residential', False) # Added residential flag
        }

    def _format_scans(self, scans: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not scans:
            return []
        formatted_scans = []
        for scan in scans:
            location_details = scan.get('scanLocation', {})
            formatted_scans.append({
                'timestamp': scan.get('date'),
                'event_type': scan.get('eventType'), # Example: PU for Picked Up, DL for Delivered
                'description': scan.get('eventDescription'),
                'location': self._format_location(location_details),
                'exception_code': scan.get('exceptionCode'), # Added exception info
                'exception_description': scan.get('exceptionDescription')
            })
        return formatted_scans

    def process_tracking_data(self, tracking_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not tracking_data or 'output' not in tracking_data or not tracking_data['output'].get('completeTrackResults'):
            self.logger.warning("FedEx tracking data is missing or incomplete for processing.")
            return None

        result = tracking_data['output']['completeTrackResults'][0]
        tracking_number = result.get('trackingNumber')
        
        if not result.get('trackResults'):
            self.logger.warning(f"No trackResults found for FedEx tracking number {tracking_number}")
            return {'tracking_number': tracking_number, 'status': 'No tracking details available'}

        track_result = result['trackResults'][0]
        status_detail = track_result.get('latestStatusDetail', {})
        scan_location = status_detail.get('scanLocation', {})
        
        processed = {
            'tracking_number': tracking_number,
            'carrier_code': track_result.get('trackingNumberInfo', {}).get('carrierCode', 'FDXE'), # Default to FDXE
            'service_description': track_result.get('serviceDetail', {}).get('description'), # Corrected key
            'status_code': status_detail.get('code'), # Example: OC (Order Created), DL (Delivered)
            'status_description': status_detail.get('description'),
            'status_location': self._format_location(scan_location),
            'ship_timestamp': track_result.get('shipDatestamps', {}).get('actual', ''), # Added ship timestamp
            'estimated_delivery_timestamp': track_result.get('deliveryDatestamps', {}).get('estimated', ''), # Added est. delivery
            'actual_delivery_timestamp': track_result.get('deliveryDatestamps', {}).get('actual', ''), # Added actual delivery
            'scans': self._format_scans(track_result.get('scanEvents', [])),
            'package_details': { # Added package details section
                'weight': track_result.get('packageDetails', {}).get('weightAndDimensions', {}).get('weight', [{}])[0].get('value'),
                'weight_units': track_result.get('packageDetails', {}).get('weightAndDimensions', {}).get('weight', [{}])[0].get('unit'),
                # Dimensions can also be added here if needed
            },
            'delivery_address': self._format_location(track_result.get('deliveryAddress', {})) # Added delivery address
        }
        self.logger.debug(f"Processed FedEx tracking data for: {tracking_number}")
        return processed


# --- Pompeii3 API Functions ---
def get_order_status_from_pompeii3(order_id: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Queries the Pompeii3 specific order status API."""
    log = logger or logging.getLogger(__name__)
    log.info(f"Querying Pompeii3 API for order/PO: {order_id}")
    url = f"https://api.pompeii3.com/order_status.php?type=order&query={order_id}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "error" in data and data["error"]: # Check if error field is present and not empty/false
            log.warning(f"Pompeii3 API returned error for {order_id}: {data['error']}")
            return {"error": data["error"], "source": "pompeii3_api"}

        # Extract and rename fields for consistency
        shipping_firstname = data.get("shipping_firstname", "")
        shipping_lastname = data.get("shipping_lastname", "")
        recipient_name = f"{shipping_firstname} {shipping_lastname}".strip()

        processed_data = {
            "order_id_internal": data.get("client_order_identifier"), # Pompeii's internal ID
            "order_number": order_id, # The queried ID, for consistency
            "shipping_method": data.get("shipping_method"),
            "shipping_class": data.get("shipping_class"),
            "tracking_number": data.get("tracking_number"),
            "estimated_ship_date": data.get("estimated_ship_date"),
            "recipient_name": recipient_name if recipient_name else None,
            "items": data.get("items", []), # Assuming items is a list of dicts
            "order_status": data.get("status"),
            "production_percentage": data.get("percentage"), # Renamed for clarity
            "source": "pompeii3_api"
        }

        tracking_link_html = data.get("trackingLink")
        if tracking_link_html:
            href_match = re.search(r'href="([^"]+)"', tracking_link_html)
            processed_data["tracking_link"] = href_match.group(1) if href_match else tracking_link_html # Fallback to raw HTML if no href

        log.info(f"Successfully retrieved and processed data from Pompeii3 API for: {order_id}")
        if processed_data.get("tracking_number"):
            log.info(f"Pompeii3 API returned tracking number: {processed_data['tracking_number']}")
        return processed_data

    except requests.exceptions.Timeout:
        error_msg = f"Pompeii3 API request timed out for {order_id}."
        log.error(error_msg)
        return {"error": error_msg, "source": "pompeii3_api"}
    except requests.exceptions.RequestException as e:
        error_msg = f"Pompeii3 API request failed for {order_id}: {e}"
        log.error(error_msg)
        return {"error": error_msg, "source": "pompeii3_api"}
    except json.JSONDecodeError:
        error_msg = f"Failed to decode JSON response from Pompeii3 API for {order_id}."
        log.error(error_msg)
        return {"error": error_msg, "source": "pompeii3_api"}


def detect_intent_with_gemini(text: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Uses Gemini 3 Pro model to accurately classify customer intent from text.
    """
    log = logger or logging.getLogger(__name__)
    
    if not GEMINI_API_KEY:
        log.warning("Gemini API key not available. Using fallback intent detection.")
        return detect_intent_fallback(text, log)
    
    if not text or len(text.strip()) < 5:
        log.warning("Text too short for Gemini intent classification. Using fallback.")
        return detect_intent_fallback(text, log)

    # Use the correct Gemini 3 Pro model name
    gemini_model_name = "gemini-3-pro-preview-preview"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model_name}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    # Define the 7 intent categories for jewelry e-commerce
    intent_categories = {
        "order-related-query": """Order status, delivery updates, tracking information, shipping address/method changes,
        delivery issues, packaging questions, invoice requests, shipping delays""",
        
        "payment-related-query": """Payment declined, accepted payment methods, payment issues, multiple charges,
        billing discrepancies, sales tax questions, discount/gift card problems, financing options, receipt requests""",
        
        "order-cancel-return-refund-query": """Order cancellation, return/exchange process, return policy questions,
        refund status, damaged/defective items, wrong item received, return shipping issues""",
        
        "product-related-query": """Product specifications (metals, diamonds, gemstones), availability, sizing,
        dimensions, authenticity, warranty, certifications, product photos, durability questions""",
        
        "account-related-query": """Account creation, password reset, login issues, profile updates, order history,
        payment methods management, account deletion, security issues, email preferences""",
        
        "customise-product-query": """Customization options, custom sizing, engraving services, custom jewelry creation,
        customization lead times and costs, custom item return policies""",
        
        "general-query": """Website technical issues, customer service contact, business hours, wholesale inquiries,
        feedback, insurance recommendations, website errors, general support questions"""
    }

    system_prompt = f"""
    You are an expert customer service intent classifier for a jewelry e-commerce business. Analyze the customer message and classify it into one of these 7 categories:

    1. order-related-query: {intent_categories['order-related-query']}
    
    2. payment-related-query: {intent_categories['payment-related-query']}
    
    3. order-cancel-return-refund-query: {intent_categories['order-cancel-return-refund-query']}
    
    4. product-related-query: {intent_categories['product-related-query']}
    
    5. account-related-query: {intent_categories['account-related-query']}
    
    6. customise-product-query: {intent_categories['customise-product-query']}
    
    7. general-query: {intent_categories['general-query']}

    Respond ONLY with a JSON object in this exact format:
    {{
        "primary_intent": "intent_name",
        "confidence": 0.95,
        "reasoning": "brief explanation",
        "secondary_intents": [
            {{"intent": "intent_name", "confidence": 0.3}}
        ]
    }}

    Rules:
    - confidence should be between 0.1 and 1.0
    - primary_intent must be one of the 7 defined categories above
    - reasoning should be 1-2 sentences explaining the classification
    - secondary_intents are optional, include only if relevant (max 2)
    - If uncertain, use "general-query" with lower confidence
    - Focus on the main customer need or request in the message
    """

    prompt_text = f"""
    Customer Message: "{text.strip()}"
    
    Classify this jewelry customer's message intent:
    """

    api_payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt_text}]}
        ],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "temperature": 0.1,  # Low temperature for consistent classification
            "maxOutputTokens": 500,
            "topP": 0.8,
            "topK": 10
        }
    }

    log.info(f"Classifying jewelry customer intent using Gemini 3 Pro for text: {text[:100]}...")
    
    try:
        response = requests.post(url, headers=headers, json=api_payload, timeout=15)
        response.raise_for_status()
        
        response_data = response.json()
        
        if (response_data.get("candidates") and
            response_data["candidates"][0].get("content") and
            response_data["candidates"][0]["content"].get("parts") and
            response_data["candidates"][0]["content"]["parts"][0].get("text")):
            
            gemini_response = response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            log.debug(f"Raw Gemini response: {gemini_response}")
            
            # Parse the JSON response
            try:
                # Extract JSON from response (in case there's extra text)
                json_start = gemini_response.find('{')
                json_end = gemini_response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = gemini_response[json_start:json_end]
                    parsed_result = json.loads(json_str)
                else:
                    raise json.JSONDecodeError("No JSON found", gemini_response, 0)
                
                # Validate the response structure
                primary_intent = parsed_result.get("primary_intent", "general-query")
                confidence = float(parsed_result.get("confidence", 0.5))
                reasoning = parsed_result.get("reasoning", "")
                secondary_intents = parsed_result.get("secondary_intents", [])
                
                # Ensure primary intent is valid
                if primary_intent not in intent_categories:
                    log.warning(f"Invalid primary intent '{primary_intent}' from Gemini. Defaulting to general-query.")
                    primary_intent = "general-query"
                    confidence = 0.5
                
                # Build all_intents list
                all_intents = [{"type": primary_intent, "confidence": confidence}]
                for sec_intent in secondary_intents[:2]:  # Max 2 secondary intents
                    if isinstance(sec_intent, dict) and "intent" in sec_intent:
                        intent_type = sec_intent["intent"]
                        if intent_type in intent_categories and intent_type != primary_intent:
                            all_intents.append({
                                "type": intent_type,
                                "confidence": float(sec_intent.get("confidence", 0.3))
                            })
                
                result = {
                    'primary': {'type': primary_intent, 'confidence': confidence},
                    'all_intents': all_intents,
                    'reasoning': reasoning
                }
                
                log.info(f"Gemini classified jewelry intent: {primary_intent} (confidence: {confidence:.2f}) - {reasoning}")
                return result
                
            except json.JSONDecodeError as e:
                log.error(f"Failed to parse Gemini JSON response: {e}")
                log.debug(f"Raw response was: {gemini_response}")
                # Fallback to simple detection
                return detect_intent_fallback(text, log)
                
        else:
            log.error(f"Unexpected Gemini API response structure: {response_data}")
            return detect_intent_fallback(text, log)
            
    except requests.exceptions.RequestException as e:
        log.error(f"Error calling Gemini API for intent detection: {e}")
        if hasattr(e, 'response') and e.response is not None:
            log.error(f"Gemini API Response: {e.response.text}")
        return detect_intent_fallback(text, log)
    except Exception as e:
        log.error(f"Unexpected error in Gemini intent detection: {e}")
        return detect_intent_fallback(text, log)


# Gemini-based sentiment analysis to replace Google NLP
def analyze_email_sentiment_with_gemini(text: str, logger: Optional[logging.Logger] = None) -> Optional[Dict[str, float]]:
    log = logger or logging.getLogger(__name__)
    
    if not GEMINI_API_KEY:
        log.error("Gemini API key not available for sentiment analysis.")
        return None
    
    if not text or len(text.strip()) < 3:
        log.warning("Text too short for sentiment analysis.")
        return None

    gemini_model_name = "gemini-3-pro-preview-preview"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model_name}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    system_prompt = """
    You are an expert sentiment analyzer. Analyze the sentiment of customer emails and provide:
    
    1. Score: A value between -1.0 (very negative) and +1.0 (very positive), where 0 is neutral
    2. Magnitude: A value between 0.0 and 1.0 indicating the strength/intensity of the sentiment
    
    Respond ONLY with a JSON object in this exact format:
    {
        "score": -0.8,
        "magnitude": 0.9,
        "sentiment_label": "negative",
        "confidence": 0.95,
        "reasoning": "brief explanation"
    }
    
    Rules:
    - score: -1.0 to +1.0 (negative to positive)
    - magnitude: 0.0 to 1.0 (weak to strong emotional intensity)
    - sentiment_label: "positive", "negative", or "neutral"
    - confidence: 0.0 to 1.0 (how certain you are)
    - reasoning: 1-2 sentences explaining the analysis
    """

    prompt_text = f"""
    Analyze the sentiment of this customer email:
    
    "{text.strip()}"
    
    Provide sentiment analysis:
    """

    api_payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt_text}]}
        ],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 300,
            "topP": 0.8,
            "topK": 10
        }
    }

    log.info(f"Analyzing sentiment with Gemini for text: {text[:100]}...")
    
    try:
        response = requests.post(url, headers=headers, json=api_payload, timeout=15)
        response.raise_for_status()
        
        response_data = response.json()
        
        if (response_data.get("candidates") and
            response_data["candidates"][0].get("content") and
            response_data["candidates"][0]["content"].get("parts") and
            response_data["candidates"][0]["content"]["parts"][0].get("text")):
            
            gemini_response = response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            log.debug(f"Raw Gemini sentiment response: {gemini_response}")
            
            try:
                # Extract JSON from response
                json_start = gemini_response.find('{')
                json_end = gemini_response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = gemini_response[json_start:json_end]
                    parsed_result = json.loads(json_str)
                else:
                    raise json.JSONDecodeError("No JSON found", gemini_response, 0)
                
                # Extract and validate values
                score = float(parsed_result.get("score", 0.0))
                magnitude = float(parsed_result.get("magnitude", 0.0))
                sentiment_label = parsed_result.get("sentiment_label", "neutral")
                confidence = float(parsed_result.get("confidence", 0.5))
                reasoning = parsed_result.get("reasoning", "")
                
                # Validate ranges
                score = max(-1.0, min(1.0, score))
                magnitude = max(0.0, min(1.0, magnitude))
                confidence = max(0.0, min(1.0, confidence))
                
                result = {
                    'score': score,
                    'magnitude': magnitude,
                    'sentiment_label': sentiment_label,
                    'confidence': confidence,
                    'reasoning': reasoning
                }
                
                log.info(f"Gemini sentiment analysis: {sentiment_label} (score: {score:.2f}, magnitude: {magnitude:.2f})")
                return result
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                log.error(f"Failed to parse Gemini sentiment response: {e}")
                log.debug(f"Raw response was: {gemini_response}")
                return None
                
        else:
            log.error(f"Unexpected Gemini API response structure for sentiment: {response_data}")
            return None
            
    except requests.exceptions.RequestException as e:
        log.error(f"Error calling Gemini API for sentiment analysis: {e}")
        if hasattr(e, 'response') and e.response is not None:
            log.error(f"Gemini API Response: {e.response.text}")
        return None
    except Exception as e:
        log.error(f"Unexpected error in Gemini sentiment analysis: {e}")
        return None


# Updated main sentiment analysis function to use Gemini by default
def analyze_email_sentiment(text: str, logger: Optional[logging.Logger] = None) -> Optional[Dict[str, float]]:
    return analyze_email_sentiment_with_gemini(text, logger)


# Simple fallback for when Gemini is unavailable
def detect_intent_fallback(text: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Simple fallback when Gemini API is unavailable - returns general-query."""
    log = logger or logging.getLogger(__name__)
    log.warning("Using fallback intent detection - returning general-query.")
    return {
        'primary': {'type': 'general-query', 'confidence': 0.5},
        'all_intents': [{'type': 'general-query', 'confidence': 0.5}],
        'reasoning': 'Gemini API unavailable, using fallback classification'
    }


def _get_gcp_vision_client(logger: logging.Logger) -> Optional[vision.ImageAnnotatorClient]:
    """Helper to initialize Google Cloud Vision Service Client."""
    try:
        if SERVICE_ACCOUNT_FILE and os.path.exists(SERVICE_ACCOUNT_FILE):
            credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
            client = vision.ImageAnnotatorClient(credentials=credentials)
            logger.info("Google VisionServiceClient initialized with service account.")
        else:
            client = vision.ImageAnnotatorClient()
            logger.info("Google VisionServiceClient initialized with default credentials.")
        return client
    except Exception as e:
        logger.error(f"Error initializing Google VisionServiceClient: {e}")
        return None


def extract_text_from_image(image_data_b64: str, logger: Optional[logging.Logger] = None) -> str:
    """Extracts text from a base64 encoded image using Google Vision API."""
    log = logger or logging.getLogger(__name__)
    client = _get_gcp_vision_client(log)
    if not client or not image_data_b64:
        return ""
    try:
        image_content = base64.b64decode(image_data_b64)
        image = vision.Image(content=image_content)
        
        log.info("Performing text detection on image via Google Vision API.")
        response = client.text_detection(image=image)
        
        if response.error.message:
            log.error(f"Google Vision API error: {response.error.message}")
            return ""
        
        if response.text_annotations:
            full_text = response.text_annotations[0].description
            log.info(f"Successfully extracted text from image (raw length: {len(full_text)}).")
            cleaned_text = clean_extracted_text(full_text) # Utility from utils.py
            log.info(f"Cleaned image text length: {len(cleaned_text)}.")
            log.debug(f"Cleaned text preview: {cleaned_text[:200]}...")
            return cleaned_text
        else:
            log.info("No text found in image by Vision API.")
            return ""
    except Exception as e:
        log.error(f"Error extracting text from image: {e}")
        log.debug("Full error details:", exc_info=True)
        return ""


def call_gemini_api(
    initial_response_text: str,
    llm_payload: Dict[str, Any],
    api_key: Optional[str],
    vector_db: Optional[Any] = None,  # Changed to Any to avoid direct import if not ready
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Calls the Gemini API to refine a response, potentially using a vector database for context.
    """
    log = logger or logging.getLogger(__name__)
    if not api_key:
        log.error("Gemini API key not provided. Cannot call API.")
        return initial_response_text  # Return original if no key

    gemini_model_name = "gemini-3-pro-preview-preview"  # Using consistent model across the codebase
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    email_content_data = llm_payload.get('email_content', {})
    current_message = email_content_data.get('current_message', '') if isinstance(email_content_data, dict) else str(email_content_data)

    vector_context_str = ""
    # Check if vector_db is not None and has a 'search' method and is 'ready'
    if vector_db and hasattr(vector_db, 'search') and hasattr(vector_db, 'is_ready') and vector_db.is_ready():
        log.info(f"Searching vector database for query related to intent: {llm_payload.get('intent',{}).get('category','unknown')}")
        try:
            # Ensure current_message is a string for search
            search_query = str(current_message) if current_message else llm_payload.get('subject', '')
            if search_query:
                search_results = vector_db.search(search_query, n_results=3)  # Limit results for context
                if search_results:
                    vector_context_str = "Relevant Information from Knowledge Base (for context, do not cite directly):\n"
                    for result in search_results:
                        vector_context_str += f"- {result.get('text', '')} (Source: {result.get('source', 'Unknown')})\n"
                    log.info(f"Added {len(search_results)} results from vector DB to Gemini prompt.")
                    log.debug(f"Vector context: {vector_context_str[:200]}...")
            else:
                log.info("No search query for vector DB (current message and subject are empty).")
        except Exception as e:
            log.error(f"Error searching vector database: {e}")

    system_prompt = llm_payload.get("system_prompt_from_generate_llm_payload", """
    You are an AI Customer Support Assistant for Pompeii3.
    Refine the following base response to be empathetic, clear, and directly address the customer's query.
    Use the provided payload details for context. If knowledge base info is provided, integrate it naturally.
    Ensure the final response adheres to all formatting guidelines (Greeting: "Hello [Name]," or "Hello ,", single main paragraph, Closing: "Thank You" then "Customer Support Team").
    """)  # Fallback to a generic prompt

    # Constructing the prompt for Gemini
    # Gemini API prefers a structured turn-based conversation.
    # The 'text' part of the user's turn should contain all necessary context.
    prompt_text = (
        f"{system_prompt}\n\n"
        f"CUSTOMER INQUIRY (Payload Summary):\n{json.dumps(llm_payload, indent=2, default=str)}\n\n"
        f"{vector_context_str}\n"  # Add vector context here
        f"BASE RESPONSE TO REFINE:\n{initial_response_text}\n\n"
        f"REFINED RESPONSE (following all guidelines):"
    )

    api_payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": 0.1,  # Lower for more factual, less creative responses
            "maxOutputTokens": 1024,  # Adjust as needed
            "topP": 0.95,
            "topK": 40
        },
        # "system_instruction": { # For newer models like Gemini 1.5, system prompt can go here
        # "parts": [{"text": system_prompt_from_llm_payload_or_default}]
        # },
    }

    log.info(f"Calling Gemini API ({gemini_model_name}) to refine response.")
    log.debug(f"Gemini request payload (excluding full prompt text for brevity): { {k:v for k,v in api_payload.items() if k != 'contents'} }")
    log.debug(f"Gemini prompt text starts with: {prompt_text[:300]}...")

    try:
        response = requests.post(url, headers=headers, json=api_payload, timeout=30)  # Increased timeout
        response.raise_for_status()  # Check for HTTP errors

        response_data = response.json()
        if response_data.get("candidates") and \
           response_data["candidates"][0].get("content") and \
           response_data["candidates"][0]["content"].get("parts") and \
           response_data["candidates"][0]["content"]["parts"][0].get("text"):

            refined_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
            log.info("Successfully refined response with Gemini API.")
            log.debug(f"Refined text: {refined_text[:200]}...")
            return refined_text
        else:
            log.error(f"Unexpected Gemini API response structure: {response_data}")
            return initial_response_text  # Fallback

    except requests.exceptions.RequestException as e:
        log.error(f"Error calling Gemini API ({gemini_model_name}): {e}")
        if hasattr(e, 'response') and e.response is not None:
            log.error(f"Gemini API Response Content: {e.response.text}")
    except Exception as e:  # Catch any other exceptions
        log.error(f"An unexpected error occurred during Gemini API call: {e}")

    return initial_response_text  # Fallback to original response on any error

