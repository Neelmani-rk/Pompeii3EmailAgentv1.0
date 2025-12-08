import logging
from typing import Dict, Any, Optional, List, Set
import datetime
import google.generativeai as genai
import os

# Assuming utils.py is in the same directory or package
from utils import separate_email_parts

# Define system prompts for each intent category
SYSTEM_PROMPTS = {
    "order-related-query": """
    You are an AI Customer Support Assistant for Pompeii3, a jewelry e-commerce brand, specializing in ORDER-RELATED queries. Your primary mission is to provide accurate, empathetic responses for order status, delivery updates, tracking information, shipping changes, and delivery issues.

    **Core Operational Protocol:**

    1. **Order Status & Tracking Analysis:**
    * Analyze order_details.order_status, order_details.estimated_ship_date, order_details.tracking_number, and order_details.tracking_link
    * For tracking inquiries, provide the direct tracking link from order_details.tracking_link when available
    * Cross-reference current_message with order_details to identify specific concerns about delivery timing or status
    * Address shipping address changes or delivery method questions using order_details.shipping_method

    2. **Response Crafting - Order-Focused:**
    * Directly address order status questions with specific information from order_details.order_status
    * For shipped orders, always provide tracking information and direct links
    * For unshipped orders, reference estimated_ship_date and explain next steps
    * Acknowledge delivery urgency mentioned in current_message (e.g., "My son is getting married")

    3. **Mandatory Resources for Order Queries:**
    * Order Status Tracking: https://www.pompeii3.com/order-status-tracker/
    * For delivery issues or changes: https://support.pompeii3.com/

    **Formatting Requirements:**
    * Start with "Hello [First Name]," or "Hello ," if no name available
    * Single paragraph for main content unless multiple distinct topics require separation
    * End with "Thank You" on separate line, followed by "Customer Support Team"
    * Include relevant tracking links and order status tracker link
    """,

    "payment-related-query": """
    You are an AI Customer Support Assistant for Pompeii3, a jewelry e-commerce brand, specializing in PAYMENT-RELATED queries. Your expertise covers payment issues, billing discrepancies, declined payments, refund processing, and payment method questions.

    **Core Operational Protocol:**

    1. **Payment Issue Analysis:**
    * Review current_message for specific payment concerns (declined cards, multiple charges, billing errors)
    * Reference order_details.order_number when addressing payment issues for specific orders
    * Identify payment method questions, financing inquiries, or gift card problems
    * Address sales tax questions and discount code issues

    2. **Response Crafting - Payment-Focused:**
    * Provide clear steps for resolving payment declines or billing issues
    * Explain accepted payment methods and financing options
    * For refund inquiries, reference order status and provide timeline expectations
    * Address security concerns about payment processing

    3. **Mandatory Resources for Payment Queries:**
    * General Support for payment issues: https://support.pompeii3.com/
    * FAQs for payment methods: https://www.pompeii3.com/faqs/

    **Formatting Requirements:**
    * Start with "Hello [First Name]," or "Hello ," if no name available
    * Single paragraph for main content unless multiple distinct topics require separation
    * End with "Thank You" on separate line, followed by "Customer Support Team"
    """,

    "order-cancel-return-refund-query": """
    You are an AI Customer Support Assistant for Pompeii3, a jewelry e-commerce brand, specializing in ORDER CANCELLATION, RETURNS, EXCHANGES, and REFUNDS. Your expertise covers return processes, exchange requests, damaged items, wrong items received, and refund status.

    **Core Operational Protocol:**

    1. **Return/Exchange Analysis:**
    * Analyze current_message for specific return/exchange requests (wrong size, damaged items, etc.)
    * Compare customer complaints with order_details.items to understand discrepancies
    * Identify cancellation requests and assess order status for feasibility
    * For damaged/wrong items, acknowledge the customer's report empathetically

    2. **Response Crafting - Return/Exchange-Focused:**
    * Provide clear return/exchange process steps
    * Reference specific order details when addressing wrong items or sizing issues
    * Explain return shipping procedures and timeline expectations
    * For cancellations, reference order status and cancellation possibilities

    3. **Mandatory Resources for Return/Exchange Queries:**
    * Return Process & Support: https://support.pompeii3.com/
    * Return Policy Details: https://www.pompeii3.com/returns/

    **Formatting Requirements:**
    * Start with "Hello [First Name]," or "Hello ," if no name available
    * Single paragraph for main content unless multiple distinct topics require separation
    * End with "Thank You" on separate line, followed by "Customer Support Team"
    * Always include return policy and support links
    """,

    "product-related-query": """
    You are an AI Customer Support Assistant for Pompeii3, a jewelry e-commerce brand, specializing in PRODUCT-RELATED queries. Your expertise covers product specifications, materials, sizing, availability, warranties, and jewelry education.

    **Core Operational Protocol:**

    1. **Product Information Analysis:**
    * Address questions about metals, diamonds, gemstones, and jewelry specifications
    * Provide sizing guidance and dimension information
    * Explain product availability and authenticity certifications
    * Reference warranty coverage and durability questions

    2. **Response Crafting - Product-Focused:**
    * Utilize jewelry knowledge to answer technical questions about materials and craftsmanship
    * Provide detailed product specifications when requested
    * Explain care instructions and maintenance recommendations
    * Address authenticity and certification questions

    3. **Mandatory Resources for Product Queries:**
    * Website information : https://www.pompeii3.com
    * Jewelry Education: https://www.pompeii3.com/jewelry-education/
    * Warranty Information: https://www.pompeii3.com/warranty/
    * FAQs for product specifications: https://www.pompeii3.com/faqs/

    **Formatting Requirements:**
    * Start with "Hello [First Name]," or "Hello ," if no name available
    * Single paragraph for main content unless multiple distinct topics require separation
    * End with "Thank You" on separate line, followed by "Customer Support Team"
    """,

    "account-related-query": """
    You are an AI Customer Support Assistant for Pompeii3, a jewelry e-commerce brand, specializing in ACCOUNT-RELATED queries. Your expertise covers account management, login issues, profile updates, password resets, and account security.

    **Core Operational Protocol:**

    1. **Account Issue Analysis:**
    * Address login problems, password reset requests, and account access issues
    * Provide guidance for profile updates and account information changes
    * Handle order history inquiries and saved payment method questions
    * Address email preference changes and account security concerns

    2. **Response Crafting - Account-Focused:**
    * Provide step-by-step account recovery instructions
    * Explain account management features and capabilities
    * Address privacy and security questions about account information
    * Guide customers through account settings and preferences

    3. **Mandatory Resources for Account Queries:**
    * General Support for account issues: https://support.pompeii3.com/
    * FAQs for account management: https://www.pompeii3.com/faqs/

    **Formatting Requirements:**
    * Start with "Hello [First Name]," or "Hello ," if no name available
    * Single paragraph for main content unless multiple distinct topics require separation
    * End with "Thank You" on separate line, followed by "Customer Support Team"
    """,

    "customise-product-query": """
    You are an AI Customer Support Assistant for Pompeii3, a jewelry e-commerce brand, specializing in PRODUCT CUSTOMIZATION queries. Your expertise covers custom jewelry creation, engraving services, custom sizing, and personalization options.

    **Core Operational Protocol:**

    1. **Customization Request Analysis:** based on the website https://www.pompeii3.com/
    * Address custom jewelry creation requests and design possibilities
    * Provide information about engraving services and personalization options
    * Explain custom sizing processes and availability
    * Discuss customization timelines, costs, and limitations
    * Can customize Earrings,Engagement Rings , Necklaces , Bracelets , Anniversary jewellery and gemstone accessories .(include in the response)
    * Custom Lab Grown ,Diamond Jewelry From your wildest dreams to your jewelry box, in just 20 Days. (include this in response)

    2. **Response Crafting - Customization-Focused:**
    * For steps , Step 1 : Upload Your Inspiration On the Truly Custom portal , Step 2: Get Your Custom Design Receive high-quality renders of your design within 48 hours. , Step 3: Delivered & Perfect Delivered in as little as 20 days
    * Explain available customization options and design processes
    * Provide realistic timelines for custom work and special orders
    * Address custom item return policies and special considerations
    * Connect customers with appropriate customization resources

    3. **Mandatory Resources for Customization Queries:**
    * Customization options : https://www.pompeii3.com/truly-custom-engagement-rings/
    * General Support for custom orders: https://www.pompeii3.com/truly-custom-jewelry/
    * FAQs for customization: https://www.pompeii3.com/truly-custom-jewelry/

    **Formatting Requirements:**
    * Start with "Hello [First Name]," or "Hello ," if no name available
    * Single paragraph for main content unless multiple distinct topics require separation
    * Strictly include the link to customization mentioned under 'Customization options'
    *
    * End with "Thank You" on separate line, followed by "Customer Support Team"
    """,

    "general-query": """
    You are an AI Customer Support Assistant for Pompeii3, a jewelry e-commerce brand, specializing in GENERAL queries and support. Your expertise covers website issues, business information, contact details, and miscellaneous customer service needs.

    **Core Operational Protocol:**

    1. **General Inquiry Analysis:**
    * Address website technical issues and navigation problems
    * Provide business hours, contact information, and general company details
    * Handle wholesale inquiries and business partnership questions
    * Address feedback, suggestions, and general support requests

    2. **Response Crafting - General Support:**
    * Provide comprehensive company information and contact details
    * Address website functionality issues with troubleshooting steps
    * Handle miscellaneous requests with appropriate resource connections
    * Maintain professional, helpful tone for diverse inquiry types

    3. **Mandatory Resources for General Queries:**
    * General Support Portal: https://support.pompeii3.com/
    * FAQs for general questions: https://www.pompeii3.com/faqs/
    * Jewelry Education: https://www.pompeii3.com/jewelry-education/

    **Formatting Requirements:**
    * Start with "Hello [First Name]," or "Hello ," if no name available
    * Single paragraph for main content unless multiple distinct topics require separation
    * End with "Thank You" on separate line, followed by "Customer Support Team"
    """
}


def generate_llm_payload(email_data: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Enhanced LLM payload generation with conversation history support and improved order processing.
    """
    log = logger or logging.getLogger(__name__)
    log.info("Generating LLM payload from email data.")
    
    # Separate current message from previous conversation from the full body
    current_message = email_data.get('current_message', '')
    previous_conversation = email_data.get('previous_conversation', '')
    
    # If not pre-separated, try to do it now
    if not current_message and email_data.get('body'):
        full_body = email_data.get('body', '')
        current_message, previous_conversation = separate_email_parts(full_body)
        log.debug("Separated email parts within generate_llm_payload.")
    
    # Clean ellipsis (often used in emails) and trim
    current_message = current_message.replace('…', ' ').strip()
    previous_conversation = previous_conversation.replace('…', ' ').strip()
    subject = email_data.get('subject', '').replace('…', ' ').strip()
    
    # Combine subject with current message for better context
    contextual_current_message = f"Subject: {subject}\n\n{current_message}".strip()
    
    email_content_structured = {
        "current_message": contextual_current_message,
        "previous_conversation": previous_conversation
    }
    
    payload: Dict[str, Any] = {
        "email_content": email_content_structured,
        "sender_name": email_data.get('sender_name', ''),
        "sender_email": email_data.get('sender', ''),
        "subject": subject,
        "thread_id": email_data.get('thread_id', ''),
        "message_id": email_data.get('message_id', '')
    }
    
    # Add intent information
    primary_intent_info = email_data.get('intent', {}).get('primary', {})
    if primary_intent_info and isinstance(primary_intent_info, dict):
        payload["intent"] = {
            "category": primary_intent_info.get('type', 'general-query'),
            "confidence": primary_intent_info.get('confidence', 0.7)
        }
    else:
        payload["intent"] = {"category": "general-query", "confidence": 0.6}
    
    log.debug(f"LLM Payload Intent: {payload['intent']}")
    
    # Add extracted entities
    entities_list = email_data.get('entities', [])
    if entities_list and isinstance(entities_list, list):
        payload["entities"] = [
            {
                "type": e.get('type'), 
                "value": e.get('value'), 
                "confidence": e.get('confidence', 0.9), 
                "source": e.get('source', 'text')
            }
            for e in entities_list if isinstance(e, dict)
        ]
    else:
        payload["entities"] = []
    
    log.debug(f"LLM Payload Entities Count: {len(payload['entities'])}")
    
    # Add structured conversation history
    conversation_history = email_data.get('conversation_history', {})
    if conversation_history and isinstance(conversation_history, dict):
        if 'conversation_history' in conversation_history:
            payload["conversation_history"] = conversation_history['conversation_history']
            payload["conversation_summary"] = conversation_history.get('conversation_summary', {})
            log.debug(f"Added structured conversation history with {len(payload['conversation_history'])} messages")
        else:
            payload["conversation_history"] = []
            payload["conversation_summary"] = {}
    else:
        payload["conversation_history"] = []
        payload["conversation_summary"] = {}
    
    # Enhanced order details processing
    order_info_list = email_data.get('order_info', [])
    if order_info_list and isinstance(order_info_list, list) and len(order_info_list) > 0:
        # Find the most relevant order (prefer orders found by order number)
        relevant_order_data = None
        
        # First, try to find an order that was found by order number
        for order_item in order_info_list:
            if isinstance(order_item, dict) and not order_item.get('error'):
                if order_item.get('search_method') != 'customer_name':  # Prefer non-customer-name searches
                    relevant_order_data = order_item
                    break
        
        # If no order found by order number, use the first valid order
        if not relevant_order_data:
            for order_item in order_info_list:
                if isinstance(order_item, dict) and not order_item.get('error'):
                    relevant_order_data = order_item
                    break
        
        if relevant_order_data:
            # Create comprehensive order details
            unified_data = relevant_order_data.get('unified_data', {})
            
            payload["order_details"] = {
                "source": relevant_order_data.get('source', 'unknown'),
                "search_method": relevant_order_data.get('search_method', 'order_number'),
                "order_id": relevant_order_data.get('order_id'),
                "order_number": relevant_order_data.get('order_number'),
                "order_status": relevant_order_data.get('order_status'),
                "order_date": relevant_order_data.get('order_date'),
                "order_total": relevant_order_data.get('order_total'),
                "recipient_name": relevant_order_data.get('recipient_name'),
                "customer_email": relevant_order_data.get('customer_email'),
                "shipping_method": relevant_order_data.get('shipping_method'),
                "estimated_ship_date": relevant_order_data.get('estimated_ship_date'),
                "ship_by_date": relevant_order_data.get('ship_by_date'),
                "shipping_details": relevant_order_data.get('shipping_details', {}),
                "items": relevant_order_data.get('items', []),
                "production_percentage": relevant_order_data.get('production_percentage'),
                "tracking_link": relevant_order_data.get('tracking_link'),
                "is_purchase_order": len(str(relevant_order_data.get('order_number', ''))) == 15,
                "additional_orders_count": len(order_info_list) - 1 if len(order_info_list) > 1 else 0
            }
            
            # Add purchase order number if applicable
            if payload["order_details"]["is_purchase_order"]:
                payload["order_details"]["purchase_order_number"] = relevant_order_data.get('order_number')
            
            log.info(f"Added order details for order: {payload['order_details'].get('order_number')} (Source: {payload['order_details'].get('source')})")
        else:
            payload["order_details"] = {}
    else:
        payload["order_details"] = {}
    
    # Enhanced tracking history processing
    tracking_info_list = email_data.get('tracking_info', [])
    if tracking_info_list and isinstance(tracking_info_list, list) and len(tracking_info_list) > 0:
        # Process all tracking information
        tracking_records = []
        
        for track_item in tracking_info_list:
            if isinstance(track_item, dict) and not track_item.get('error'):
                tracking_record = {
                    "source": track_item.get('source', 'unknown'),
                    "tracking_number": track_item.get('tracking_number'),
                    "carrier": track_item.get('carrier'),
                    "service": track_item.get('service'),
                    "ship_date": track_item.get('ship_date'),
                    "tracking_link": track_item.get('tracking_link'),
                    "order_id": track_item.get('order_id'),
                    "order_number": track_item.get('order_number')
                }
                tracking_records.append(tracking_record)
        
        if tracking_records:
            payload["tracking_history"] = {
                "total_tracking_numbers": len(tracking_records),
                "tracking_records": tracking_records,
                "primary_tracking": tracking_records[0] if tracking_records else {}
            }
            log.info(f"Added {len(tracking_records)} tracking records to payload")
        else:
            payload["tracking_history"] = {}
    else:
        payload["tracking_history"] = {}
    
    # Add sentiment
    sentiment_data = email_data.get('sentiment')
    if sentiment_data and isinstance(sentiment_data, dict):
        payload["sentiment"] = sentiment_data
        log.debug(f"LLM Payload Sentiment: {payload['sentiment']}")
    else:
        payload["sentiment"] = {'score': 0.0, 'magnitude': 0.0}
    
    log.info(f"LLM payload generated successfully. Keys: {list(payload.keys())}")
    log.info(f"Order details populated: {bool(payload.get('order_details'))}")
    log.info(f"Tracking history populated: {bool(payload.get('tracking_history'))}")
    
    return payload


def get_system_prompt_for_intent(intent_category: str) -> str:
    """
    Returns the appropriate system prompt based on the intent category.
    """
    return SYSTEM_PROMPTS.get(intent_category, SYSTEM_PROMPTS["general-query"])

def call_gemini_api(llm_payload: Dict[str, Any], logger: Optional[logging.Logger] = None) -> str:
    """
    Calls the Gemini API to generate a customer service response based on the LLM payload.
    Uses intent-specific system prompts for better response quality.
    """
    log = logger or logging.getLogger(__name__)
    
    try:
        # Configure Gemini API
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found")
        
        genai.configure(api_key=api_key)
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-3-pro-preview')
        
        # Get the appropriate system prompt based on intent
        intent_category = llm_payload.get('intent', {}).get('category', 'general-query')
        system_prompt = get_system_prompt_for_intent(intent_category)
        
        log.info(f"Using system prompt for intent: {intent_category}")
        
        # Prepare the user prompt with the payload data
        user_prompt = f"""
        Please analyze the following customer service payload and generate an appropriate response according to the system instructions:

        **Customer Email Content:**
        Current Message: {llm_payload.get('email_content', {}).get('current_message', '')}
        Previous Conversation: {llm_payload.get('email_content', {}).get('previous_conversation', '')}
        
        **Customer Information:**
        Sender Name: {llm_payload.get('sender_name', '')}
        Sender Email: {llm_payload.get('sender_email', '')}
        Subject: {llm_payload.get('subject', '')}
        
        **Intent Analysis:**
        Category: {llm_payload.get('intent', {}).get('category', 'general-query')}
        Confidence: {llm_payload.get('intent', {}).get('confidence', 0.0)}
        
        **Extracted Entities:**
        {llm_payload.get('entities', [])}
        
        **Order Details:**
        {llm_payload.get('order_details', {})}
        
        **Tracking History:**
        {llm_payload.get('tracking_history', {})}
        
        **Sentiment Analysis:**
        {llm_payload.get('sentiment', {})}
        
        Please generate a customer service response following all the formatting and content guidelines specified in the system prompt.
        """
        
        # Generate response using Gemini with intent-specific system prompt
        log.info("Calling Gemini API to generate customer service response...")
        
        response = model.generate_content(
            [system_prompt, user_prompt],
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,  # Lower temperature for more consistent, professional responses
                max_output_tokens=1000,  # Reasonable limit for customer service responses
                top_p=0.8,
                top_k=40
            )
        )
        
        if response and response.text:
            generated_response = response.text.strip()
            log.info(f"Successfully generated response via Gemini API (length: {len(generated_response)}) using {intent_category} prompt")
            log.debug(f"Generated response preview: {generated_response[:200]}...")
            return generated_response
        else:
            log.error("Gemini API returned empty response")
            return generate_fallback_response(llm_payload, log)
            
    except Exception as e:
        log.error(f"Error calling Gemini API: {str(e)}")
        # Return a fallback response in case of API error
        return generate_fallback_response(llm_payload, log)

def generate_fallback_response(llm_payload: Dict[str, Any], logger: Optional[logging.Logger] = None) -> str:
    """
    Generates a basic fallback response when Gemini API is unavailable.
    """
    log = logger or logging.getLogger(__name__)
    log.info("Generating fallback response due to API unavailability")
    
    # Extract customer name for greeting
    sender_name = llm_payload.get('sender_name', '')
    first_name = sender_name.split()[0] if sender_name and ' ' in sender_name else sender_name
    greeting = f"Hello {first_name}," if first_name else "Hello ,"
    
    # Extract order number if available
    order_details = llm_payload.get('order_details', {})
    order_number = order_details.get('order_number', '')
    order_ref = f"order #{order_number}" if order_number else "your inquiry"
    
    fallback_response = f"""{greeting}

Thank you for contacting Pompeii3 regarding {order_ref}. We have received your message and our customer support team is reviewing your request. We will respond to you as soon as possible with the information you need.
For urgent assistance or immediate support, please contact us via one of the following methods: Our dedicated support line is available 24 hours a day, 7 days a week by telephone at 847-367-7022 . You may also utilize the live chat feature available on our website, https://www.pompeii3.com/.Alternatively, you can submit your inquiry through the contact form on our website at https://www.pompeii3.com/contact-us/.


Thank You
Customer Support Team"""
    
    return fallback_response

def generate_response(llm_payload: Dict[str, Any], logger: Optional[logging.Logger] = None) -> str:
    """
    Generates a customer service response using the Gemini API based on the LLM payload.
    This replaces the previous template-based approach.
    """
    log = logger or logging.getLogger(__name__)
    log.info("Generating response using Gemini API...")
    
    # Call Gemini API to generate the response
    response = call_gemini_api(llm_payload, log)
    
    return response
