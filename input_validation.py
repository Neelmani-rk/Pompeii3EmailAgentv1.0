"""
Input validation and sanitization module for the Email Agent.
Provides comprehensive validation for email content, API inputs, and configuration.
"""

import re
import html
import logging
import bleach
from typing import Any, Dict, List, Optional, Union, Tuple
from email.utils import parseaddr
from urllib.parse import urlparse
import json
import base64
from datetime import datetime
import os


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class InputValidator:
    """Comprehensive input validation and sanitization class."""
    
    # Email validation patterns
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    # Order number patterns
    ORDER_PATTERNS = [
        re.compile(r'^[A-Z]{2,3}\d{6,10}$'),  # Standard format
        re.compile(r'^\d{8,12}$'),            # Numeric only
        re.compile(r'^[A-Z0-9]{8,15}$'),      # Alphanumeric
    ]
    
    # Phone number pattern
    PHONE_PATTERN = re.compile(r'^[\+]?[1-9][\d\s\-\(\)]{7,15}$')
    
    # API key pattern (basic validation)
    API_KEY_PATTERN = re.compile(r'^[A-Za-z0-9_\-]{20,}$')
    
    # Allowed HTML tags for email content
    ALLOWED_HTML_TAGS = [
        'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li', 'a', 'span', 'div'
    ]
    
    # Allowed HTML attributes
    ALLOWED_HTML_ATTRS = {
        'a': ['href', 'title'],
        'span': ['style'],
        'div': ['style']
    }
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the validator with optional logger."""
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_email_address(self, email: str) -> str:
        """
        Validate and sanitize email address.
        
        Args:
            email: Email address to validate
            
        Returns:
            Sanitized email address
            
        Raises:
            ValidationError: If email is invalid
        """
        if not email or not isinstance(email, str):
            raise ValidationError("Email address is required and must be a string")
        
        # Clean and normalize
        email = email.strip().lower()
        
        # Parse email to extract address part
        parsed_name, parsed_email = parseaddr(email)
        if parsed_email:
            email = parsed_email
        
        # Validate format
        if not self.EMAIL_PATTERN.match(email):
            raise ValidationError(f"Invalid email format: {email}")
        
        # Additional security checks
        if len(email) > 254:  # RFC 5321 limit
            raise ValidationError("Email address too long")
        
        return email
    
    def validate_email_subject(self, subject: str) -> str:
        """
        Validate and sanitize email subject.
        
        Args:
            subject: Email subject to validate
            
        Returns:
            Sanitized subject
        """
        if not subject:
            return ""
        
        if not isinstance(subject, str):
            subject = str(subject)
        
        # Remove control characters and excessive whitespace
        subject = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', subject)
        subject = ' '.join(subject.split())
        
        # Limit length
        if len(subject) > 998:  # RFC 2822 limit
            subject = subject[:995] + "..."
        
        return subject.strip()
    
    def sanitize_email_body(self, body: str) -> str:
        """
        Sanitize email body content.
        
        Args:
            body: Email body to sanitize
            
        Returns:
            Sanitized body content
        """
        if not body:
            return ""
        
        if not isinstance(body, str):
            body = str(body)
        
        # Use bleach to sanitize HTML
        sanitized = bleach.clean(
            body,
            tags=self.ALLOWED_HTML_TAGS,
            attributes=self.ALLOWED_HTML_ATTRS,
            strip=True
        )
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\n\s*\n\s*\n', '\n\n', sanitized)
        
        return sanitized.strip()
    
    def validate_order_number(self, order_number: str) -> str:
        """
        Validate order number format.
        
        Args:
            order_number: Order number to validate
            
        Returns:
            Validated order number
            
        Raises:
            ValidationError: If order number is invalid
        """
        if not order_number or not isinstance(order_number, str):
            raise ValidationError("Order number is required and must be a string")
        
        order_number = order_number.strip().upper()
        
        # Check against known patterns
        for pattern in self.ORDER_PATTERNS:
            if pattern.match(order_number):
                return order_number
        
        raise ValidationError(f"Invalid order number format: {order_number}")
    
    def validate_tracking_number(self, tracking_number: str) -> str:
        """
        Validate tracking number format.
        
        Args:
            tracking_number: Tracking number to validate
            
        Returns:
            Validated tracking number
            
        Raises:
            ValidationError: If tracking number is invalid
        """
        if not tracking_number or not isinstance(tracking_number, str):
            raise ValidationError("Tracking number is required and must be a string")
        
        tracking_number = tracking_number.strip().upper()
        
        # Basic validation - alphanumeric, 8-30 characters
        if not re.match(r'^[A-Z0-9]{8,30}$', tracking_number):
            raise ValidationError(f"Invalid tracking number format: {tracking_number}")
        
        return tracking_number
    
    def validate_phone_number(self, phone: str) -> str:
        """
        Validate phone number format.
        
        Args:
            phone: Phone number to validate
            
        Returns:
            Validated phone number
            
        Raises:
            ValidationError: If phone number is invalid
        """
        if not phone or not isinstance(phone, str):
            raise ValidationError("Phone number is required and must be a string")
        
        # Remove common formatting characters
        cleaned = re.sub(r'[\s\-\(\)\.]', '', phone)
        
        if not self.PHONE_PATTERN.match(phone):
            raise ValidationError(f"Invalid phone number format: {phone}")
        
        return phone.strip()
    
    def validate_api_key(self, api_key: str, service_name: str = "API") -> str:
        """
        Validate API key format.
        
        Args:
            api_key: API key to validate
            service_name: Name of the service for error messages
            
        Returns:
            Validated API key
            
        Raises:
            ValidationError: If API key is invalid
        """
        if not api_key or not isinstance(api_key, str):
            raise ValidationError(f"{service_name} API key is required and must be a string")
        
        api_key = api_key.strip()
        
        if len(api_key) < 20:
            raise ValidationError(f"{service_name} API key too short")
        
        if not self.API_KEY_PATTERN.match(api_key):
            raise ValidationError(f"Invalid {service_name} API key format")
        
        return api_key
    
    def validate_url(self, url: str) -> str:
        """
        Validate URL format.
        
        Args:
            url: URL to validate
            
        Returns:
            Validated URL
            
        Raises:
            ValidationError: If URL is invalid
        """
        if not url or not isinstance(url, str):
            raise ValidationError("URL is required and must be a string")
        
        url = url.strip()
        
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValidationError(f"Invalid URL format: {url}")
            
            if parsed.scheme not in ['http', 'https']:
                raise ValidationError(f"URL must use HTTP or HTTPS: {url}")
            
        except Exception as e:
            raise ValidationError(f"Invalid URL: {e}")
        
        return url
    
    def validate_json_data(self, data: str) -> Dict[str, Any]:
        """
        Validate and parse JSON data.
        
        Args:
            data: JSON string to validate
            
        Returns:
            Parsed JSON data
            
        Raises:
            ValidationError: If JSON is invalid
        """
        if not data or not isinstance(data, str):
            raise ValidationError("JSON data is required and must be a string")
        
        try:
            parsed = json.loads(data)
            return parsed
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON format: {e}")
    
    def validate_base64_data(self, data: str) -> bytes:
        """
        Validate and decode base64 data.
        
        Args:
            data: Base64 string to validate
            
        Returns:
            Decoded bytes
            
        Raises:
            ValidationError: If base64 is invalid
        """
        if not data or not isinstance(data, str):
            raise ValidationError("Base64 data is required and must be a string")
        
        try:
            # Remove data URL prefix if present
            if data.startswith('data:'):
                data = data.split(',', 1)[1]
            
            decoded = base64.b64decode(data)
            return decoded
        except Exception as e:
            raise ValidationError(f"Invalid base64 data: {e}")
    
    def validate_file_path(self, file_path: str) -> str:
        """
        Validate file path for security.
        
        Args:
            file_path: File path to validate
            
        Returns:
            Validated file path
            
        Raises:
            ValidationError: If file path is invalid
        """
        if not file_path or not isinstance(file_path, str):
            raise ValidationError("File path is required and must be a string")
        
        file_path = file_path.strip()
        
        # Check for path traversal attempts
        if '..' in file_path or file_path.startswith('/'):
            raise ValidationError("Invalid file path: path traversal detected")
        
        # Check for invalid characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in file_path for char in invalid_chars):
            raise ValidationError("Invalid characters in file path")
        
        return file_path
    
    def validate_gmail_message_id(self, message_id: str) -> str:
        """
        Validate Gmail message ID format.
        
        Args:
            message_id: Gmail message ID to validate
            
        Returns:
            Validated message ID
            
        Raises:
            ValidationError: If message ID is invalid
        """
        if not message_id or not isinstance(message_id, str):
            raise ValidationError("Gmail message ID is required and must be a string")
        
        message_id = message_id.strip()
        
        # Gmail message IDs are typically 16-20 character hex strings
        if not re.match(r'^[a-fA-F0-9]{16,20}$', message_id):
            raise ValidationError(f"Invalid Gmail message ID format: {message_id}")
        
        return message_id
    
    def validate_gmail_thread_id(self, thread_id: str) -> str:
        """
        Validate Gmail thread ID format.
        
        Args:
            thread_id: Gmail thread ID to validate
            
        Returns:
            Validated thread ID
            
        Raises:
            ValidationError: If thread ID is invalid
        """
        if not thread_id or not isinstance(thread_id, str):
            raise ValidationError("Gmail thread ID is required and must be a string")
        
        thread_id = thread_id.strip()
        
        # Gmail thread IDs are typically 16-20 character hex strings
        if not re.match(r'^[a-fA-F0-9]{16,20}$', thread_id):
            raise ValidationError(f"Invalid Gmail thread ID format: {thread_id}")
        
        return thread_id


class ConfigValidator:
    """Validator for configuration and environment variables."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the config validator."""
        self.logger = logger or logging.getLogger(__name__)
        self.input_validator = InputValidator(logger)
    
    def validate_environment_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate environment configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Validated configuration
            
        Raises:
            ValidationError: If configuration is invalid
        """
        validated_config = {}
        
        # Validate API keys
        if config.get('GEMINI_API_KEY'):
            validated_config['GEMINI_API_KEY'] = self.input_validator.validate_api_key(
                config['GEMINI_API_KEY'], 'Gemini'
            )
        
        if config.get('SHIPSTATION_API_KEY'):
            validated_config['SHIPSTATION_API_KEY'] = self.input_validator.validate_api_key(
                config['SHIPSTATION_API_KEY'], 'ShipStation'
            )
        
        if config.get('SHIPSTATION_API_SECRET'):
            validated_config['SHIPSTATION_API_SECRET'] = self.input_validator.validate_api_key(
                config['SHIPSTATION_API_SECRET'], 'ShipStation Secret'
            )
        
        if config.get('FEDEX_CLIENT_ID'):
            validated_config['FEDEX_CLIENT_ID'] = self.input_validator.validate_api_key(
                config['FEDEX_CLIENT_ID'], 'FedEx Client ID'
            )
        
        if config.get('FEDEX_CLIENT_SECRET'):
            validated_config['FEDEX_CLIENT_SECRET'] = self.input_validator.validate_api_key(
                config['FEDEX_CLIENT_SECRET'], 'FedEx Client Secret'
            )
        
        # Validate file paths
        for file_key in ['SERVICE_ACCOUNT_FILE', 'TOKEN_FILE', 'CREDENTIALS_FILE']:
            if config.get(file_key):
                validated_config[file_key] = self.input_validator.validate_file_path(
                    config[file_key]
                )
        
        # Validate bucket names
        for bucket_key in ['BUCKET_NAME', 'RESPONSES_BUCKET_NAME', 'VECTOR_DB_BUCKET']:
            if config.get(bucket_key):
                bucket_name = config[bucket_key]
                if not isinstance(bucket_name, str) or not bucket_name.strip():
                    raise ValidationError(f"{bucket_key} must be a non-empty string")
                validated_config[bucket_key] = bucket_name.strip()
        
        # Validate rate limiting
        if config.get('RATE_PER_MINUTE'):
            rate = config['RATE_PER_MINUTE']
            if not isinstance(rate, (int, str)):
                raise ValidationError("RATE_PER_MINUTE must be a number")
            
            try:
                rate_int = int(rate)
                if rate_int <= 0 or rate_int > 1000:
                    raise ValidationError("RATE_PER_MINUTE must be between 1 and 1000")
                validated_config['RATE_PER_MINUTE'] = rate_int
            except ValueError:
                raise ValidationError("RATE_PER_MINUTE must be a valid integer")
        
        return validated_config


def validate_email_data(email_data: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Validate processed email data structure.
    
    Args:
        email_data: Email data dictionary to validate
        logger: Optional logger instance
        
    Returns:
        Validated email data
        
    Raises:
        ValidationError: If email data is invalid
    """
    validator = InputValidator(logger)
    
    if not isinstance(email_data, dict):
        raise ValidationError("Email data must be a dictionary")
    
    validated_data = {}
    
    # Validate sender email
    if email_data.get('sender'):
        validated_data['sender'] = validator.validate_email_address(email_data['sender'])
    
    # Validate subject
    if email_data.get('subject'):
        validated_data['subject'] = validator.validate_email_subject(email_data['subject'])
    
    # Validate body content
    if email_data.get('current_message'):
        validated_data['current_message'] = validator.sanitize_email_body(email_data['current_message'])
    
    if email_data.get('full_conversation'):
        validated_data['full_conversation'] = validator.sanitize_email_body(email_data['full_conversation'])
    
    # Validate message and thread IDs
    if email_data.get('message_id'):
        validated_data['message_id'] = validator.validate_gmail_message_id(email_data['message_id'])
    
    if email_data.get('thread_id'):
        validated_data['thread_id'] = validator.validate_gmail_thread_id(email_data['thread_id'])
    
    # Copy other validated fields
    for key in ['entities', 'sentiment', 'intent', 'images', 'order_info']:
        if key in email_data:
            validated_data[key] = email_data[key]
    
    return validated_data


def sanitize_api_response(response_data: Any, logger: Optional[logging.Logger] = None) -> Any:
    """
    Sanitize API response data.
    
    Args:
        response_data: API response data
        logger: Optional logger instance
        
    Returns:
        Sanitized response data
    """
    if isinstance(response_data, dict):
        sanitized = {}
        for key, value in response_data.items():
            if isinstance(key, str):
                sanitized[key.strip()] = sanitize_api_response(value, logger)
        return sanitized
    elif isinstance(response_data, list):
        return [sanitize_api_response(item, logger) for item in response_data]
    elif isinstance(response_data, str):
        # Remove control characters and excessive whitespace
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', response_data)
        return ' '.join(sanitized.split())
    else:
        return response_data
