"""
Email Quality Validation Agent for the Email Agent system.
This module validates generated email drafts for quality, helpfulness, and accuracy.
It tracks knowledge gaps and provides improvement suggestions.

Enhanced Features:
- Robust error handling with fallback mechanisms
- Response format validation and correction
- Configurable quality thresholds
- Detailed logging and analytics
- Retry logic for API calls
- Caching for performance optimization
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import google.generativeai as genai
from dataclasses import dataclass, asdict, field

from config import (
    GEMINI_API_KEY, SPREADSHEET_ID, SERVICE_ACCOUNT_FILE,
    ENABLE_QUALITY_VALIDATION, MIN_QUALITY_THRESHOLD, MIN_HELPFULNESS_THRESHOLD
)
from input_validation import InputValidator, ValidationError
from utils import setup_logging

# Conditional imports with fallbacks
try:
    import gspread
    from google.oauth2 import service_account
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    logging.warning("gspread not available - Google Sheets tracking disabled")


@dataclass
class QualityMetrics:
    """Quality assessment metrics for email responses."""
    helpfulness_score: float = 0.5  # 0.0 - 1.0
    information_density: float = 0.5  # 0.0 - 1.0
    specificity_score: float = 0.5  # 0.0 - 1.0
    actionability_score: float = 0.5  # 0.0 - 1.0
    empathy_score: float = 0.5  # 0.0 - 1.0
    professionalism_score: float = 0.5  # 0.0 - 1.0
    format_compliance_score: float = 0.5  # 0.0 - 1.0
    overall_score: float = 0.5  # 0.0 - 1.0
    has_specific_info: bool = False
    has_links: bool = False
    has_greeting: bool = False
    has_closing: bool = False
    word_count: int = 0
    knowledge_gaps: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    validation_status: str = "pending"  # pending, passed, failed, error
    reasoning: str = ""
    
    def __post_init__(self):
        """Ensure all scores are within valid range."""
        score_fields = [
            'helpfulness_score', 'information_density', 'specificity_score',
            'actionability_score', 'empathy_score', 'professionalism_score',
            'format_compliance_score', 'overall_score'
        ]
        for field_name in score_fields:
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)):
                setattr(self, field_name, 0.5)
            else:
                setattr(self, field_name, max(0.0, min(1.0, float(value))))


@dataclass
class KnowledgeGap:
    """Represents a knowledge gap identified in customer service."""
    timestamp: str
    customer_query: str
    missing_knowledge_area: str
    specific_question: str
    customer_email: str
    order_number: Optional[str]
    urgency_level: str  # low, medium, high, critical
    suggested_solution: str
    intent_category: str = "general"
    frequency_count: int = 1
    status: str = "new"  # new, in_progress, resolved


class EmailQualityValidator:
    """
    Validates email draft quality and tracks knowledge gaps.
    
    This class provides comprehensive quality assessment for generated
    email responses, ensuring they meet customer service standards.
    """
    
    # Default quality thresholds (can be overridden via config)
    DEFAULT_MIN_HELPFULNESS_SCORE = 0.6
    DEFAULT_MIN_INFORMATION_DENSITY = 0.4
    DEFAULT_MIN_OVERALL_SCORE = 0.5
    DEFAULT_MIN_EMPATHY_SCORE = 0.5
    DEFAULT_MIN_FORMAT_COMPLIANCE = 0.6
    
    # Knowledge gap categories
    KNOWLEDGE_CATEGORIES = {
        'product_info': 'Product Information',
        'shipping_policy': 'Shipping Policies',
        'return_policy': 'Return/Exchange Policies',
        'order_processing': 'Order Processing',
        'payment_issues': 'Payment and Billing',
        'technical_support': 'Technical Support',
        'inventory_status': 'Inventory and Availability',
        'promotional_codes': 'Promotions and Discounts',
        'account_management': 'Account Management',
        'custom_orders': 'Custom Orders and Personalization',
        'warranty': 'Warranty Information',
        'sizing': 'Sizing and Fit',
        'materials': 'Materials and Care'
    }
    
    # Required response format elements
    REQUIRED_ELEMENTS = {
        'greeting': [r'hello\s*,', r'hi\s*,', r'dear\s*'],
        'closing': [r'thank\s*you', r'best\s*regards', r'sincerely', r'customer\s*support\s*team'],
        'links': [r'https?://[^\s]+', r'www\.[^\s]+']
    }
    
    # API retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds
    
    def __init__(self, logger: Optional[logging.Logger] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the EmailQualityValidator with required components.
        
        Args:
            logger: Optional logger instance
            config: Optional configuration dictionary to override defaults
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        # Initialize input validator
        try:
            self.validator = InputValidator(self.logger)
        except Exception as e:
            self.logger.warning(f"InputValidator initialization failed: {e}")
            self.validator = None
        
        # Initialize caches
        self.knowledge_gap_cache: Dict[str, int] = {}
        self.validation_cache: Dict[str, Tuple[bool, QualityMetrics]] = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        self.cache_timestamps: Dict[str, float] = {}
        
        # Sheets client
        self.knowledge_gap_sheet = None
        self.sheets_client = None
        
        # Quality thresholds (configurable)
        self.min_helpfulness_score = self.config.get(
            'min_helpfulness_score', 
            MIN_HELPFULNESS_THRESHOLD if MIN_HELPFULNESS_THRESHOLD else self.DEFAULT_MIN_HELPFULNESS_SCORE
        )
        self.min_information_density = self.config.get(
            'min_information_density', 
            self.DEFAULT_MIN_INFORMATION_DENSITY
        )
        self.min_overall_score = self.config.get(
            'min_overall_score', 
            MIN_QUALITY_THRESHOLD if MIN_QUALITY_THRESHOLD else self.DEFAULT_MIN_OVERALL_SCORE
        )
        self.min_empathy_score = self.config.get(
            'min_empathy_score', 
            self.DEFAULT_MIN_EMPATHY_SCORE
        )
        self.min_format_compliance = self.config.get(
            'min_format_compliance', 
            self.DEFAULT_MIN_FORMAT_COMPLIANCE
        )
        
        # Initialize Gemini API for quality assessment
        self.model = None
        self._init_gemini_model()
        
        # Initialize Google Sheets client for knowledge gap tracking
        if GSPREAD_AVAILABLE:
            self._init_sheets_client()
        else:
            self.logger.info("Google Sheets integration skipped - gspread not available")
    
    def _init_gemini_model(self) -> None:
        """Initialize Gemini model with error handling and retry logic."""
        if not GEMINI_API_KEY:
            self.logger.warning("GEMINI_API_KEY not available - quality validation will use rule-based fallback")
            return
        
        for attempt in range(self.MAX_RETRIES):
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.model = genai.GenerativeModel(
                    'gemini-3-pro',
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,  # Low temperature for consistent evaluation
                        max_output_tokens=1500,
                        top_p=0.95
                    )
                )
                self.logger.info("Gemini model initialized for quality validation")
                return
            except Exception as e:
                self.logger.warning(f"Gemini initialization attempt {attempt + 1} failed: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
        
        self.logger.error("Failed to initialize Gemini model after all retries")
        self.model = None
    
    def _init_sheets_client(self) -> None:
        """Initialize Google Sheets client for knowledge gap tracking."""
        if not GSPREAD_AVAILABLE:
            return
            
        try:
            if SERVICE_ACCOUNT_FILE and os.path.exists(SERVICE_ACCOUNT_FILE):
                credentials = service_account.Credentials.from_service_account_file(
                    SERVICE_ACCOUNT_FILE,
                    scopes=[
                        "https://www.googleapis.com/auth/spreadsheets",
                        "https://www.googleapis.com/auth/drive",
                    ],
                )
                gc = gspread.authorize(credentials)
            else:
                # Fallback to default credentials
                try:
                    gc = gspread.service_account()
                except Exception:
                    self.logger.warning("No service account available for Sheets")
                    return

            if SPREADSHEET_ID:
                try:
                    sheet = gc.open_by_key(SPREADSHEET_ID)
                    try:
                        self.knowledge_gap_sheet = sheet.worksheet("Knowledge_Gaps")
                    except gspread.WorksheetNotFound:
                        self.knowledge_gap_sheet = sheet.add_worksheet(
                            title="Knowledge_Gaps", rows=1000, cols=15
                        )
                        self._setup_knowledge_gap_headers()
                    
                    self.sheets_client = gc
                    self.logger.info("Google Sheets client initialized successfully")
                except Exception as e:
                    self.logger.warning(f"Could not access spreadsheet: {e}")
            else:
                self.logger.info("SPREADSHEET_ID not configured - Sheets tracking disabled")

        except Exception as e:
            self.logger.error(f"Failed to initialize Google Sheets client: {e}")
            self.sheets_client = None
            self.knowledge_gap_sheet = None
    
    def _setup_knowledge_gap_headers(self) -> None:
        """Set up headers for the knowledge gap tracking sheet."""
        headers = [
            "Timestamp", "Customer Email", "Order Number", "Intent Category",
            "Customer Query", "Knowledge Gap Category", "Specific Missing Info",
            "Urgency Level", "Suggested Solution", "Frequency Count", 
            "Status", "Resolution Notes", "Resolved By", "Resolution Date", "Last Updated"
        ]
        if self.knowledge_gap_sheet:
            try:
                self.knowledge_gap_sheet.update('A1:O1', [headers])
                self.knowledge_gap_sheet.format('A1:O1', {
                    'textFormat': {'bold': True},
                    'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
                })
            except Exception as e:
                self.logger.warning(f"Could not set up sheet headers: {e}")
    
    def validate_email_draft(
        self, 
        draft_content: str, 
        original_query: str, 
        customer_context: Dict[str, Any]
    ) -> Tuple[bool, QualityMetrics, List[str]]:
        """
        Validate an email draft for quality and helpfulness.
        
        Args:
            draft_content: The generated email draft
            original_query: The original customer query
            customer_context: Context about the customer and their query
            
        Returns:
            Tuple of (should_send, quality_metrics, improvement_suggestions)
        """
        start_time = time.time()
        
        # Check if quality validation is enabled
        if not ENABLE_QUALITY_VALIDATION:
            self.logger.info("Quality validation disabled - auto-approving draft")
            return True, self._create_default_metrics("validation_disabled"), []
        
        # Validate inputs
        if not draft_content or not draft_content.strip():
            self.logger.warning("Empty draft content provided")
            return False, self._create_default_metrics("empty_draft", overall_score=0.0), ["Draft content is empty"]
        
        if not original_query:
            original_query = customer_context.get('subject', 'Customer inquiry')
        
        try:
            # Sanitize inputs
            if self.validator:
                draft_content = self.validator.sanitize_email_body(draft_content)
                original_query = self.validator.sanitize_email_body(original_query)
            
            # Check cache first
            cache_key = self._generate_cache_key(draft_content, original_query)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.logger.debug("Using cached validation result")
                return cached_result[0], cached_result[1], cached_result[1].improvement_suggestions
            
            # Perform rule-based validation first (faster)
            rule_based_metrics = self._perform_rule_based_validation(draft_content)
            
            # If AI model is available, enhance with AI assessment
            if self.model:
                try:
                    ai_metrics = self._perform_ai_validation(
                        draft_content, original_query, customer_context
                    )
                    # Merge rule-based and AI metrics
                    metrics = self._merge_metrics(rule_based_metrics, ai_metrics)
                except Exception as ai_error:
                    self.logger.warning(f"AI validation failed, using rule-based only: {ai_error}")
                    metrics = rule_based_metrics
            else:
                metrics = rule_based_metrics
            
            # Calculate final overall score
            metrics.overall_score = self._calculate_overall_score(metrics)
            
            # Determine if draft should be sent
            should_send = self._should_send_draft(metrics)
            metrics.validation_status = "passed" if should_send else "failed"
            
            # Generate improvement suggestions
            suggestions = self._generate_improvement_suggestions(metrics, {})
            metrics.improvement_suggestions = suggestions
            
            # Track knowledge gaps if any
            if metrics.knowledge_gaps:
                self._track_knowledge_gaps(
                    metrics.knowledge_gaps, original_query, customer_context
                )
            
            # Cache the result
            self._add_to_cache(cache_key, (should_send, metrics))
            
            # Log validation summary
            validation_time = time.time() - start_time
            self.logger.info(
                f"Draft validation completed in {validation_time:.2f}s - "
                f"Overall: {metrics.overall_score:.2f}, Approved: {should_send}"
            )
            
            return should_send, metrics, suggestions
            
        except Exception as e:
            self.logger.error(f"Error validating email draft: {e}", exc_info=True)
            default_metrics = self._create_default_metrics("error", overall_score=0.5)
            default_metrics.improvement_suggestions = ["Error during validation - manual review recommended"]
            return True, default_metrics, default_metrics.improvement_suggestions
    
    def _create_default_metrics(self, status: str = "default", overall_score: float = 0.5) -> QualityMetrics:
        """Create default metrics with specified status."""
        return QualityMetrics(
            helpfulness_score=0.5,
            information_density=0.5,
            specificity_score=0.5,
            actionability_score=0.5,
            empathy_score=0.5,
            professionalism_score=0.5,
            format_compliance_score=0.5,
            overall_score=overall_score,
            has_specific_info=False,
            has_links=False,
            has_greeting=True,
            has_closing=True,
            word_count=0,
            knowledge_gaps=[],
            improvement_suggestions=[],
            validation_status=status,
            reasoning="Default metrics applied"
        )
    
    def _perform_rule_based_validation(self, draft_content: str) -> QualityMetrics:
        """
        Perform rule-based validation on the draft content.
        This provides fast, deterministic validation without API calls.
        """
        content_lower = draft_content.lower()
        
        # Check format elements
        has_greeting = any(
            re.search(pattern, content_lower) 
            for pattern in self.REQUIRED_ELEMENTS['greeting']
        )
        has_closing = any(
            re.search(pattern, content_lower) 
            for pattern in self.REQUIRED_ELEMENTS['closing']
        )
        has_links = any(
            re.search(pattern, draft_content) 
            for pattern in self.REQUIRED_ELEMENTS['links']
        )
        
        # Word count and content analysis
        words = draft_content.split()
        word_count = len(words)
        
        # Calculate format compliance score
        format_elements_present = sum([has_greeting, has_closing, has_links])
        format_compliance_score = format_elements_present / 3.0
        
        # Check for specific information indicators
        specific_info_patterns = [
            r'order\s*#?\s*\d+',
            r'tracking\s*(?:number|#)',
            r'\$\s*[\d,]+\.?\d*',
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            r'(?:shipped|delivered|processing|pending)',
        ]
        has_specific_info = any(
            re.search(pattern, content_lower) 
            for pattern in specific_info_patterns
        )
        
        # Calculate information density based on content length and specific info
        if word_count < 20:
            information_density = 0.2
        elif word_count < 50:
            information_density = 0.4
        elif word_count < 150:
            information_density = 0.6 + (0.2 if has_specific_info else 0)
        else:
            information_density = 0.7 + (0.2 if has_specific_info else 0)
        
        # Check for empathy indicators
        empathy_patterns = [
            r'understand', r'sorry', r'apologize', r'appreciate', 
            r'thank you for', r'happy to help', r'glad to assist',
            r'concern', r'important to us'
        ]
        empathy_matches = sum(1 for p in empathy_patterns if re.search(p, content_lower))
        empathy_score = min(1.0, empathy_matches * 0.15 + 0.3)
        
        # Check for actionability
        action_patterns = [
            r'please\s+(?:contact|call|visit|click|follow)',
            r'you\s+can',
            r'next\s+step',
            r'to\s+(?:track|check|view)',
            r'https?://',
            r'if\s+you\s+(?:have|need)',
        ]
        action_matches = sum(1 for p in action_patterns if re.search(p, content_lower))
        actionability_score = min(1.0, action_matches * 0.2 + 0.2)
        
        # Professionalism check
        unprofessional_patterns = [
            r'lol', r'omg', r'btw', r'idk', r'imo',
            r'!!+', r'\?\?+', r'\.\.\.+',
        ]
        unprofessional_matches = sum(1 for p in unprofessional_patterns if re.search(p, content_lower))
        professionalism_score = max(0.3, 1.0 - (unprofessional_matches * 0.2))
        
        return QualityMetrics(
            helpfulness_score=0.5,  # Will be enhanced by AI
            information_density=min(1.0, information_density),
            specificity_score=0.7 if has_specific_info else 0.4,
            actionability_score=min(1.0, actionability_score),
            empathy_score=min(1.0, empathy_score),
            professionalism_score=professionalism_score,
            format_compliance_score=format_compliance_score,
            overall_score=0.5,  # Will be calculated later
            has_specific_info=has_specific_info,
            has_links=has_links,
            has_greeting=has_greeting,
            has_closing=has_closing,
            word_count=word_count,
            knowledge_gaps=[],
            improvement_suggestions=[],
            validation_status="rule_based",
            reasoning="Rule-based validation completed"
        )
    
    def _perform_ai_validation(
        self, 
        draft_content: str, 
        original_query: str, 
        customer_context: Dict[str, Any]
    ) -> QualityMetrics:
        """Perform AI-powered validation using Gemini."""
        prompt = self._create_quality_assessment_prompt(
            draft_content, original_query, customer_context
        )
        
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.model.generate_content(prompt)
                if response and response.text:
                    assessment_data = self._parse_quality_response(response.text)
                    
                    return QualityMetrics(
                        helpfulness_score=float(assessment_data.get('helpfulness_score', 0.5)),
                        information_density=float(assessment_data.get('information_density', 0.5)),
                        specificity_score=float(assessment_data.get('specificity_score', 0.5)),
                        actionability_score=float(assessment_data.get('actionability_score', 0.5)),
                        empathy_score=float(assessment_data.get('empathy_score', 0.5)),
                        professionalism_score=float(assessment_data.get('professionalism_score', 0.8)),
                        format_compliance_score=float(assessment_data.get('format_compliance_score', 0.5)),
                        overall_score=float(assessment_data.get('overall_score', 0.5)),
                        has_specific_info=bool(assessment_data.get('has_specific_info', False)),
                        has_links=bool(assessment_data.get('has_links', False)),
                        has_greeting=bool(assessment_data.get('has_greeting', True)),
                        has_closing=bool(assessment_data.get('has_closing', True)),
                        word_count=int(assessment_data.get('word_count', 0)),
                        knowledge_gaps=assessment_data.get('knowledge_gaps', []),
                        improvement_suggestions=assessment_data.get('improvement_suggestions', []),
                        validation_status="ai_validated",
                        reasoning=assessment_data.get('reasoning', '')
                    )
            except Exception as e:
                self.logger.warning(f"AI validation attempt {attempt + 1} failed: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
        
        # Return default metrics if all attempts fail
        return self._create_default_metrics("ai_failed")
    
    def _merge_metrics(self, rule_based: QualityMetrics, ai_based: QualityMetrics) -> QualityMetrics:
        """Merge rule-based and AI-based metrics with weighted average."""
        # Weight: 40% rule-based, 60% AI-based for most scores
        rule_weight = 0.4
        ai_weight = 0.6
        
        return QualityMetrics(
            helpfulness_score=ai_based.helpfulness_score,  # Trust AI for helpfulness
            information_density=(rule_based.information_density * rule_weight + 
                               ai_based.information_density * ai_weight),
            specificity_score=(rule_based.specificity_score * rule_weight + 
                             ai_based.specificity_score * ai_weight),
            actionability_score=(rule_based.actionability_score * 0.5 + 
                               ai_based.actionability_score * 0.5),
            empathy_score=(rule_based.empathy_score * rule_weight + 
                         ai_based.empathy_score * ai_weight),
            professionalism_score=(rule_based.professionalism_score * 0.6 + 
                                 ai_based.professionalism_score * 0.4),
            format_compliance_score=rule_based.format_compliance_score,  # Trust rules for format
            overall_score=0.5,  # Will be calculated
            has_specific_info=rule_based.has_specific_info or ai_based.has_specific_info,
            has_links=rule_based.has_links,
            has_greeting=rule_based.has_greeting,
            has_closing=rule_based.has_closing,
            word_count=rule_based.word_count,
            knowledge_gaps=ai_based.knowledge_gaps,
            improvement_suggestions=ai_based.improvement_suggestions,
            validation_status="merged",
            reasoning=ai_based.reasoning
        )
    
    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """Calculate weighted overall score from individual metrics."""
        weights = {
            'helpfulness_score': 0.25,
            'information_density': 0.15,
            'specificity_score': 0.15,
            'actionability_score': 0.15,
            'empathy_score': 0.15,
            'professionalism_score': 0.10,
            'format_compliance_score': 0.05
        }
        
        total_score = sum(
            getattr(metrics, metric) * weight 
            for metric, weight in weights.items()
        )
        
        return min(1.0, max(0.0, total_score))
    
    def _create_quality_assessment_prompt(
        self, 
        draft_content: str, 
        original_query: str, 
        customer_context: Dict[str, Any]
    ) -> str:
        """Create a detailed prompt for AI quality assessment."""
        
        intent_info = customer_context.get('intent', {})
        if isinstance(intent_info, dict):
            primary_intent = intent_info.get('primary', {})
            if isinstance(primary_intent, dict):
                intent_type = primary_intent.get('type', 'Unknown')
            else:
                intent_type = 'Unknown'
        else:
            intent_type = 'Unknown'
        
        prompt = f"""You are an expert customer service quality assessor for Pompeii3, a jewelry e-commerce brand.
Evaluate the following email draft response comprehensively.

ORIGINAL CUSTOMER QUERY:
{original_query[:1000]}

CUSTOMER CONTEXT:
- Email: {customer_context.get('sender', 'Unknown')}
- Order Number: {customer_context.get('order_number', 'N/A')}
- Intent Category: {intent_type}
- Sender Name: {customer_context.get('sender_name', 'Customer')}

EMAIL DRAFT TO EVALUATE:
{draft_content[:2000]}

Evaluate this email and respond with ONLY valid JSON (no markdown, no extra text):
{{
    "helpfulness_score": <0.0-1.0>,
    "information_density": <0.0-1.0>,
    "specificity_score": <0.0-1.0>,
    "actionability_score": <0.0-1.0>,
    "empathy_score": <0.0-1.0>,
    "professionalism_score": <0.0-1.0>,
    "format_compliance_score": <0.0-1.0>,
    "overall_score": <0.0-1.0>,
    "has_specific_info": <true/false>,
    "has_links": <true/false>,
    "has_greeting": <true/false>,
    "has_closing": <true/false>,
    "word_count": <number>,
    "knowledge_gaps": ["list of specific missing information"],
    "improvement_suggestions": ["list of specific improvements"],
    "reasoning": "brief explanation"
}}

SCORING GUIDELINES:
- 0.9-1.0: Exceptional, exceeds all standards
- 0.7-0.8: Good, meets customer service standards  
- 0.5-0.6: Acceptable but could improve
- 0.3-0.4: Below expectations, needs work
- 0.0-0.2: Poor, major issues

CRITICAL EVALUATION POINTS:
1. Does the response DIRECTLY address the customer's specific question?
2. Does it include relevant order/tracking/product information?
3. Does it provide clear next steps or solutions?
4. Is the tone professional yet warm and empathetic?
5. Does it follow proper email format (greeting, body, closing)?
6. Are helpful links included where appropriate?

Return ONLY the JSON object, no other text."""

        return prompt
    
    def _parse_quality_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the AI quality assessment response with robust error handling."""
        try:
            # Clean the response text
            cleaned = response_text.strip()
            
            # Remove markdown code blocks if present
            if cleaned.startswith('```'):
                cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                cleaned = re.sub(r'\s*```$', '', cleaned)
            
            # Try to extract JSON object
            json_match = re.search(r'\{[\s\S]*\}', cleaned)
            if json_match:
                json_str = json_match.group()
                
                # Fix common JSON issues
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                
                parsed = json.loads(json_str)
                
                # Validate and normalize scores
                score_fields = [
                    'helpfulness_score', 'information_density', 'specificity_score',
                    'actionability_score', 'empathy_score', 'professionalism_score',
                    'format_compliance_score', 'overall_score'
                ]
                for field in score_fields:
                    if field in parsed:
                        try:
                            value = float(parsed[field])
                            parsed[field] = max(0.0, min(1.0, value))
                        except (ValueError, TypeError):
                            parsed[field] = 0.5
                
                # Ensure lists are lists
                for list_field in ['knowledge_gaps', 'improvement_suggestions']:
                    if list_field not in parsed or not isinstance(parsed[list_field], list):
                        parsed[list_field] = []
                
                return parsed
            else:
                self.logger.warning("No JSON found in quality assessment response")
                return {}
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing quality assessment JSON: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Unexpected error parsing response: {e}")
            return {}
    
    def _should_send_draft(self, metrics: QualityMetrics) -> bool:
        """Determine if a draft should be sent based on quality metrics."""
        
        # Check minimum thresholds
        if metrics.overall_score < self.min_overall_score:
            self.logger.debug(f"Overall score {metrics.overall_score:.2f} below threshold {self.min_overall_score}")
            return False
        
        if metrics.helpfulness_score < self.min_helpfulness_score:
            self.logger.debug(f"Helpfulness score {metrics.helpfulness_score:.2f} below threshold {self.min_helpfulness_score}")
            return False
        
        if metrics.information_density < self.min_information_density:
            self.logger.debug(f"Information density {metrics.information_density:.2f} below threshold {self.min_information_density}")
            return False
        
        # Check format compliance
        if metrics.format_compliance_score < self.min_format_compliance:
            self.logger.debug(f"Format compliance {metrics.format_compliance_score:.2f} below threshold {self.min_format_compliance}")
            # Don't fail on format alone, just log warning
            self.logger.warning("Draft has format issues but will proceed")
        
        # Check for critical knowledge gaps
        if metrics.knowledge_gaps:
            critical_keywords = ['urgent', 'critical', 'immediate', 'emergency', 'safety', 'legal']
            critical_gaps = [
                gap for gap in metrics.knowledge_gaps 
                if any(keyword in gap.lower() for keyword in critical_keywords)
            ]
            if critical_gaps:
                self.logger.warning(f"Critical knowledge gaps found: {critical_gaps[:3]}")
                return False
        
        return True
    
    def _track_knowledge_gaps(
        self, 
        knowledge_gaps: List[str], 
        original_query: str, 
        customer_context: Dict[str, Any]
    ) -> None:
        """Track identified knowledge gaps in Google Sheets."""
        if not self.knowledge_gap_sheet:
            self.logger.debug("Knowledge gap tracking skipped - no sheet available")
            return
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            customer_email = customer_context.get('sender', 'Unknown')
            order_number = str(customer_context.get('order_number', ''))
            
            # Get intent category
            intent_info = customer_context.get('intent', {})
            if isinstance(intent_info, dict):
                primary_intent = intent_info.get('primary', {})
                intent_category = primary_intent.get('type', 'general') if isinstance(primary_intent, dict) else 'general'
            else:
                intent_category = 'general'
            
            # Determine urgency level
            urgency = self._determine_urgency(original_query, customer_context)
            
            rows_to_add = []
            for gap in knowledge_gaps[:5]:  # Limit to 5 gaps per email
                gap_key = f"{gap.lower().strip()[:100]}"
                
                if gap_key in self.knowledge_gap_cache:
                    self.knowledge_gap_cache[gap_key] += 1
                else:
                    self.knowledge_gap_cache[gap_key] = 1
                    
                    category = self._categorize_knowledge_gap(gap)
                    suggested_solution = self._generate_knowledge_gap_solution(gap, original_query)
                    
                    row_data = [
                        timestamp,
                        customer_email[:100],
                        order_number[:50],
                        intent_category,
                        original_query[:500],
                        category,
                        gap[:300],
                        urgency,
                        suggested_solution[:300],
                        1,
                        "New",
                        "",
                        "",
                        "",
                        timestamp
                    ]
                    rows_to_add.append(row_data)
            
            # Batch append rows
            if rows_to_add:
                for row in rows_to_add:
                    try:
                        self.knowledge_gap_sheet.append_row(row, value_input_option='RAW')
                    except Exception as e:
                        self.logger.warning(f"Could not append row to sheet: {e}")
                        break
                
                self.logger.info(f"Tracked {len(rows_to_add)} new knowledge gaps")
                
        except Exception as e:
            self.logger.error(f"Error tracking knowledge gaps: {e}")
    
    def _determine_urgency(self, query: str, context: Dict[str, Any]) -> str:
        """Determine urgency level of a customer query."""
        query_lower = query.lower() if query else ""
        
        critical_keywords = [
            'urgent', 'emergency', 'asap', 'immediately', 'critical',
            'wedding today', 'funeral', 'need today', 'tonight', 'this morning'
        ]
        
        high_keywords = [
            'wedding', 'anniversary', 'birthday', 'proposal', 'engagement',
            'tomorrow', 'this week', 'soon', 'quickly', 'rush',
            'important event', 'special occasion', 'deadline', 'gift'
        ]
        
        medium_keywords = [
            'when will', 'status', 'update', 'tracking', 'delayed',
            'where is', 'how long', 'expected', 'estimate'
        ]
        
        if any(keyword in query_lower for keyword in critical_keywords):
            return 'critical'
        elif any(keyword in query_lower for keyword in high_keywords):
            return 'high'
        elif any(keyword in query_lower for keyword in medium_keywords):
            return 'medium'
        else:
            return 'low'
    
    def _categorize_knowledge_gap(self, gap: str) -> str:
        """Categorize a knowledge gap into predefined categories."""
        gap_lower = gap.lower()
        
        category_keywords = {
            'product_info': ['product', 'item', 'specification', 'feature', 'material', 'size', 'color', 'style'],
            'shipping_policy': ['shipping', 'delivery', 'tracking', 'carrier', 'transit', 'shipment'],
            'return_policy': ['return', 'exchange', 'refund', 'warranty', 'defective', 'damaged'],
            'order_processing': ['order', 'processing', 'fulfillment', 'preparation', 'status'],
            'payment_issues': ['payment', 'billing', 'charge', 'invoice', 'credit', 'price', 'cost'],
            'technical_support': ['technical', 'setup', 'installation', 'troubleshoot', 'issue'],
            'inventory_status': ['inventory', 'stock', 'availability', 'restock', 'out of stock'],
            'promotional_codes': ['promo', 'discount', 'coupon', 'sale', 'offer', 'code'],
            'account_management': ['account', 'login', 'password', 'profile', 'email'],
            'custom_orders': ['custom', 'personalization', 'engraving', 'modification', 'special order'],
            'warranty': ['warranty', 'guarantee', 'protection', 'coverage'],
            'sizing': ['size', 'fit', 'measurement', 'resize', 'adjust'],
            'materials': ['material', 'gold', 'silver', 'diamond', 'gemstone', 'metal', 'care']
        }
        
        best_match = 'product_info'
        max_matches = 0
        
        for category, keywords in category_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in gap_lower)
            if matches > max_matches:
                max_matches = matches
                best_match = category
        
        return self.KNOWLEDGE_CATEGORIES.get(best_match, 'General Information')
    
    def _generate_knowledge_gap_solution(self, gap: str, query: str) -> str:
        """Generate a suggested solution for addressing a knowledge gap."""
        gap_lower = gap.lower()
        
        solutions = {
            'product': "Add detailed product specifications to the knowledge base",
            'shipping': "Update shipping policy and timeline documentation",
            'return': "Clarify return/exchange procedures and eligibility",
            'inventory': "Implement real-time inventory status in responses",
            'payment': "Document payment processing and billing procedures",
            'tracking': "Integrate tracking information into automated responses",
            'price': "Add pricing information and payment options",
            'warranty': "Document warranty coverage and claim procedures",
            'size': "Add sizing guides and measurement information",
            'custom': "Document customization options and timelines"
        }
        
        for keyword, solution in solutions.items():
            if keyword in gap_lower:
                return solution
        
        return f"Research and document: {gap[:100]}"
    
    def _generate_improvement_suggestions(
        self, 
        metrics: QualityMetrics, 
        assessment_data: Dict[str, Any]
    ) -> List[str]:
        """Generate specific improvement suggestions based on quality metrics."""
        
        suggestions = []
        
        # Helpfulness improvements
        if metrics.helpfulness_score < 0.7:
            suggestions.append("Address the customer's specific question more directly")
        
        # Information density improvements
        if metrics.information_density < 0.6:
            suggestions.append("Include more substantial, relevant information")
        
        # Specificity improvements
        if metrics.specificity_score < 0.6:
            suggestions.append("Add specific order/product details instead of generic statements")
        
        # Actionability improvements
        if metrics.actionability_score < 0.6:
            suggestions.append("Provide clear next steps or actions for the customer")
        
        # Empathy improvements
        if metrics.empathy_score < 0.6:
            suggestions.append("Acknowledge the customer's concern with more empathetic language")
        
        # Format improvements
        if not metrics.has_greeting:
            suggestions.append("Add a proper greeting (e.g., 'Hello [Name],')")
        
        if not metrics.has_closing:
            suggestions.append("Add a proper closing (e.g., 'Thank You\\nCustomer Support Team')")
        
        if not metrics.has_links and metrics.actionability_score < 0.7:
            suggestions.append("Include relevant support links or tracking URLs")
        
        # Word count
        if metrics.word_count < 30:
            suggestions.append("Provide a more detailed response")
        elif metrics.word_count > 500:
            suggestions.append("Consider making the response more concise")
        
        # Knowledge gaps
        if metrics.knowledge_gaps:
            suggestions.append(f"Address knowledge gaps: {', '.join(metrics.knowledge_gaps[:2])}")
        
        # Add AI reasoning if available
        if metrics.reasoning and len(metrics.reasoning) > 10:
            suggestions.append(f"AI insight: {metrics.reasoning[:150]}")
        
        return suggestions[:6]  # Limit to 6 suggestions
    
    def _generate_cache_key(self, draft: str, query: str) -> str:
        """Generate a cache key for validation results."""
        content = f"{draft[:500]}:{query[:200]}"
        return str(hash(content))
    
    def _get_from_cache(self, key: str) -> Optional[Tuple[bool, QualityMetrics]]:
        """Get validation result from cache if not expired."""
        if key in self.validation_cache and key in self.cache_timestamps:
            if time.time() - self.cache_timestamps[key] < self.cache_ttl:
                return self.validation_cache[key]
            else:
                # Clean up expired entry
                del self.validation_cache[key]
                del self.cache_timestamps[key]
        return None
    
    def _add_to_cache(self, key: str, result: Tuple[bool, QualityMetrics]) -> None:
        """Add validation result to cache."""
        self.validation_cache[key] = result
        self.cache_timestamps[key] = time.time()
        
        # Limit cache size
        if len(self.validation_cache) > 100:
            oldest_key = min(self.cache_timestamps, key=self.cache_timestamps.get)
            del self.validation_cache[oldest_key]
            del self.cache_timestamps[oldest_key]


# =============================================================================
# Public API Functions
# =============================================================================

def validate_draft_quality(
    draft_content: str,
    original_query: str,
    customer_context: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function to validate email draft quality.
    
    This is the main entry point for the email quality validation system.
    
    Args:
        draft_content: The generated email draft
        original_query: The original customer query
        customer_context: Context about the customer and their query
        logger: Optional logger instance
        
    Returns:
        Tuple of (should_send, validation_results)
        - should_send: Boolean indicating if draft meets quality standards
        - validation_results: Dictionary containing detailed metrics and suggestions
    """
    log = logger or logging.getLogger(__name__)
    
    try:
        validator = EmailQualityValidator(logger=log)
        
        should_send, metrics, suggestions = validator.validate_email_draft(
            draft_content, original_query, customer_context
        )
        
        validation_results = {
            'should_send': should_send,
            'quality_metrics': asdict(metrics),
            'improvement_suggestions': suggestions,
            'validation_timestamp': datetime.now().isoformat(),
            'validation_status': metrics.validation_status
        }
        
        return should_send, validation_results
        
    except Exception as e:
        log.error(f"Critical error in validate_draft_quality: {e}", exc_info=True)
        # Return safe defaults
        return True, {
            'should_send': True,
            'quality_metrics': {},
            'improvement_suggestions': ["Validation failed - manual review recommended"],
            'validation_timestamp': datetime.now().isoformat(),
            'validation_status': 'error'
        }


def get_quality_validator(logger: Optional[logging.Logger] = None) -> EmailQualityValidator:
    """
    Get a singleton instance of EmailQualityValidator.
    Useful for reusing the same validator across multiple calls.
    """
    return EmailQualityValidator(logger=logger)
