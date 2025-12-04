"""
Enhanced error handling and recovery module for the Email Agent system.
Provides comprehensive error handling, retry mechanisms, and graceful degradation.
"""

import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timedelta
import json


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    component: str
    customer_email: Optional[str] = None
    message_id: Optional[str] = None
    thread_id: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class RetryableError(Exception):
    """Exception that indicates an operation should be retried."""
    pass


class NonRetryableError(Exception):
    """Exception that indicates an operation should not be retried."""
    pass


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker pattern implementation for external service calls."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                else:
                    raise CircuitBreakerError(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        )
    
    def _on_success(self):
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'


class ErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts = {}
        self.circuit_breakers = {}
        self.fallback_handlers = {}
    
    def register_circuit_breaker(self, service_name: str, failure_threshold: int = 5, recovery_timeout: int = 60):
        """Register a circuit breaker for a service."""
        self.circuit_breakers[service_name] = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
    
    def register_fallback(self, operation: str, fallback_func: Callable):
        """Register a fallback function for an operation."""
        self.fallback_handlers[operation] = fallback_func
    
    def with_retry(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        exceptions: Tuple[type, ...] = (Exception,)
    ):
        """Decorator for retry logic with exponential backoff."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if isinstance(e, NonRetryableError):
                            self.logger.error(f"Non-retryable error in {func.__name__}: {e}")
                            raise e
                        
                        if attempt < max_attempts - 1:
                            wait_time = delay * (backoff_factor ** attempt)
                            self.logger.warning(
                                f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                                f"Retrying in {wait_time:.2f} seconds..."
                            )
                            time.sleep(wait_time)
                        else:
                            self.logger.error(
                                f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                            )
                
                raise last_exception
            
            return wrapper
        return decorator
    
    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        use_fallback: bool = True
    ) -> Optional[Any]:
        """Central error handling with context and fallback support."""
        
        # Log the error with context
        self._log_error(error, context)
        
        # Update error statistics
        self._update_error_stats(context.operation, error)
        
        # Try fallback if available and requested
        if use_fallback and context.operation in self.fallback_handlers:
            try:
                self.logger.info(f"Attempting fallback for {context.operation}")
                return self.fallback_handlers[context.operation](context)
            except Exception as fallback_error:
                self.logger.error(f"Fallback failed for {context.operation}: {fallback_error}")
        
        # Re-raise the original error if no fallback or fallback failed
        raise error
    
    def _log_error(self, error: Exception, context: ErrorContext):
        """Log error with comprehensive context information."""
        error_info = {
            'operation': context.operation,
            'component': context.component,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'customer_email': context.customer_email,
            'message_id': context.message_id,
            'thread_id': context.thread_id,
            'timestamp': context.timestamp,
            'traceback': traceback.format_exc()
        }
        
        self.logger.error(f"Error in {context.operation}: {json.dumps(error_info, indent=2)}")
    
    def _update_error_stats(self, operation: str, error: Exception):
        """Update error statistics for monitoring."""
        if operation not in self.error_counts:
            self.error_counts[operation] = {}
        
        error_type = type(error).__name__
        if error_type not in self.error_counts[operation]:
            self.error_counts[operation][error_type] = 0
        
        self.error_counts[operation][error_type] += 1
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get current error statistics."""
        return {
            'error_counts': self.error_counts,
            'circuit_breaker_states': {
                name: cb.state for name, cb in self.circuit_breakers.items()
            }
        }
    
    def reset_error_stats(self):
        """Reset error statistics."""
        self.error_counts.clear()


# Global error handler instance
error_handler = ErrorHandler()


def with_error_handling(
    operation: str,
    component: str,
    use_fallback: bool = True,
    max_attempts: int = 3
):
    """Decorator that combines error handling with retry logic."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = ErrorContext(
                operation=operation,
                component=component
            )
            
            # Extract context from kwargs if available
            if 'message_id' in kwargs:
                context.message_id = kwargs['message_id']
            if 'thread_id' in kwargs:
                context.thread_id = kwargs['thread_id']
            if 'customer_email' in kwargs:
                context.customer_email = kwargs['customer_email']
            
            @error_handler.with_retry(max_attempts=max_attempts)
            def execute_with_retry():
                return func(*args, **kwargs)
            
            try:
                return execute_with_retry()
            except Exception as e:
                return error_handler.handle_error(e, context, use_fallback)
        
        return wrapper
    return decorator


# Fallback functions for common operations
def email_processing_fallback(context: ErrorContext) -> Dict[str, Any]:
    """Fallback for email processing failures."""
    return {
        'message_id': context.message_id,
        'status': 'fallback_processed',
        'error': 'Primary processing failed, using fallback',
        'timestamp': datetime.now().isoformat()
    }


def quality_validation_fallback(context: ErrorContext) -> Tuple[bool, Dict[str, Any]]:
    """Fallback for quality validation failures."""
    return True, {
        'should_send': True,
        'quality_metrics': {
            'overall_score': 0.5,
            'helpfulness_score': 0.5,
            'fallback_used': True
        },
        'improvement_suggestions': ['Quality validation failed - manual review recommended']
    }


def analytics_logging_fallback(context: ErrorContext) -> bool:
    """Fallback for analytics logging failures."""
    # Log to local file as fallback
    try:
        with open('analytics_fallback.log', 'a') as f:
            f.write(f"{datetime.now().isoformat()}: Analytics logging failed for {context.operation}\n")
        return True
    except Exception:
        return False


def api_request_fallback(context: ErrorContext) -> Dict[str, Any]:
    """Fallback for API request failures."""
    return {
        'error': 'API request failed',
        'fallback_used': True,
        'timestamp': datetime.now().isoformat(),
        'operation': context.operation
    }


# Register fallback handlers
error_handler.register_fallback('email_processing', email_processing_fallback)
error_handler.register_fallback('quality_validation', quality_validation_fallback)
error_handler.register_fallback('analytics_logging', analytics_logging_fallback)
error_handler.register_fallback('api_request', api_request_fallback)

# Register circuit breakers for external services
error_handler.register_circuit_breaker('gemini_api', failure_threshold=3, recovery_timeout=30)
error_handler.register_circuit_breaker('shipstation_api', failure_threshold=5, recovery_timeout=60)
error_handler.register_circuit_breaker('fedex_api', failure_threshold=5, recovery_timeout=60)
error_handler.register_circuit_breaker('google_sheets', failure_threshold=3, recovery_timeout=45)


class HealthChecker:
    """System health monitoring and reporting."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.health_checks = {}
    
    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function."""
        self.health_checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        for name, check_func in self.health_checks.items():
            try:
                is_healthy = check_func()
                results['checks'][name] = {
                    'status': 'healthy' if is_healthy else 'unhealthy',
                    'timestamp': datetime.now().isoformat()
                }
                
                if not is_healthy:
                    results['overall_status'] = 'degraded'
                    
            except Exception as e:
                results['checks'][name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results['overall_status'] = 'unhealthy'
        
        return results
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        return {
            'error_stats': error_handler.get_error_stats(),
            'health_status': self.run_health_checks(),
            'timestamp': datetime.now().isoformat()
        }


# Global health checker instance
health_checker = HealthChecker()


def check_gemini_api_health() -> bool:
    """Health check for Gemini API."""
    try:
        # Simple test request to Gemini API
        import requests
        from config import GEMINI_API_KEY
        
        if not GEMINI_API_KEY:
            return False
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def check_google_sheets_health() -> bool:
    """Health check for Google Sheets integration."""
    try:
        import gspread
        from google.oauth2 import service_account
        from config import SERVICE_ACCOUNT_FILE, SPREADSHEET_ID
        
        if not SERVICE_ACCOUNT_FILE or not SPREADSHEET_ID:
            return False
        
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=['https://spreadsheets.google.com/feeds']
        )
        gc = gspread.authorize(credentials)
        sheet = gc.open_by_key(SPREADSHEET_ID)
        return sheet is not None
    except Exception:
        return False


def check_storage_health() -> bool:
    """Health check for Google Cloud Storage."""
    try:
        from google.cloud import storage
        from config import BUCKET_NAME
        
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        return bucket.exists()
    except Exception:
        return False


# Register health checks
health_checker.register_health_check('gemini_api', check_gemini_api_health)
health_checker.register_health_check('google_sheets', check_google_sheets_health)
health_checker.register_health_check('cloud_storage', check_storage_health)
