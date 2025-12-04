"""
Response Analytics and Monitoring Module for the Email Agent system.
Tracks performance metrics, quality trends, and provides operational insights.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import gspread
from google.oauth2 import service_account

from config import SPREADSHEET_ID
from utils import setup_logging


@dataclass
class ResponseMetrics:
    """Metrics for email response performance."""
    timestamp: str
    response_time_seconds: float
    quality_score: float
    helpfulness_score: float
    customer_satisfaction: Optional[float]
    knowledge_gaps_found: int
    improvement_suggestions_count: int
    draft_approved: bool
    customer_email: str
    order_number: Optional[str]
    response_category: str


@dataclass
class AlertCondition:
    """Defines conditions for triggering alerts."""
    metric_name: str
    threshold_value: float
    comparison_operator: str  # 'lt', 'gt', 'eq'
    alert_message: str
    severity: str  # 'low', 'medium', 'high', 'critical'


class ResponseAnalytics:
    """Analytics and monitoring system for email responses."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the analytics system."""
        self.logger = logger or setup_logging()
        
        # Initialize Google Sheets client
        self.sheets_client = None
        self.analytics_sheet = None
        self.alerts_sheet = None
        self._init_sheets_client()
        
        # Performance tracking
        self.daily_metrics = []
        self.alert_conditions = self._setup_default_alerts()
        
        # Cache for performance
        self.metrics_cache = {}
        self.last_cache_update = None
    
    def _init_sheets_client(self):
        """Initialize Google Sheets client for analytics tracking."""
        try:
            from config import SERVICE_ACCOUNT_FILE
            if SERVICE_ACCOUNT_FILE and os.path.exists(SERVICE_ACCOUNT_FILE):
                credentials = service_account.Credentials.from_service_account_file(
                    SERVICE_ACCOUNT_FILE,
                    scopes=['https://spreadsheets.google.com/feeds',
                           'https://www.googleapis.com/auth/drive']
                )
                gc = gspread.authorize(credentials)
            else:
                # Use default service account (for Google Cloud environments)
                gc = gspread.service_account()
            
            # Open or create analytics sheets
            try:
                sheet = gc.open_by_key(SPREADSHEET_ID)
                
                # Analytics sheet
                try:
                    self.analytics_sheet = sheet.worksheet("Response_Analytics")
                except gspread.WorksheetNotFound:
                    self.analytics_sheet = sheet.add_worksheet(
                        title="Response_Analytics", rows=1000, cols=15
                    )
                    self._setup_analytics_headers()
                
                # Alerts sheet
                try:
                    self.alerts_sheet = sheet.worksheet("System_Alerts")
                except gspread.WorksheetNotFound:
                    self.alerts_sheet = sheet.add_worksheet(
                        title="System_Alerts", rows=1000, cols=10
                    )
                    self._setup_alerts_headers()
                    
            except gspread.SpreadsheetNotFound:
                self.logger.error(f"Spreadsheet not found: {SPREADSHEET_ID}")
                self.sheets_client = None
                return
            
            self.sheets_client = gc
            self.logger.info("Analytics Google Sheets client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize analytics Google Sheets client: {e}")
            self.sheets_client = None
    
    def _setup_analytics_headers(self):
        """Set up headers for the analytics tracking sheet."""
        headers = [
            "Timestamp", "Response Time (s)", "Quality Score", "Helpfulness Score",
            "Customer Satisfaction", "Knowledge Gaps Found", "Improvement Suggestions",
            "Draft Approved", "Customer Email", "Order Number", "Response Category",
            "Date", "Hour", "Day of Week", "Notes"
        ]
        if self.analytics_sheet:
            self.analytics_sheet.append_row(headers)
    
    def _setup_alerts_headers(self):
        """Set up headers for the alerts tracking sheet."""
        headers = [
            "Timestamp", "Alert Type", "Severity", "Message", "Metric Value",
            "Threshold", "Customer Email", "Order Number", "Status", "Resolution Notes"
        ]
        if self.alerts_sheet:
            self.alerts_sheet.append_row(headers)
    
    def _setup_default_alerts(self) -> List[AlertCondition]:
        """Set up default alert conditions."""
        return [
            AlertCondition(
                metric_name="quality_score",
                threshold_value=0.5,
                comparison_operator="lt",
                alert_message="Low quality score detected",
                severity="medium"
            ),
            AlertCondition(
                metric_name="response_time_seconds",
                threshold_value=30.0,
                comparison_operator="gt",
                alert_message="High response time detected",
                severity="low"
            ),
            AlertCondition(
                metric_name="knowledge_gaps_found",
                threshold_value=3,
                comparison_operator="gt",
                alert_message="Multiple knowledge gaps in single response",
                severity="high"
            ),
            AlertCondition(
                metric_name="helpfulness_score",
                threshold_value=0.4,
                comparison_operator="lt",
                alert_message="Very low helpfulness score",
                severity="high"
            )
        ]
    
    def log_response_metrics(
        self,
        response_time: float,
        quality_metrics: Dict[str, Any],
        customer_context: Dict[str, Any],
        draft_approved: bool
    ):
        """
        Log response metrics for analytics.
        
        Args:
            response_time: Time taken to generate response in seconds
            quality_metrics: Quality assessment metrics
            customer_context: Customer and order context
            draft_approved: Whether the draft was approved for sending
        """
        try:
            timestamp = datetime.now()
            
            # Create metrics object
            metrics = ResponseMetrics(
                timestamp=timestamp.isoformat(),
                response_time_seconds=response_time,
                quality_score=quality_metrics.get('overall_score', 0.0),
                helpfulness_score=quality_metrics.get('helpfulness_score', 0.0),
                customer_satisfaction=None,  # To be updated later if available
                knowledge_gaps_found=len(quality_metrics.get('knowledge_gaps', [])),
                improvement_suggestions_count=len(quality_metrics.get('improvement_suggestions', [])),
                draft_approved=draft_approved,
                customer_email=customer_context.get('sender', 'Unknown'),
                order_number=customer_context.get('order_number'),
                response_category=customer_context.get('intent', {}).get('primary', {}).get('type', 'Unknown')
            )
            
            # Add to daily metrics
            self.daily_metrics.append(metrics)
            
            # Log to Google Sheets
            self._log_to_sheets(metrics, timestamp)
            
            # Check for alerts
            self._check_alert_conditions(metrics)
            
            self.logger.info(f"Logged response metrics for {metrics.customer_email}")
            
        except Exception as e:
            self.logger.error(f"Error logging response metrics: {e}")
    
    def _log_to_sheets(self, metrics: ResponseMetrics, timestamp: datetime):
        """Log metrics to Google Sheets."""
        if not self.analytics_sheet:
            return
        
        try:
            row_data = [
                metrics.timestamp,
                metrics.response_time_seconds,
                metrics.quality_score,
                metrics.helpfulness_score,
                metrics.customer_satisfaction or "",
                metrics.knowledge_gaps_found,
                metrics.improvement_suggestions_count,
                "Yes" if metrics.draft_approved else "No",
                metrics.customer_email,
                metrics.order_number or "",
                metrics.response_category,
                timestamp.strftime("%Y-%m-%d"),
                timestamp.hour,
                timestamp.strftime("%A"),
                ""  # Notes
            ]
            
            self.analytics_sheet.append_row(row_data)
            
        except Exception as e:
            self.logger.error(f"Error logging to analytics sheet: {e}")
    
    def _check_alert_conditions(self, metrics: ResponseMetrics):
        """Check if any alert conditions are met."""
        for condition in self.alert_conditions:
            try:
                metric_value = getattr(metrics, condition.metric_name, None)
                if metric_value is None:
                    continue
                
                should_alert = False
                
                if condition.comparison_operator == 'lt' and metric_value < condition.threshold_value:
                    should_alert = True
                elif condition.comparison_operator == 'gt' and metric_value > condition.threshold_value:
                    should_alert = True
                elif condition.comparison_operator == 'eq' and metric_value == condition.threshold_value:
                    should_alert = True
                
                if should_alert:
                    self._trigger_alert(condition, metric_value, metrics)
                    
            except Exception as e:
                self.logger.error(f"Error checking alert condition {condition.metric_name}: {e}")
    
    def _trigger_alert(self, condition: AlertCondition, metric_value: float, metrics: ResponseMetrics):
        """Trigger an alert when conditions are met."""
        try:
            alert_timestamp = datetime.now().isoformat()
            
            # Log alert
            self.logger.warning(
                f"ALERT [{condition.severity.upper()}]: {condition.alert_message} "
                f"(Value: {metric_value}, Threshold: {condition.threshold_value}) "
                f"for customer {metrics.customer_email}"
            )
            
            # Log to alerts sheet
            if self.alerts_sheet:
                alert_row = [
                    alert_timestamp,
                    condition.metric_name,
                    condition.severity,
                    condition.alert_message,
                    metric_value,
                    condition.threshold_value,
                    metrics.customer_email,
                    metrics.order_number or "",
                    "New",
                    ""  # Resolution notes
                ]
                self.alerts_sheet.append_row(alert_row)
            
        except Exception as e:
            self.logger.error(f"Error triggering alert: {e}")
    
    def get_daily_summary(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get daily performance summary.
        
        Args:
            date: Date to get summary for (defaults to today)
            
        Returns:
            Dictionary with daily summary metrics
        """
        if date is None:
            date = datetime.now()
        
        try:
            # Filter metrics for the specified date
            date_str = date.strftime("%Y-%m-%d")
            daily_metrics = [
                m for m in self.daily_metrics 
                if m.timestamp.startswith(date_str)
            ]
            
            if not daily_metrics:
                return {
                    'date': date_str,
                    'total_responses': 0,
                    'message': 'No responses processed today'
                }
            
            # Calculate summary statistics
            total_responses = len(daily_metrics)
            avg_quality = sum(m.quality_score for m in daily_metrics) / total_responses
            avg_helpfulness = sum(m.helpfulness_score for m in daily_metrics) / total_responses
            avg_response_time = sum(m.response_time_seconds for m in daily_metrics) / total_responses
            
            approved_count = sum(1 for m in daily_metrics if m.draft_approved)
            approval_rate = approved_count / total_responses if total_responses > 0 else 0
            
            total_knowledge_gaps = sum(m.knowledge_gaps_found for m in daily_metrics)
            
            return {
                'date': date_str,
                'total_responses': total_responses,
                'average_quality_score': round(avg_quality, 3),
                'average_helpfulness_score': round(avg_helpfulness, 3),
                'average_response_time': round(avg_response_time, 2),
                'approval_rate': round(approval_rate, 3),
                'total_knowledge_gaps': total_knowledge_gaps,
                'approved_responses': approved_count,
                'rejected_responses': total_responses - approved_count
            }
            
        except Exception as e:
            self.logger.error(f"Error generating daily summary: {e}")
            return {'error': str(e)}
    
    def get_weekly_trends(self) -> Dict[str, Any]:
        """Get weekly performance trends."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            # Filter metrics for the past week
            weekly_metrics = [
                m for m in self.daily_metrics 
                if start_date <= datetime.fromisoformat(m.timestamp.split('T')[0]) <= end_date
            ]
            
            if not weekly_metrics:
                return {'message': 'No data available for the past week'}
            
            # Group by day
            daily_summaries = {}
            for i in range(7):
                current_date = start_date + timedelta(days=i)
                date_str = current_date.strftime("%Y-%m-%d")
                day_metrics = [
                    m for m in weekly_metrics 
                    if m.timestamp.startswith(date_str)
                ]
                
                if day_metrics:
                    daily_summaries[date_str] = {
                        'responses': len(day_metrics),
                        'avg_quality': sum(m.quality_score for m in day_metrics) / len(day_metrics),
                        'approval_rate': sum(1 for m in day_metrics if m.draft_approved) / len(day_metrics)
                    }
                else:
                    daily_summaries[date_str] = {
                        'responses': 0,
                        'avg_quality': 0,
                        'approval_rate': 0
                    }
            
            return {
                'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                'daily_summaries': daily_summaries,
                'total_responses': len(weekly_metrics),
                'week_avg_quality': sum(m.quality_score for m in weekly_metrics) / len(weekly_metrics),
                'week_approval_rate': sum(1 for m in weekly_metrics if m.draft_approved) / len(weekly_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating weekly trends: {e}")
            return {'error': str(e)}
    
    def export_metrics(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Export metrics for a date range.
        
        Args:
            start_date: Start date for export
            end_date: End date for export
            
        Returns:
            List of metrics dictionaries
        """
        try:
            filtered_metrics = [
                m for m in self.daily_metrics 
                if start_date <= datetime.fromisoformat(m.timestamp.split('T')[0]) <= end_date
            ]
            
            return [asdict(m) for m in filtered_metrics]
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            return []
    
    def add_custom_alert(self, alert_condition: AlertCondition):
        """Add a custom alert condition."""
        self.alert_conditions.append(alert_condition)
        self.logger.info(f"Added custom alert: {alert_condition.alert_message}")
    
    def update_customer_satisfaction(self, customer_email: str, satisfaction_score: float):
        """
        Update customer satisfaction score for recent responses.
        
        Args:
            customer_email: Customer email address
            satisfaction_score: Satisfaction score (0.0 - 1.0)
        """
        try:
            # Find recent metrics for this customer
            recent_metrics = [
                m for m in self.daily_metrics[-50:]  # Check last 50 responses
                if m.customer_email == customer_email and m.customer_satisfaction is None
            ]
            
            if recent_metrics:
                # Update the most recent one
                recent_metrics[-1].customer_satisfaction = satisfaction_score
                self.logger.info(f"Updated satisfaction score for {customer_email}: {satisfaction_score}")
            
        except Exception as e:
            self.logger.error(f"Error updating customer satisfaction: {e}")


def log_response_analytics(
    response_time: float,
    quality_metrics: Dict[str, Any],
    customer_context: Dict[str, Any],
    draft_approved: bool,
    logger: Optional[logging.Logger] = None
):
    """
    Convenience function to log response analytics.
    
    Args:
        response_time: Time taken to generate response
        quality_metrics: Quality assessment metrics
        customer_context: Customer context information
        draft_approved: Whether draft was approved
        logger: Optional logger instance
    """
    analytics = ResponseAnalytics(logger=logger)
    analytics.log_response_metrics(
        response_time, quality_metrics, customer_context, draft_approved
    )
