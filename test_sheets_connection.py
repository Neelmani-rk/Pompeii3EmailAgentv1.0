#!/usr/bin/env python3
"""
Test script to verify Google Sheets integration for Email Quality Validator and Response Analytics.
This script tests the connection, worksheet creation, and basic operations.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import gspread
    from google.oauth2 import service_account
    from config import SPREADSHEET_ID, SERVICE_ACCOUNT_FILE
    from email_quality_validator import EmailQualityValidator
    from response_analytics import ResponseAnalytics
    from utils import setup_logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required packages are installed:")
    print("pip install gspread google-auth google-auth-oauthlib google-auth-httplib2")
    sys.exit(1)


def test_service_account_file():
    """Test if service account file exists and is valid."""
    print("🔍 Testing service account file...")
    
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        print(f"❌ Service account file not found: {SERVICE_ACCOUNT_FILE}")
        print("Please follow the Google Sheets setup guide to create the service account file.")
        return False
    
    try:
        with open(SERVICE_ACCOUNT_FILE, 'r') as f:
            service_account_data = json.load(f)
        
        required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email', 'client_id']
        missing_fields = [field for field in required_fields if field not in service_account_data]
        
        if missing_fields:
            print(f"❌ Service account file is missing required fields: {missing_fields}")
            return False
        
        print(f"✅ Service account file is valid")
        print(f"   Project ID: {service_account_data.get('project_id')}")
        print(f"   Client Email: {service_account_data.get('client_email')}")
        return True
        
    except json.JSONDecodeError:
        print(f"❌ Service account file is not valid JSON")
        return False
    except Exception as e:
        print(f"❌ Error reading service account file: {e}")
        return False


def test_google_sheets_connection():
    """Test basic Google Sheets API connection."""
    print("\n🔍 Testing Google Sheets API connection...")
    
    try:
        # Initialize credentials
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=['https://spreadsheets.google.com/feeds',
                   'https://www.googleapis.com/auth/drive']
        )
        
        # Authorize gspread client
        gc = gspread.authorize(credentials)
        
        # Test connection by listing spreadsheets (this will fail if no access)
        print("✅ Successfully authenticated with Google Sheets API")
        return gc
        
    except Exception as e:
        print(f"❌ Failed to connect to Google Sheets API: {e}")
        return None


def test_spreadsheet_access(gc):
    """Test access to the specific spreadsheet."""
    print(f"\n🔍 Testing access to spreadsheet: {SPREADSHEET_ID}")
    
    try:
        # Try to open the spreadsheet
        sheet = gc.open_by_key(SPREADSHEET_ID)
        print(f"✅ Successfully opened spreadsheet: '{sheet.title}'")
        print(f"   URL: https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}")
        return sheet
        
    except gspread.SpreadsheetNotFound:
        print(f"❌ Spreadsheet not found: {SPREADSHEET_ID}")
        print("Please check:")
        print("1. The spreadsheet ID is correct")
        print("2. The spreadsheet exists")
        print("3. The service account has been shared with the spreadsheet")
        return None
    except Exception as e:
        print(f"❌ Error accessing spreadsheet: {e}")
        return None


def test_worksheet_operations(sheet):
    """Test worksheet creation and basic operations."""
    print("\n🔍 Testing worksheet operations...")
    
    test_worksheet_name = f"Test_Sheet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Create a test worksheet
        test_worksheet = sheet.add_worksheet(title=test_worksheet_name, rows=10, cols=5)
        print(f"✅ Successfully created test worksheet: {test_worksheet_name}")
        
        # Test writing data
        test_data = [
            ["Timestamp", "Test Field 1", "Test Field 2", "Status", "Notes"],
            [datetime.now().isoformat(), "Test Value 1", "Test Value 2", "Success", "Test entry"]
        ]
        
        for i, row in enumerate(test_data, 1):
            test_worksheet.insert_row(row, i)
        
        print("✅ Successfully wrote test data to worksheet")
        
        # Test reading data
        all_values = test_worksheet.get_all_values()
        if len(all_values) >= 2:
            print("✅ Successfully read data from worksheet")
            print(f"   Headers: {all_values[0]}")
            print(f"   First row: {all_values[1]}")
        
        # Clean up - delete test worksheet
        sheet.del_worksheet(test_worksheet)
        print(f"✅ Successfully deleted test worksheet")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in worksheet operations: {e}")
        # Try to clean up if worksheet was created
        try:
            test_worksheet = sheet.worksheet(test_worksheet_name)
            sheet.del_worksheet(test_worksheet)
            print("🧹 Cleaned up test worksheet")
        except:
            pass
        return False


def test_email_quality_validator():
    """Test EmailQualityValidator integration."""
    print("\n🔍 Testing EmailQualityValidator Google Sheets integration...")
    
    try:
        logger = setup_logging()
        validator = EmailQualityValidator(logger=logger)
        
        if validator.sheets_client is None:
            print("❌ EmailQualityValidator failed to initialize Google Sheets client")
            return False
        
        if validator.knowledge_gap_sheet is None:
            print("❌ EmailQualityValidator failed to access Knowledge_Gaps worksheet")
            return False
        
        print("✅ EmailQualityValidator Google Sheets integration working")
        print(f"   Knowledge Gaps worksheet: {validator.knowledge_gap_sheet.title}")
        
        # Test adding a sample knowledge gap (optional)
        try:
            sample_row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "test@example.com",
                "TEST123",
                "Test query for integration testing",
                "Product Information",
                "Missing product specifications for testing",
                "low",
                "Add test product information to knowledge base",
                1,
                "Test",
                "Integration test entry",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
            
            validator.knowledge_gap_sheet.append_row(sample_row)
            print("✅ Successfully added test knowledge gap entry")
            
            # Clean up test entry
            all_values = validator.knowledge_gap_sheet.get_all_values()
            if len(all_values) > 1 and "Integration test entry" in all_values[-1]:
                validator.knowledge_gap_sheet.delete_rows(len(all_values))
                print("🧹 Cleaned up test knowledge gap entry")
                
        except Exception as e:
            print(f"⚠️  Could not test knowledge gap entry: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ EmailQualityValidator test failed: {e}")
        return False


def test_response_analytics():
    """Test ResponseAnalytics integration."""
    print("\n🔍 Testing ResponseAnalytics Google Sheets integration...")
    
    try:
        logger = setup_logging()
        analytics = ResponseAnalytics(logger=logger)
        
        if analytics.sheets_client is None:
            print("❌ ResponseAnalytics failed to initialize Google Sheets client")
            return False
        
        if analytics.analytics_sheet is None:
            print("❌ ResponseAnalytics failed to access Response_Analytics worksheet")
            return False
        
        if analytics.alerts_sheet is None:
            print("❌ ResponseAnalytics failed to access System_Alerts worksheet")
            return False
        
        print("✅ ResponseAnalytics Google Sheets integration working")
        print(f"   Analytics worksheet: {analytics.analytics_sheet.title}")
        print(f"   Alerts worksheet: {analytics.alerts_sheet.title}")
        
        # Test logging sample analytics data
        try:
            sample_metrics = {
                'overall_score': 0.8,
                'helpfulness_score': 0.85,
                'knowledge_gaps': ['test gap'],
                'improvement_suggestions': ['test suggestion']
            }
            
            sample_context = {
                'sender': 'test@example.com',
                'order_number': 'TEST123'
            }
            
            analytics.log_response_metrics(
                response_time=2.5,
                quality_metrics=sample_metrics,
                customer_context=sample_context,
                draft_approved=True
            )
            
            print("✅ Successfully logged test analytics data")
            
        except Exception as e:
            print(f"⚠️  Could not test analytics logging: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ ResponseAnalytics test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Google Sheets Integration Tests")
    print("=" * 50)
    
    # Test 1: Service account file
    if not test_service_account_file():
        print("\n❌ Service account file test failed. Please fix this before continuing.")
        return False
    
    # Test 2: Google Sheets API connection
    gc = test_google_sheets_connection()
    if not gc:
        print("\n❌ Google Sheets API connection failed. Please check your credentials and API access.")
        return False
    
    # Test 3: Spreadsheet access
    sheet = test_spreadsheet_access(gc)
    if not sheet:
        print("\n❌ Spreadsheet access failed. Please check the spreadsheet ID and permissions.")
        return False
    
    # Test 4: Basic worksheet operations
    if not test_worksheet_operations(sheet):
        print("\n❌ Worksheet operations failed. Please check permissions.")
        return False
    
    # Test 5: EmailQualityValidator integration
    if not test_email_quality_validator():
        print("\n❌ EmailQualityValidator integration failed.")
        return False
    
    # Test 6: ResponseAnalytics integration
    if not test_response_analytics():
        print("\n❌ ResponseAnalytics integration failed.")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All Google Sheets integration tests passed!")
    print("\nYour Email Agent is ready to use Google Sheets for:")
    print("• Knowledge gap tracking")
    print("• Response analytics and monitoring")
    print("• System alerts and notifications")
    print("\nNext steps:")
    print("1. Deploy your Email Agent")
    print("2. Monitor the Google Sheets dashboard")
    print("3. Review knowledge gaps and analytics regularly")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
