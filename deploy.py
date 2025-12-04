#!/usr/bin/env python3
"""
Deployment automation script for the Email Agent system.
Handles Google Cloud deployment, configuration validation, and service setup.
"""

import os
import sys
import json
import subprocess
import argparse
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import shutil
from datetime import datetime


class DeploymentManager:
    """Manages deployment process for the Email Agent system."""
    
    def __init__(self, project_id: str, region: str = 'us-central1', logger: Optional[logging.Logger] = None):
        self.project_id = project_id
        self.region = region
        self.logger = logger or self._setup_logger()
        self.deployment_config = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup deployment logger."""
        logger = logging.getLogger('deployment')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def validate_environment(self) -> bool:
        """Validate deployment environment and prerequisites."""
        self.logger.info("Validating deployment environment...")
        
        # Check required files
        required_files = [
            'main.py',
            'requirements.txt',
            'config.py',
            '.env',
            'service-account.json'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            self.logger.error(f"Missing required files: {missing_files}")
            return False
        
        # Check Google Cloud CLI
        try:
            result = subprocess.run(['gcloud', 'version'], capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error("Google Cloud CLI not found or not working")
                return False
        except FileNotFoundError:
            self.logger.error("Google Cloud CLI not installed")
            return False
        
        # Check project authentication
        try:
            result = subprocess.run(['gcloud', 'config', 'get-value', 'project'], capture_output=True, text=True)
            current_project = result.stdout.strip()
            if current_project != self.project_id:
                self.logger.warning(f"Current project ({current_project}) differs from target ({self.project_id})")
        except Exception as e:
            self.logger.error(f"Error checking project authentication: {e}")
            return False
        
        self.logger.info("Environment validation passed")
        return True
    
    def create_app_yaml(self) -> bool:
        """Create app.yaml for Google App Engine deployment."""
        self.logger.info("Creating app.yaml configuration...")
        
        app_config = {
            'runtime': 'python39',
            'service': 'email-agent',
            'instance_class': 'F2',
            'automatic_scaling': {
                'min_instances': 1,
                'max_instances': 10,
                'target_cpu_utilization': 0.6
            },
            'env_variables': {
                'GOOGLE_CLOUD_PROJECT': self.project_id,
                'ENVIRONMENT': 'production'
            },
            'handlers': [
                {
                    'url': '/health',
                    'script': 'auto'
                },
                {
                    'url': '.*',
                    'script': 'auto'
                }
            ]
        }
        
        try:
            with open('app.yaml', 'w') as f:
                yaml.dump(app_config, f, default_flow_style=False)
            
            self.logger.info("app.yaml created successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error creating app.yaml: {e}")
            return False
    
    def create_cloudbuild_yaml(self) -> bool:
        """Create cloudbuild.yaml for CI/CD pipeline."""
        self.logger.info("Creating cloudbuild.yaml configuration...")
        
        build_config = {
            'steps': [
                {
                    'name': 'python:3.9',
                    'entrypoint': 'pip',
                    'args': ['install', '-r', 'requirements.txt']
                },
                {
                    'name': 'python:3.9',
                    'entrypoint': 'python',
                    'args': ['-m', 'pytest', 'tests/', '-v']
                },
                {
                    'name': 'gcr.io/cloud-builders/gcloud',
                    'args': ['app', 'deploy', '--quiet']
                }
            ],
            'timeout': '1200s'
        }
        
        try:
            with open('cloudbuild.yaml', 'w') as f:
                yaml.dump(build_config, f, default_flow_style=False)
            
            self.logger.info("cloudbuild.yaml created successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error creating cloudbuild.yaml: {e}")
            return False
    
    def setup_cloud_services(self) -> bool:
        """Enable required Google Cloud services."""
        self.logger.info("Setting up Google Cloud services...")
        
        required_services = [
            'appengine.googleapis.com',
            'cloudbuild.googleapis.com',
            'storage-api.googleapis.com',
            'sheets.googleapis.com',
            'gmail.googleapis.com',
            'secretmanager.googleapis.com',
            'logging.googleapis.com',
            'monitoring.googleapis.com'
        ]
        
        for service in required_services:
            try:
                self.logger.info(f"Enabling {service}...")
                result = subprocess.run([
                    'gcloud', 'services', 'enable', service,
                    '--project', self.project_id
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    self.logger.error(f"Failed to enable {service}: {result.stderr}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error enabling {service}: {e}")
                return False
        
        self.logger.info("All required services enabled")
        return True
    
    def create_storage_buckets(self) -> bool:
        """Create required Google Cloud Storage buckets."""
        self.logger.info("Creating storage buckets...")
        
        buckets = [
            f"{self.project_id}-email-agent-data",
            f"{self.project_id}-email-agent-responses",
            f"{self.project_id}-email-agent-vector-db",
            f"{self.project_id}-email-agent-backups"
        ]
        
        for bucket in buckets:
            try:
                self.logger.info(f"Creating bucket {bucket}...")
                result = subprocess.run([
                    'gsutil', 'mb', '-p', self.project_id,
                    '-c', 'STANDARD', '-l', self.region,
                    f'gs://{bucket}'
                ], capture_output=True, text=True)
                
                if result.returncode != 0 and 'already exists' not in result.stderr:
                    self.logger.error(f"Failed to create bucket {bucket}: {result.stderr}")
                    return False
                elif 'already exists' in result.stderr:
                    self.logger.info(f"Bucket {bucket} already exists")
                    
            except Exception as e:
                self.logger.error(f"Error creating bucket {bucket}: {e}")
                return False
        
        self.logger.info("Storage buckets created successfully")
        return True
    
    def setup_secrets(self) -> bool:
        """Setup secrets in Google Secret Manager."""
        self.logger.info("Setting up secrets in Secret Manager...")
        
        # Read environment variables from .env file
        env_vars = {}
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key] = value
        
        # Secrets to store
        secrets = [
            'GEMINI_API_KEY',
            'SHIPSTATION_API_KEY',
            'SHIPSTATION_API_SECRET',
            'FEDEX_CLIENT_ID',
            'FEDEX_CLIENT_SECRET'
        ]
        
        for secret_name in secrets:
            if secret_name in env_vars and env_vars[secret_name]:
                try:
                    self.logger.info(f"Creating secret {secret_name}...")
                    
                    # Create secret
                    result = subprocess.run([
                        'gcloud', 'secrets', 'create', secret_name.lower().replace('_', '-'),
                        '--project', self.project_id
                    ], capture_output=True, text=True)
                    
                    if result.returncode != 0 and 'already exists' not in result.stderr:
                        self.logger.error(f"Failed to create secret {secret_name}: {result.stderr}")
                        continue
                    
                    # Add secret version
                    result = subprocess.run([
                        'gcloud', 'secrets', 'versions', 'add', secret_name.lower().replace('_', '-'),
                        '--data-file=-',
                        '--project', self.project_id
                    ], input=env_vars[secret_name], text=True, capture_output=True)
                    
                    if result.returncode != 0:
                        self.logger.error(f"Failed to add secret version for {secret_name}: {result.stderr}")
                        continue
                        
                except Exception as e:
                    self.logger.error(f"Error setting up secret {secret_name}: {e}")
                    continue
        
        self.logger.info("Secrets setup completed")
        return True
    
    def deploy_to_app_engine(self) -> bool:
        """Deploy application to Google App Engine."""
        self.logger.info("Deploying to Google App Engine...")
        
        try:
            result = subprocess.run([
                'gcloud', 'app', 'deploy',
                '--project', self.project_id,
                '--quiet'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Deployment failed: {result.stderr}")
                return False
            
            self.logger.info("Deployment to App Engine successful")
            self.logger.info(f"Application URL: https://{self.project_id}.appspot.com")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during deployment: {e}")
            return False
    
    def setup_monitoring(self) -> bool:
        """Setup monitoring and alerting."""
        self.logger.info("Setting up monitoring and alerting...")
        
        # Create monitoring dashboard configuration
        dashboard_config = {
            "displayName": "Email Agent Dashboard",
            "mosaicLayout": {
                "tiles": [
                    {
                        "width": 6,
                        "height": 4,
                        "widget": {
                            "title": "Request Rate",
                            "xyChart": {
                                "dataSets": [{
                                    "timeSeriesQuery": {
                                        "timeSeriesFilter": {
                                            "filter": f'resource.type="gae_app" AND resource.label.project_id="{self.project_id}"',
                                            "aggregation": {
                                                "alignmentPeriod": "60s",
                                                "perSeriesAligner": "ALIGN_RATE"
                                            }
                                        }
                                    }
                                }]
                            }
                        }
                    }
                ]
            }
        }
        
        try:
            with open('monitoring_dashboard.json', 'w') as f:
                json.dump(dashboard_config, f, indent=2)
            
            self.logger.info("Monitoring configuration created")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up monitoring: {e}")
            return False
    
    def run_deployment(self, skip_tests: bool = False) -> bool:
        """Run complete deployment process."""
        self.logger.info(f"Starting deployment to project: {self.project_id}")
        
        steps = [
            ("Validating environment", self.validate_environment),
            ("Creating app.yaml", self.create_app_yaml),
            ("Creating cloudbuild.yaml", self.create_cloudbuild_yaml),
            ("Setting up cloud services", self.setup_cloud_services),
            ("Creating storage buckets", self.create_storage_buckets),
            ("Setting up secrets", self.setup_secrets),
            ("Setting up monitoring", self.setup_monitoring),
            ("Deploying to App Engine", self.deploy_to_app_engine)
        ]
        
        if not skip_tests:
            # Run tests before deployment
            self.logger.info("Running tests...")
            try:
                result = subprocess.run(['python', '-m', 'pytest', 'tests/', '-v'], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    self.logger.error(f"Tests failed: {result.stdout}\n{result.stderr}")
                    return False
                self.logger.info("All tests passed")
            except Exception as e:
                self.logger.error(f"Error running tests: {e}")
                return False
        
        # Execute deployment steps
        for step_name, step_func in steps:
            self.logger.info(f"Executing: {step_name}")
            if not step_func():
                self.logger.error(f"Failed at step: {step_name}")
                return False
        
        self.logger.info("Deployment completed successfully!")
        self.logger.info(f"Application URL: https://{self.project_id}.appspot.com")
        
        return True
    
    def rollback_deployment(self, version: str = None) -> bool:
        """Rollback to previous deployment version."""
        self.logger.info("Rolling back deployment...")
        
        try:
            if version:
                result = subprocess.run([
                    'gcloud', 'app', 'services', 'set-traffic', 'email-agent',
                    f'--splits={version}=1',
                    '--project', self.project_id
                ], capture_output=True, text=True)
            else:
                # Get previous version
                result = subprocess.run([
                    'gcloud', 'app', 'versions', 'list',
                    '--service=email-agent',
                    '--sort-by=~version.createTime',
                    '--limit=2',
                    '--format=value(version.id)',
                    '--project', self.project_id
                ], capture_output=True, text=True)
                
                versions = result.stdout.strip().split('\n')
                if len(versions) >= 2:
                    previous_version = versions[1]
                    result = subprocess.run([
                        'gcloud', 'app', 'services', 'set-traffic', 'email-agent',
                        f'--splits={previous_version}=1',
                        '--project', self.project_id
                    ], capture_output=True, text=True)
                else:
                    self.logger.error("No previous version found for rollback")
                    return False
            
            if result.returncode != 0:
                self.logger.error(f"Rollback failed: {result.stderr}")
                return False
            
            self.logger.info("Rollback completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during rollback: {e}")
            return False


def main():
    """Main deployment script entry point."""
    parser = argparse.ArgumentParser(description='Deploy Email Agent to Google Cloud')
    parser.add_argument('--project-id', required=True, help='Google Cloud Project ID')
    parser.add_argument('--region', default='us-central1', help='Deployment region')
    parser.add_argument('--skip-tests', action='store_true', help='Skip running tests')
    parser.add_argument('--rollback', help='Rollback to specific version')
    parser.add_argument('--dry-run', action='store_true', help='Validate without deploying')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create deployment manager
    deployer = DeploymentManager(args.project_id, args.region)
    
    try:
        if args.rollback:
            success = deployer.rollback_deployment(args.rollback)
        elif args.dry_run:
            success = deployer.validate_environment()
        else:
            success = deployer.run_deployment(args.skip_tests)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        deployer.logger.info("Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        deployer.logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
