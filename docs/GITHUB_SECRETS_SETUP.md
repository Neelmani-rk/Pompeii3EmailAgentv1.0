# GitHub Secrets Setup Guide for GCP Deployment

This document explains how to set up the required GitHub Secrets for the automated GCP deployment workflow.

## Required GitHub Secrets

Navigate to your repository: **Settings → Secrets and variables → Actions → New repository secret**

### GCP Authentication

| Secret Name | Description | How to Obtain |
|-------------|-------------|---------------|
| `GCP_PROJECT_ID` | Your Google Cloud Project ID | GCP Console → Project selector → Copy project ID (e.g., `dotted-electron-447414-m1`) |
| `GCP_SA_KEY` | Service Account JSON key | GCP Console → IAM → Service Accounts → Create key → JSON format |

### API Keys

| Secret Name | Description |
|-------------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key for AI features |
| `SHIPSTATION_API_KEY` | ShipStation API key for order integration |
| `SHIPSTATION_API_SECRET` | ShipStation API secret |
| `FEDEX_CLIENT_ID` | FedEx API client ID for tracking |
| `FEDEX_CLIENT_SECRET` | FedEx API client secret |

### Application Configuration

| Secret Name | Description |
|-------------|-------------|
| `SPREADSHEET_ID` | Google Sheets spreadsheet ID for data storage |

## Optional GitHub Variables

Navigate to: **Settings → Secrets and variables → Actions → Variables tab → New repository variable**

| Variable Name | Default Value | Description |
|---------------|---------------|-------------|
| `GCP_REGION` | `us-central1` | GCP region for deployment |

## Service Account Permissions

The GCP service account (`GCP_SA_KEY`) needs the following roles:

```
- Cloud Functions Developer
- Cloud Run Admin
- Artifact Registry Administrator
- Storage Admin
- Pub/Sub Admin
- Service Account User
```

### Creating the Service Account

```bash
# Create service account
gcloud iam service-accounts create github-actions-deployer \
    --description="GitHub Actions deployment account" \
    --display-name="GitHub Actions Deployer"

# Grant roles
PROJECT_ID="your-project-id"
SA_EMAIL="github-actions-deployer@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/cloudfunctions.developer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/artifactregistry.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/pubsub.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/iam.serviceAccountUser"

# Create and download key
gcloud iam service-accounts keys create key.json \
    --iam-account=${SA_EMAIL}

# The contents of key.json should be added as GCP_SA_KEY secret
```

## Running the Workflow

1. Go to **Actions** tab in your GitHub repository
2. Select **"Deploy Email Agent to GCP"** workflow
3. Click **"Run workflow"**
4. Configure options:
   - **Environment**: production/staging/development
   - **Deploy Cloud Functions**: true/false
   - **Deploy Cloud Run**: true/false
   - **Function name**: Custom name or use default
   - **Cloud Run service**: Custom name or use default
5. Click **"Run workflow"**

## Workflow Features

- ✅ **Manual trigger only** - No automatic deployments
- ✅ **Selective deployment** - Deploy Functions, Cloud Run, or both
- ✅ **Environment variables** - All secrets passed securely
- ✅ **Change detection** - Identifies modified and new files
- ✅ **Deployment summary** - Post-deployment report with URLs
- ✅ **Multi-stage Docker build** - Optimized container images

## File Structure

```
.github/
└── workflows/
    └── deploy-gcp.yml     # Main deployment workflow
Dockerfile                  # Docker configuration for Cloud Run
.env                        # Local environment variables (not committed)
GITHUB_SECRETS_SETUP.md     # This file
```
