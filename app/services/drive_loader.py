# backend/app/services/drive_loader.py
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from app.core.config import settings
import logging
import os
import json

logger = logging.getLogger(__name__)


def _is_retryable_http_error(exc: BaseException) -> bool:
    """Retry only on transient Google API errors (rate limit / server errors)."""
    if isinstance(exc, HttpError):
        status = getattr(getattr(exc, "resp", None), "status", None)
        try:
            status = int(status)
        except (TypeError, ValueError):
            return False
        return status == 429 or status >= 500
    return False


@retry(
    reraise=True,
    stop=stop_after_attempt(settings.DRIVE_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, max=30),
    retry=retry_if_exception(_is_retryable_http_error),
)
def _execute_with_retry(request):
    """Execute a Google API request with exponential backoff on transient errors."""
    return request.execute()

def get_drive_service():
    """Create and return Google Drive service client"""
    try:
        scopes = ["https://www.googleapis.com/auth/drive.readonly"]
        
        # Method 1: Try individual environment variables
        project_id = getattr(settings, 'GOOGLE_PROJECT_ID', None) or os.getenv("GOOGLE_PROJECT_ID")
        client_email = getattr(settings, 'GOOGLE_CLIENT_EMAIL', None) or os.getenv("GOOGLE_CLIENT_EMAIL")
        private_key = getattr(settings, 'GOOGLE_PRIVATE_KEY', None) or os.getenv("GOOGLE_PRIVATE_KEY")
        private_key_id = getattr(settings, 'GOOGLE_PRIVATE_KEY_ID', None) or os.getenv("GOOGLE_PRIVATE_KEY_ID")
        client_id = getattr(settings, 'GOOGLE_CLIENT_ID', None) or os.getenv("GOOGLE_CLIENT_ID")
        
        if all([project_id, client_email, private_key, private_key_id, client_id]):
            # Format the private key properly (add BEGIN/END markers if missing)
            if not private_key.startswith("-----BEGIN PRIVATE KEY-----"):
                private_key = f"-----BEGIN PRIVATE KEY-----\n{private_key}\n-----END PRIVATE KEY-----"
            
            # Create service account info dictionary
            service_account_info = {
                "type": "service_account",
                "project_id": project_id,
                "private_key_id": private_key_id,
                "private_key": private_key,
                "client_email": client_email,
                "client_id": client_id,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{client_email.replace('@', '%40')}",
                "universe_domain": "googleapis.com"
            }
            
            creds = service_account.Credentials.from_service_account_info(service_account_info, scopes=scopes)
            logger.info("✅ Google Drive service initialized using individual environment variables")
            return build("drive", "v3", credentials=creds)
        
        # If none of the methods work, raise an error
        raise ValueError("No valid Google service account configuration found. Please check your environment variables.")
        
    except Exception as e:
        logger.error(f"❌ Error initializing Google Drive service: {e}")
        raise


def get_drive_files(folder_id: str = None):
    """
    Get files from Google Drive folder
    
    Args:
        folder_id: Google Drive folder ID. If None, gets files from root or configured folder
    
    Returns:
        List of file dictionaries with id, name, mimeType, and size
    """
    try:
        service = get_drive_service()
          # Build query
        if folder_id:
            query = f"'{folder_id}' in parents and trashed = false"
        elif settings.GOOGLE_DRIVE_FOLDER_ID:
            # Use default folder ID from settings
            query = f"'{settings.GOOGLE_DRIVE_FOLDER_ID}' in parents and trashed = false"
            logger.info(f"📁 Using default folder ID: {settings.GOOGLE_DRIVE_FOLDER_ID}")
        else:
            # If no folder_id provided, get files from root
            query = "trashed = false"
        
        logger.info(f"🔍 Searching Google Drive with query: {query}")
        
        files = []
        page_token = None

        while True:
            results = _execute_with_retry(service.files().list(
                q=query,
                pageSize=1000,
                fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, parents)",
                pageToken=page_token
            ))

            batch_files = results.get("files", [])
            files.extend(batch_files)
            
            logger.info(f"📁 Found {len(batch_files)} files in this batch")
            
            page_token = results.get("nextPageToken", None)
            if not page_token:
                break

        logger.info(f"📋 Total files found: {len(files)}")
        
        # Filter for document types that can be processed
        supported_types = [
            'application/pdf',
            'application/vnd.google-apps.document',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'text/plain',
            'application/vnd.google-apps.presentation',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'application/vnd.ms-powerpoint'
        ]
        
        filtered_files = [f for f in files if f.get('mimeType') in supported_types]
        
        if len(filtered_files) != len(files):
            logger.info(f"📄 Filtered to {len(filtered_files)} supported document types")
        
        return filtered_files

    except Exception as e:
        logger.error(f"❌ Error getting files from Google Drive: {e}")
        raise

# Alias for backward compatibility
def list_files(folder_id: str):
    """Backward compatibility alias for get_drive_files"""
    return get_drive_files(folder_id)