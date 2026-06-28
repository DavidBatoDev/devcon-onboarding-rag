"""
Compilation script for the LlamaIndex RAG system (OpenAI-powered).
Processes Google Drive documents and builds the RAG index using OpenAI
embeddings (text-embedding-3-small) and the OpenAI LLM (gpt-4o-mini).
"""

import sys
import logging
from pathlib import Path
from typing import List
import time
from llama_index.core import Settings as LlamaSettings

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.services.llamaindex_service import rag_service
from app.services.document_processor import document_processor
from app.services.drive_loader import get_drive_files
from app.core.config import settings

# Setup logging with better encoding handling
def setup_logging():
    """Setup logging with proper encoding for Windows console"""
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler (can handle unicode)
    file_handler = logging.FileHandler('compilation.log', encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    
    # Console handler (avoid emojis for Windows compatibility)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    # Setup root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()


def print_banner():
    """Print a nice banner for the compilation process"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    DEVCON RAG COMPILER                       ║
    ║                   Powered by OpenAI                          ║
    ║          gpt-4o-mini  +  text-embedding-3-small             ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def validate_environment():
    """Validate that all required environment variables are set"""
    logger.info("Validating environment...")

    required_settings = [
        ("OPENAI_API_KEY", settings.OPENAI_API_KEY),
    ]

    # Google Drive must be configured via one of the supported methods.
    if not settings.google_service_account_configured:
        required_settings.append(("GOOGLE service account (any method)", None))

    missing_vars = [name for name, value in required_settings if not value]

    if missing_vars:
        logger.error(f"Missing required configuration values: {missing_vars}")
        logger.error("OPENAI_API_KEY and a Google Drive service account are required")
        sys.exit(1)

    logger.info("Environment validation passed")
    logger.info(f"LLM: {settings.OPENAI_LLM_MODEL} | Embeddings: {settings.OPENAI_EMBED_MODEL}")


def test_embedding_model():
    """Test the OpenAI embedding model via a single sample call."""
    logger.info("Testing OpenAI embedding model...")

    try:
        embed_model = LlamaSettings.embed_model
        embedding = embed_model.get_text_embedding(
            "Testing OpenAI embeddings for DEVCON documents"
        )

        if embedding and len(embedding) > 0:
            logger.info("OpenAI embedding model working successfully")
            logger.info(f"Model: {settings.OPENAI_EMBED_MODEL}")
            logger.info(f"Embedding dimensions: {len(embedding)}")
            if len(embedding) != settings.EMBEDDING_DIM:
                logger.warning(
                    f"Embedding dim {len(embedding)} != configured EMBEDDING_DIM "
                    f"{settings.EMBEDDING_DIM}; update config if you changed the model."
                )
            return True
        logger.error("Embedding test failed: No embedding returned")
        return False

    except Exception as e:
        logger.error(f"Error testing OpenAI embedding model: {e}")
        return False


def get_compilation_stats():
    """Get current compilation statistics"""
    try:
        stats = rag_service.get_index_stats()
        logger.info("Current Index Stats:")
        logger.info(f"  Status: {stats.get('status', 'unknown')}")
        logger.info(f"  Documents: {stats.get('document_count', 0)}")
        logger.info(f"  Embedding Model: {stats.get('embedding_model', 'unknown')}")
        logger.info(f"  Embedding API: {stats.get('embedding_api', 'unknown')}")
        logger.info(f"  Embedding Dims: {stats.get('embedding_dimensions', 'unknown')}")
        logger.info(f"  Chunk Size: {stats.get('chunk_size', 'unknown')}")
        return stats
    except Exception as e:
        logger.warning(f"Could not get compilation stats: {e}")
        return {}


def fetch_and_process_documents(folder_id: str = None) -> List:
    """Fetch documents from Google Drive and process them"""
    logger.info("Fetching documents from Google Drive...")
    
    try:
        # Get files from Google Drive
        files = get_drive_files(folder_id)
        logger.info(f"Found {len(files)} files in Google Drive")
        
        if not files:
            logger.warning("No files found in Google Drive")
            return []
        
        # Log file details
        logger.info("Files to process:")
        for i, file in enumerate(files[:10], 1):  # Show first 10 files
            name = file.get('name', 'Unknown')
            mime_type = file.get('mimeType', 'Unknown')
            size = file.get('size', 'Unknown')
            logger.info(f"  {i}. {name} ({mime_type}) - {size} bytes")
        
        if len(files) > 10:
            logger.info(f"  ... and {len(files) - 10} more files")
        
        # Process files into LlamaIndex Documents
        logger.info("Processing documents...")
        start_time = time.time()
        
        documents = document_processor.process_drive_files(files)
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(documents)} documents in {processing_time:.2f} seconds")
        
        # Log processing results
        success_rate = (len(documents) / len(files)) * 100 if files else 0
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        return documents
        
    except Exception as e:
        logger.error(f"Error fetching/processing documents: {e}")
        raise


def estimate_api_usage(documents: List):
    """Estimate HF Inference API usage"""
    logger.info("Estimating HF Inference API usage...")
    
    try:
        total_chars = sum(len(doc.text) for doc in documents)
        estimated_chunks = total_chars // settings.CHUNK_SIZE + len(documents)
        
        logger.info("Usage estimates:")
        logger.info(f"  Total characters: {total_chars:,}")
        logger.info(f"  Estimated chunks: {estimated_chunks:,}")
        logger.info(f"  API calls needed: ~{estimated_chunks}")
        
        if estimated_chunks > 1000:
            logger.warning("Large number of API calls required")
            logger.warning("Consider processing in smaller batches to respect rate limits")
            return estimated_chunks // 50  # Suggest batch size
        elif estimated_chunks > 500:
            logger.info("Moderate API usage expected")
            return estimated_chunks // 25
        else:
            logger.info("Light API usage expected")
            return None
            
    except Exception as e:
        logger.warning(f"Could not estimate API usage: {e}")
        return None


def build_rag_index(documents: List, batch_size: int = None):
    """Build the RAG index with processed documents using HF Inference API"""
    if not documents:
        logger.warning("No documents to add to index")
        return False
    
    logger.info(f"Building RAG index with {len(documents)} documents...")
    logger.info("Using HF Inference API - embeddings computed in the cloud")
    logger.info("Processing time depends on API response times and rate limits")
    
    start_time = time.time()
    
    try:
        # Estimate and suggest batch processing if needed
        suggested_batch_size = estimate_api_usage(documents)
        if suggested_batch_size and not batch_size:
            batch_size = suggested_batch_size
            logger.info(f"Using suggested batch size: {batch_size}")
        
        # Process documents in batches if specified
        if batch_size and len(documents) > batch_size:
            logger.info(f"Processing documents in batches of {batch_size}")
            logger.info("This helps respect API rate limits and provides better progress tracking")
            
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                
                batch_start = time.time()
                success = rag_service.add_documents(batch)
                batch_time = time.time() - batch_start
                
                if not success:
                    logger.error(f"Failed to process batch {batch_num}")
                    return False
                
                # Show progress and timing
                processed = min(i + batch_size, len(documents))
                progress = (processed / len(documents)) * 100
                logger.info(f"Progress: {processed}/{len(documents)} ({progress:.1f}%)")
                logger.info(f"Batch processed in {batch_time:.1f}s")
                
                # Brief pause between batches to be respectful to API
                if batch_num < total_batches:
                    logger.info("Brief pause between batches...")
                    time.sleep(2)
        else:
            # Process all documents at once
            logger.info("Processing all documents in single batch")
            success = rag_service.add_documents(documents)
            if not success:
                logger.error("Failed to build RAG index")
                return False
        
        build_time = time.time() - start_time
        logger.info(f"RAG index built successfully in {build_time:.2f} seconds")
        
        # Calculate processing rate
        avg_time_per_doc = build_time / len(documents) if len(documents) > 0 else 0
        logger.info(f"Average processing time: {avg_time_per_doc:.2f} seconds/document")
        
        return True
            
    except Exception as e:
        logger.error(f"Error building RAG index: {e}")
        if "rate limit" in str(e).lower():
            logger.error("This might be due to API rate limits")
            logger.error("Try using smaller batch sizes or waiting before retrying")
        raise


def test_rag_system():
    """Test the RAG system with sample queries"""
    logger.info("Testing RAG system with HF Inference API embeddings...")
    
    test_queries = [
        "What is DEVCON?",
        "Tell me about the event schedule",
        "Who are the speakers?",
        "What are the main topics discussed?",
        "How to apply in DEVCON Internship program?",
        "Who is Dom and what is his role and achievements?"
    ]
    
    total_query_time = 0
    
    for i, query in enumerate(test_queries, 1):
        try:
            logger.info(f"Test {i}: {query}")
            start_time = time.time()
            
            response = rag_service.query(query)
            
            query_time = time.time() - start_time
            total_query_time += query_time
            
            # Log first 100 characters of response
            preview = response[:100] + "..." if len(response) > 100 else response
            logger.info(f"Response ({query_time:.2f}s): {preview}")
            
        except Exception as e:
            logger.error(f"Test {i} failed: {e}")
    
    if total_query_time > 0:
        avg_query_time = total_query_time / len(test_queries)
        logger.info(f"Average query time: {avg_query_time:.2f}s")


def cleanup_temp_files():
    """Clean up any temporary files created during processing"""
    logger.info("Cleaning up temporary files...")
    
    try:
        # Remove any temp files that might have been created
        temp_patterns = ["temp_*.docx", "temp_*.pdf", "*.tmp"]
        
        for pattern in temp_patterns:
            for temp_file in Path(".").glob(pattern):
                temp_file.unlink()
                logger.info(f"Removed: {temp_file}")
                
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")


def main():
    """Main compilation process"""
    print_banner()
    
    try:
        # Validate environment
        validate_environment()

        # Test embedding model
        if not test_embedding_model():
            logger.error("OpenAI embedding test failed. Exiting.")
            sys.exit(1)
        
        # Get initial stats
        logger.info("Getting initial statistics...")
        initial_stats = get_compilation_stats()        # Fetch and process documents
        # You can pass a specific folder_id here if needed
        folder_id = settings.GOOGLE_DRIVE_FOLDER_ID  # Get from settings
        documents = fetch_and_process_documents(folder_id)
        
        if not documents:
            logger.warning("No documents to process. Exiting.")
            return
        
        # Build RAG index with HF Inference API
        # Use smaller batch size for API rate limiting
        batch_size = 25 if len(documents) > 50 else None
        success = build_rag_index(documents, batch_size)
        
        if not success:
            logger.error("Failed to build RAG index")
            sys.exit(1)
        
        # Get final stats
        logger.info("Getting final statistics...")
        final_stats = get_compilation_stats()
        
        # Test the system
        test_rag_system()
        
        # Cleanup
        cleanup_temp_files()
        
        # Final summary
        logger.info("Compilation completed successfully!")
        logger.info(f"Final document count: {final_stats.get('document_count', 0)}")
        logger.info(f"Embedding model: {final_stats.get('embedding_model', 'unknown')}")
        logger.info(f"Embedding API: {final_stats.get('embedding_api', 'unknown')}")
        
        print("""
        ╔══════════════════════════════════════════════════════════════╗
        ║                   COMPILATION COMPLETE                       ║
        ║                                                              ║
        ║  Your DEVCON RAG system with HF Inference API is ready!     ║
        ║  Cloud-powered BGE embeddings for excellent search quality  ║
        ║  Start the server with: python main.py                      ║
        ╚══════════════════════════════════════════════════════════════╝
        """)
        
    except KeyboardInterrupt:
        logger.info("Compilation interrupted by user")
        cleanup_temp_files()
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Compilation failed: {e}")
        if "api" in str(e).lower() or "rate" in str(e).lower():
            logger.error("This appears to be an API-related error")
            logger.error("Please check your OPENAI_API_KEY and internet connection")
        cleanup_temp_files()
        sys.exit(1)


if __name__ == "__main__":
    main()