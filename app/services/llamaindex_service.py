# backend/app/services/llamaindex_service.py
from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    Settings as LlamaSettings,
    StorageContext,
    load_index_from_storage
)
from dotenv import load_dotenv
    
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response import Response
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import PromptTemplate
from llama_index.core.schema import QueryBundle
from typing import List, Optional, Dict

import chromadb
from chromadb.config import Settings as ChromaSettings
import os
import time
import hashlib
import threading
from typing import List, Optional, Any
from pathlib import Path
from cachetools import TTLCache
from app.core.config import settings
import logging
import json
from app.services.prompt_engine import DEVCONPromptEngine

logger = logging.getLogger(__name__)


load_dotenv() 

class LlamaIndexRAGService:
    def __init__(self):
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine = None
        # Query-result cache (TTL + size bounded). TTLCache is not thread-safe,
        # so all access is guarded by this lock (queries run in worker threads).
        self._query_cache = TTLCache(maxsize=settings.QUERY_CACHE_SIZE, ttl=settings.QUERY_CACHE_TTL)
        self._query_cache_lock = threading.Lock()
        self._setup_llama_index()
        self._initialize_components()
        self.index_state_file = Path(settings.RAG_INDEX_DIR) / "index_state.json"
        self._load_index_state()
        self.prompt_engine = DEVCONPromptEngine(self)
        self._ensure_directories_exist()
    
    def _setup_llama_index(self):
        """Configure LlamaIndex global settings"""
        if not settings.OPENAI_API_KEY:
            logger.warning("⚠️ OPENAI_API_KEY is not set — LLM and embedding calls will fail until it is configured.")

        # Initialize OpenAI LLM
        llm = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.MAX_OUTPUT_TOKENS,
            timeout=settings.OPENAI_REQUEST_TIMEOUT,
            max_retries=settings.OPENAI_MAX_RETRIES,
        )

        # Initialize OpenAI embeddings
        embed_model = OpenAIEmbedding(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_EMBED_MODEL,
            embed_batch_size=settings.EMBEDDING_BATCH_SIZE,
            timeout=settings.OPENAI_REQUEST_TIMEOUT,
            max_retries=settings.OPENAI_MAX_RETRIES,
        )

        # Configure global settings
        LlamaSettings.llm = llm
        LlamaSettings.embed_model = embed_model
        LlamaSettings.chunk_size = settings.CHUNK_SIZE
        LlamaSettings.chunk_overlap = settings.CHUNK_OVERLAP

        print(f"✅ Initialized OpenAI LLM ({settings.OPENAI_LLM_MODEL}) and embeddings ({settings.OPENAI_EMBED_MODEL})")
    
    def _initialize_components(self):
        """Initialize vector store and other components"""
        # Setup ChromaDB
        chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get or create collection with NO embedding function
        collection_name = settings.CHROMA_COLLECTION_NAME  # OpenAI 1536-dim collection
        try:
            chroma_collection = chroma_client.get_collection(collection_name)
            print(f"✅ Loaded existing ChromaDB collection: {collection_name}")
        except:
            # Create collection with NO embedding function - we'll provide pre-computed embeddings
            chroma_collection = chroma_client.create_collection(
                collection_name, 
                embedding_function=None
            )
            print(f"✅ Created new ChromaDB collection: {collection_name}")
        
        # Create vector store
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Try to load existing index
        self._load_or_create_index()

    
    def _load_or_create_index(self):
        """Load existing index or create a new one"""
        storage_dir = Path(settings.RAG_INDEX_DIR)
        
        
        try:
            self._ensure_directories_exist()
            if storage_dir.exists() and any(storage_dir.iterdir()):
                # Load existing index
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(storage_dir),
                    vector_store=self.vector_store
                )
                self.index = load_index_from_storage(storage_context)
                print("✅ Loaded existing LlamaIndex")
            else:
                # Create new index
                storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                self.index = VectorStoreIndex([], storage_context=storage_context)
                print("✅ Created new LlamaIndex")
                
        except Exception as e:
            print(f"⚠️ Error loading index, creating new one: {e}")
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.index = VectorStoreIndex([], storage_context=storage_context)
        
        self._setup_query_engine()

    
    def _load_index_state(self):
        """Load index state tracking file modifications"""
        self.index_state = {}
        if self.index_state_file.exists():
            try:
                with open(self.index_state_file, "r") as f:
                    self.index_state = json.load(f)
                logger.info(f"Loaded index state with {len(self.index_state)} files")
            except Exception as e:
                logger.error(f"Error loading index state: {e}")
    

    def save_index_state(self):
        """Save the current index state to file"""
        try:
            # Ensure the directory exists before saving
            self.index_state_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.index_state_file, 'w') as f:
                json.dump(self.index_state, f, indent=2, default=str)
            logger.debug(f"Index state saved to {self.index_state_file}")
        except Exception as e:
            logger.error(f"Error saving index state: {e}")
            # Don't raise the exception - just log it so the rebuild can continue

    def get_index_state(self):
        """Return current index state"""
        return self.index_state

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """Sanitize metadata to ensure ChromaDB compatibility"""
        sanitized = {}
        
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = None
            elif isinstance(value, (str, int, float)):
                sanitized[key] = value
            elif isinstance(value, bool):
                sanitized[key] = str(value)
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                sanitized[key] = ", ".join(str(item) for item in value) if value else ""
            elif isinstance(value, dict):
                # Convert dicts to JSON-like string representation
                sanitized[key] = str(value)
            else:
                # Convert any other type to string
                sanitized[key] = str(value)
        
        return sanitized

    def _deduplicate_sources(self, source_nodes, max_sources: int = 3):
        """Deduplicate source nodes based on title and file_id"""
        seen_sources = set()
        unique_sources = []
        
        for node in source_nodes:
            source_info = node.metadata.get('title', node.metadata.get('source', 'Unknown'))
            file_id = node.metadata.get('file_id')
            
            # Create a unique identifier for this source
            source_key = f"{source_info}_{file_id}"
            
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                unique_sources.append(node)
            
            # Limit to max_sources
            if len(unique_sources) >= max_sources:
                break
        
        return unique_sources

    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the index with metadata sanitization"""
        try:
            if not self.index:
                self._load_or_create_index()
            
            # Sanitize metadata for all documents before processing
            print("🧹 Sanitizing document metadata for ChromaDB compatibility...")
            for doc in documents:
                if hasattr(doc, 'metadata') and doc.metadata:
                    doc.metadata = self._sanitize_metadata(doc.metadata)
                    print(f"📄 Sanitized metadata for document: {doc.metadata.get('title', 'Unknown')[:50]}...")
            
            # Use sentence splitter for better chunking
            parser = SentenceSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            
            # Parse documents into nodes
            nodes = parser.get_nodes_from_documents(documents)
            print(f"📄 Created {len(nodes)} nodes from {len(documents)} documents")
            
            # Sanitize node metadata as well
            print("🧹 Sanitizing node metadata...")
            for node in nodes:
                if hasattr(node, 'metadata') and node.metadata:
                    node.metadata = self._sanitize_metadata(node.metadata)
            
            # Add to index with progress tracking
            print("🔄 Adding nodes to index...")
            self.index.insert_nodes(nodes)
            
            # Persist the index
            self.persist_index()
            
            # Refresh query engine
            self._setup_query_engine()

            for doc in documents:
                file_id = doc.metadata["file_id"]
                self.index_state[file_id] = {
                    "file_id": file_id,
                    "file_name": doc.metadata["source"],
                    "title": doc.metadata.get("title", doc.metadata["source"]),
                    "modified_time": doc.metadata["modified_time"],
                    "chunk_count": len(nodes)  # Assuming nodes are created
                }
            
            self.save_index_state()
            return True
            
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            # Print more detailed error info for debugging
            import traceback
            print(f"🔍 Full error traceback:")
            traceback.print_exc()
            return False
        
    def remove_document(self, file_id: str) -> bool:
        """Remove a document from the index by file ID"""
        try:
            if self.index:
                # Delete from index
                self.index.delete_ref_doc(file_id, delete_from_docstore=True)
                
                # Delete from state
                if file_id in self.index_state:
                    del self.index_state[file_id]
                
                logger.info(f"✅ Removed document {file_id} from index")
                return True
        except Exception as e:
            logger.error(f"❌ Error removing document {file_id}: {e}")
        return False

    def _setup_query_engine(self):
        """Setup a baseline query engine (used as a readiness check).

        The main answering path uses _retrieve_and_synthesize(), which adds
        query condensation and reranking on top of this retriever.
        """
        if not self.index:
            return

        # Retrieve a wide candidate set for better recall.
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=settings.RETRIEVE_CANDIDATES,
        )

        postprocessor = SimilarityPostprocessor(similarity_cutoff=settings.SIMILARITY_CUTOFF)

        # Create query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[postprocessor],
        )

        print(f"✅ Query engine configured with similarity_cutoff={settings.SIMILARITY_CUTOFF}, "
              f"candidates={settings.RETRIEVE_CANDIDATES}")

    def _condense_question(self, question: str, history: List[Dict[str, str]]) -> str:
        """Rewrite a follow-up question into a standalone retrieval query using
        recent conversation history. Improves retrieval for pronoun/elliptical
        follow-ups (e.g. "who runs it?")."""
        if not history or not settings.ENABLE_QUERY_CONDENSE:
            return question
        try:
            turns = history[-settings.CONDENSE_HISTORY_TURNS:]
            convo = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in turns
            )
            prompt = (
                "Given the conversation below and a follow-up message, rewrite the follow-up "
                "as a standalone, self-contained search query. Resolve pronouns and references "
                "using the conversation. Return ONLY the rewritten query, nothing else.\n\n"
                f"Conversation:\n{convo}\n\nFollow-up: {question}\n\nStandalone query:"
            )
            condensed = str(LlamaSettings.llm.complete(prompt)).strip()
            # Guard against the model returning something empty or absurdly long
            if condensed and len(condensed) <= 1000:
                logger.info(f"🔁 Condensed query: {condensed}")
                return condensed
        except Exception as e:
            logger.warning(f"⚠️ Query condensation failed, using original question: {e}")
        return question

    def _postprocess_nodes(self, nodes, query_str: str, top_k: int):
        """Apply similarity cutoff then (optionally) LLM reranking, returning the
        final top_k nodes used for both the context prompt and synthesis."""
        if not nodes:
            return nodes
        query_bundle = QueryBundle(query_str)

        # 1. Drop clearly-irrelevant nodes by similarity score.
        try:
            nodes = SimilarityPostprocessor(
                similarity_cutoff=settings.SIMILARITY_CUTOFF
            ).postprocess_nodes(nodes, query_bundle=query_bundle)
        except Exception as e:
            logger.warning(f"⚠️ Similarity postprocessing failed: {e}")

        # 2. Rerank the survivors with the LLM for relevance, keep top_k.
        if settings.ENABLE_RERANK and nodes:
            try:
                from llama_index.core.postprocessor import LLMRerank
                reranker = LLMRerank(top_n=top_k, choice_batch_size=5)
                nodes = reranker.postprocess_nodes(nodes, query_bundle=query_bundle)
            except Exception as e:
                logger.warning(f"⚠️ LLM rerank failed, falling back to similarity order: {e}")
                nodes = nodes[:top_k]
        else:
            nodes = nodes[:top_k]
        return nodes

    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history into a readable context string"""
        if not history:
            return ""
        
        formatted_history = []
        # Use more history to better detect continuing conversations
        for msg in history[-10:]:  # Use last 10 exchanges for better context
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted_history)

    def _create_history_aware_prompt(self, question: str, history: List[Dict[str, str]]) -> str:
        """Create a context-aware prompt that considers conversation history"""
        if not history:
            return question
        
        # Format recent conversation
        conversation_context = self._format_conversation_history(history)
        
        # Create enhanced prompt that considers context
        enhanced_prompt = f"""
    Previous conversation context:
    {conversation_context}

    Current question: {question}

    Please provide a helpful response considering the conversation history above. 
    If the current question refers to something mentioned earlier, use that context.
    If it's a new topic, focus on the current question.
    """
        
        return enhanced_prompt

    def _get_retriever(self, candidates: int):
        """Build the retriever. Vector-only by default; hybrid (BM25 + vector)
        when ENABLE_HYBRID_SEARCH is on and an in-memory node corpus is available."""
        vector_retriever = VectorIndexRetriever(index=self.index, similarity_top_k=candidates)
        if not settings.ENABLE_HYBRID_SEARCH:
            return vector_retriever
        try:
            from llama_index.retrievers.bm25 import BM25Retriever
            from llama_index.core.retrievers import QueryFusionRetriever
            nodes = list(self.index.docstore.docs.values())
            if not nodes:
                logger.warning("⚠️ Hybrid search on but docstore is empty; using vector retriever only.")
                return vector_retriever
            bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=settings.BM25_TOP_K)
            return QueryFusionRetriever(
                [vector_retriever, bm25],
                similarity_top_k=candidates,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False,
            )
        except Exception as e:
            logger.warning(f"⚠️ Hybrid search unavailable ({e}); using vector retriever only.")
            return vector_retriever

    def _create_history_aware_anti_hallucination_engine(self, question: str, history: List[Dict[str, str]] = None,
                                                        top_k: int = None, temperature: float = None):
        """Retrieve, rerank, and synthesize a grounded answer that is aware of
        conversation history. Returns ``(response, error, retrieval_query)`` where
        ``response`` is an already-executed LlamaIndex Response (or None on error)."""
        # Format conversation history
        conversation_context = self._format_conversation_history(history) if history else ""

        # Condense follow-ups into a standalone retrieval query (history-aware).
        retrieval_query = self._condense_question(question, history) if history else question

        # Retrieve a wide candidate set, then rerank/filter down to top_k.
        final_top_k = top_k or settings.TOP_K_RETRIEVAL
        retriever = self._get_retriever(settings.RETRIEVE_CANDIDATES)
        nodes = retriever.retrieve(retrieval_query)
        nodes = self._postprocess_nodes(nodes, retrieval_query, final_top_k)

        # Check if we have meaningful context
        has_relevant_context = bool(nodes) and any(
            node.text.strip() for node in nodes if node.text.strip()
        )

        if not nodes:
            return None, "No relevant documents found for your question.", retrieval_query

        # Create context-aware prompt using the enhanced prompt engine if available
        if hasattr(self, 'prompt_engine') and self.prompt_engine:
            # Use the enhanced context prompt that handles both scenarios
            context_prompt = self.prompt_engine.create_enhanced_context_prompt(nodes, question, has_relevant_context, conversation_context)
        else:
            # Fallback to basic context prompt
            context_prompt = self._create_basic_context_prompt(nodes, question, has_relevant_context)

        # Updated QA template with enhanced context prioritization
        qa_template_str = """{context_prompt}

💡 CRITICAL RESPONSE RULES:
1. 🎯 ALWAYS check the provided context FIRST for relevant information
2. ✅ If context contains helpful information, use it and cite it naturally
3. 🚫 If context is empty/irrelevant/insufficient, use general knowledge
4. 📢 When using general knowledge, ALWAYS state: "Based on general knowledge..."
5. 🤷 Be honest about information sources - never make up information
6. 😊 Keep responses friendly and helpful with minimal emojis
7. 🎯 Focus on practical, actionable advice for chapter officers

🚫 STRICT PROHIBITIONS:
- NEVER reintroduce yourself (no "I'm DEBBIE" or similar)
- NEVER use greetings like "Hey there" in continuing conversations
- NEVER mention your name unless explicitly asked
- Respond directly to the question in continuing conversations

🔍 SOURCE HANDLING:
- If using context: "According to the documents..."
- If using general knowledge: "I don't have specific documentation, but generally..."
- If unsure: "I don't have enough information about this specific topic"

🔍 REFERENCE DECISION:
At the end, add EXACTLY ONE marker:
- "SHOW_REFERENCES: true" if answer references specific documents
- "SHOW_REFERENCES: false" if using general knowledge

Answer:"""
        
        # Insert the context prompt (which now includes conversation history if needed)
        formatted_template = qa_template_str.replace("{context_prompt}", context_prompt)

        qa_template = PromptTemplate(formatted_template)

        # Optional per-request temperature override.
        synth_llm = None
        if temperature is not None:
            try:
                synth_llm = OpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    model=settings.OPENAI_LLM_MODEL,
                    temperature=temperature,
                    max_tokens=settings.MAX_OUTPUT_TOKENS,
                    timeout=settings.OPENAI_REQUEST_TIMEOUT,
                    max_retries=settings.OPENAI_MAX_RETRIES,
                )
            except Exception as e:
                logger.warning(f"⚠️ Could not apply temperature override: {e}")

        # Create response synthesizer with the enhanced template and synthesize
        # directly over the already-reranked nodes (avoids a second retrieval).
        response_synthesizer = get_response_synthesizer(
            text_qa_template=qa_template,
            response_mode="compact",
            llm=synth_llm,
        )

        response = response_synthesizer.synthesize(question, nodes=nodes)
        return response, None, retrieval_query

    def _create_basic_context_prompt(self, context_nodes, question: str, has_relevant_context: bool) -> str:
        """Fallback context prompt when prompt engine is not available"""
        
        # Extract and format context with source tracking
        context_with_sources = []
        for i, node in enumerate(context_nodes, 1):
            source = node.metadata.get('title', node.metadata.get('source', 'Unknown'))
            context_with_sources.append(f"[Source {i}: {source}]\n{node.text}\n")
        
        full_context = "\n---\n".join(context_with_sources)
        
        if has_relevant_context:
            return f"""I have found relevant information in the provided documents. Use this context as your PRIMARY source:

CONTEXT FROM DOCUMENTS:
{full_context}

QUESTION: {question}

📋 RESPONSE REQUIREMENTS:
1. 🎯 Use the context above as your main source of information
2. ✅ When referencing context, say: "According to the documents..." or "I found this in the materials..."
3. 😊 Keep responses friendly and engaging with emojis
4. 🎯 Focus on practical, actionable advice for chapter officers
5. 📝 Be specific about which parts of the context you're using"""
        else:
            return f"""I have checked the available documents but couldn't find specific information related to your question. Provide guidance using general knowledge about DEVCON and chapter management.

QUESTION: {question}

📋 RESPONSE REQUIREMENTS:
1. 🚫 Start with: "I don't have specific information about this in the provided documents, but based on general knowledge..."
2. 😊 Keep responses friendly and engaging with emojis
3. 🎯 Focus on practical, actionable advice for chapter officers
4. 📝 Be transparent that you're using general knowledge
5. 🤔 If unsure about something, say so rather than guessing"""


    def query_with_history_and_verification(self, question: str, history: List[Dict[str, str]] = None,
                                            top_k: int = None, temperature: float = None) -> str:
        """Main query method that handles conversation history AND prevents hallucinations"""
        try:
            if not self.query_engine:
                return "❌ RAG system not initialized. Please add documents first."

            if history is None:
                history = []

            print(f"🔍 Processing query with {len(history)} history messages")

            # Retrieve, rerank and synthesize a grounded answer (history-aware).
            response, error, final_query = self._create_history_aware_anti_hallucination_engine(
                question, history, top_k=top_k, temperature=temperature
            )

            if error:
                return f"❌ {error}"

            # Verify response quality
            if not response.response or response.response.strip() in ["Empty Response", "", "I don't know"]:
                return "❌ I don't have enough relevant information in the available documents to answer your question accurately."
            
            # Check for low-confidence responses (this is actually good - shows honesty)
            if any(phrase in response.response.lower() for phrase in [
                "i don't have that information",
                "not mentioned in the context",
                "the provided documents don't contain",
                "i don't have specific information about this in the provided documents"
            ]):
                # Model is being appropriately cautious - this is good behavior
                pass
            
            # Parse the show_reference decision from the response
            response_text = response.response
            show_references = False
            
            # Look for the reference decision marker
            if "SHOW_REFERENCES: true" in response_text:
                show_references = True
                # Remove the marker from the response
                response_text = response_text.replace("SHOW_REFERENCES: true", "").strip()
            elif "SHOW_REFERENCES: false" in response_text:
                show_references = False
                # Remove the marker from the response
                response_text = response_text.replace("SHOW_REFERENCES: false", "").strip()
            else:
                # Fallback: if no marker found, default to showing references if we have source nodes
                show_references = hasattr(response, 'source_nodes') and response.source_nodes
            
            # Start building the final answer
            answer = f"{response_text}\n\n"
            
            # Only add source information if the LLM decided to show references
            if show_references and hasattr(response, 'source_nodes') and response.source_nodes:
                # Use the deduplication helper
                unique_sources = self._deduplicate_sources(response.source_nodes, max_sources=3)
                
                if unique_sources:
                    answer += "📚 **Sources Referenced:**\n"
                    for i, node in enumerate(unique_sources, 1):
                        source_info = node.metadata.get('title', node.metadata.get('source', 'Unknown'))
                        file_id = node.metadata.get('file_id')
                        
                        answer += f"{i}. {source_info}\n"
                        
                        if file_id:
                            answer += f"   [[View Document]](https://drive.google.com/file/d/{file_id}/view)\n"
            
            answer += "\n💡I'm currently on Beta and based my answer on available memo documents, FAQs, and links trained as of June 2025. When in doubt, check [https://linktr.ee/fordevconchapterleads](https://linktr.ee/fordevconchapterleads) or  contact HQ hello@devcon.ph for confirmation!*"
            
            print(f"🔍 Reference display decision: {show_references}")
            return answer
            
        except Exception as e:
            print(f"❌ Error in history-aware verified query: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ I encountered an error while processing your question. Please try rephrasing or contact support."

    # 3. Optional: Add a method to explicitly control reference display
    def query_with_reference_control(self, question: str, history: List[Dict[str, str]] = None, force_show_references: bool = None) -> str:
        """Query method with explicit reference display control"""
        try:
            # If force_show_references is specified, we need to control the reference display
            if force_show_references is not None:
                # Get the synthesized response directly to control reference display
                query_response, error, final_query = self._create_history_aware_anti_hallucination_engine(question, history)
                if error:
                    return f"❌ Error creating query engine: {error}"

                # Extract the response text without any existing source sections
                response_text = query_response.response
                
                # Remove any existing reference markers
                response_text = response_text.replace("SHOW_REFERENCES: true", "").replace("SHOW_REFERENCES: false", "").strip()
                
                # Start building the final answer
                answer = f"{response_text}\n\n"
                
                # Add sources only if forced to show
                if force_show_references and hasattr(query_response, 'source_nodes') and query_response.source_nodes:
                    # Deduplicate sources by tracking unique file_ids
                    seen_sources = set()
                    unique_sources = []
                    
                    for node in query_response.source_nodes:
                        source_info = node.metadata.get('title', node.metadata.get('source', 'Unknown'))
                        file_id = node.metadata.get('file_id')
                        
                        # Create a unique identifier for this source
                        source_key = f"{source_info}_{file_id}"
                        
                        if source_key not in seen_sources:
                            seen_sources.add(source_key)
                            unique_sources.append(node)
                        
                        # Limit to 3 unique sources
                        if len(unique_sources) >= 3:
                            break
                    
                    if unique_sources:
                        answer += "📚 **Sources Referenced:**\n"
                        for i, node in enumerate(unique_sources, 1):
                            source_info = node.metadata.get('title', node.metadata.get('source', 'Unknown'))
                            file_id = node.metadata.get('file_id')
                            
                            answer += f"{i}. {source_info}\n"
                            
                            if file_id:
                                answer += f"   [[View Document]](https://drive.google.com/file/d/{file_id}/view)\n"
                
                # Add information about the response type
                answer += "\n💡I'm currently on Beta and based my answer on available memo documents, FAQs, and links trained as of June 2025. When in doubt, check [https://linktr.ee/fordevconchapterleads](https://linktr.ee/fordevconchapterleads) or  contact HQ hello@devcon.ph for confirmation!*"
                
                return answer
            else:
                # If no force_show_references specified, use the normal method
                return self.query_with_history_and_verification(question, history)
            
        except Exception as e:
            print(f"❌ Error in reference-controlled query: {e}")
            return f"❌ I encountered an error while processing your question. Please try rephrasing or contact support."

    # Update the main query method to use the merged functionality
    def query(self, question: str, history: List[Dict[str, str]] = None,
              top_k: int = None, temperature: float = None) -> str:
        """Main query method that handles both history and verification, with caching."""
        if not settings.ENABLE_QUERY_CACHE:
            return self.query_with_history_and_verification(question, history, top_k, temperature)

        key = self._query_cache_key(question, history, top_k, temperature)
        with self._query_cache_lock:
            cached = self._query_cache.get(key)
        if cached is not None:
            logger.info("⚡ Query cache hit")
            return cached

        answer = self.query_with_history_and_verification(question, history, top_k, temperature)

        # Only cache successful answers (don't cache error placeholders)
        if answer and not answer.lstrip().startswith("❌"):
            with self._query_cache_lock:
                self._query_cache[key] = answer
        return answer

    def _query_cache_key(self, question, history, top_k, temperature) -> str:
        """Build a stable cache key from the question + history + retrieval params."""
        hist = json.dumps(history or [], sort_keys=True, default=str)
        raw = f"{(question or '').strip().lower()}|{hist}|{top_k}|{temperature}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _invalidate_query_cache(self):
        """Clear the query-result cache (call after the corpus changes)."""
        try:
            cache = getattr(self, "_query_cache", None)
            if cache is not None:
                with self._query_cache_lock:
                    cache.clear()
                logger.info("🧹 Query cache cleared")
        except Exception as e:
            logger.warning(f"⚠️ Failed to clear query cache: {e}")

    def test_retrieval_scores(self, test_query: str = "DEVCON") -> dict:
        """Test retrieval scores to help tune similarity thresholds"""
        try:
            if not self.index:
                return {"error": "Index not initialized"}
            
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=10,
            )
            
            nodes = retriever.retrieve(test_query)
            
            if not nodes:
                return {"error": "No nodes retrieved"}
            
            scores = [node.score for node in nodes]
            
            return {
                "query": test_query,
                "num_results": len(nodes),
                "score_stats": {
                    "min": min(scores),
                    "max": max(scores),
                    "mean": sum(scores) / len(scores),
                    "all_scores": scores
                },
                "sample_results": [
                    {
                        "score": node.score,
                        "text_preview": node.text[:100] + "...",
                        "source": node.metadata.get('source', 'Unknown')
                    }
                    for node in nodes[:3]
                ]
            }
        except Exception as e:
            return {"error": str(e)}

    def persist_index(self):
        """Persist the index to storage"""
        try:
            if self.index is None:
                logger.warning("No index to persist")
                return
            
            # Ensure storage directory exists
            storage_dir = Path(settings.RAG_INDEX_DIR)
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Persist the index
            self.index.storage_context.persist(persist_dir=str(storage_dir))
            logger.info(f"Index persisted to {storage_dir}")
        except Exception as e:
            logger.error(f"Error persisting index: {e}")
        
    def get_index_stats(self) -> dict:
        """Get statistics about the current index"""
        try:
            if not self.index:
                return {"status": "not_initialized", "document_count": 0}
            
            # Get document count from vector store
            collection_count = len(self.vector_store._collection.get()['ids'])
            
            return {
                "status": "ready",
                "document_count": collection_count,
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "top_k_retrieval": settings.TOP_K_RETRIEVAL,
                "embedding_model": settings.OPENAI_EMBED_MODEL,
                "embedding_api": "OpenAI",
                "embedding_dimensions": settings.EMBEDDING_DIM,
                "llm_model": settings.OPENAI_LLM_MODEL,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _ensure_directories_exist(self):
        """Ensure all required directories exist"""
        try:
            # Ensure ChromaDB directory exists
            Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
            
            # Ensure RAG index directory exists
            Path(settings.RAG_INDEX_DIR).mkdir(parents=True, exist_ok=True)
            
            # Ensure index state file directory exists
            self.index_state_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.debug("All required directories ensured to exist")
        except Exception as e:
            logger.error(f"Error ensuring directories exist: {e}") 

    def start_new_index_build(self) -> bool:
        """Start building a new index without affecting the current one"""
        try:
            # Create a new ChromaDB collection for the new index
            chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            # Use a temporary collection name
            temp_collection_name = f"{settings.CHROMA_COLLECTION_NAME}_temp"
            
            # Delete existing temp collection if it exists
            try:
                chroma_client.delete_collection(temp_collection_name)
                logger.info(f"Deleted existing temp collection: {temp_collection_name}")
            except Exception:
                pass  # Collection doesn't exist, which is fine
            
            # Create new temporary collection with NO embedding function
            # This is important - we'll provide pre-computed embeddings
            self.temp_chroma_collection = chroma_client.create_collection(
                temp_collection_name,
                embedding_function=None
            )
            self.temp_vector_store = ChromaVectorStore(chroma_collection=self.temp_chroma_collection)
            
            # Create new temporary index
            temp_storage_context = StorageContext.from_defaults(vector_store=self.temp_vector_store)
            self.temp_index = VectorStoreIndex([], storage_context=temp_storage_context)
            
            # Initialize temporary index state
            self.temp_index_state = {}
            
            logger.info("✅ Started new index build with temporary collection")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting new index build: {e}")
            import traceback
            traceback.print_exc()
            return False

    def add_documents_to_new_index(self, documents: List[Document]) -> bool:
        """Add documents to the new index being built - COMPLETE RE-EMBEDDING"""
        try:
            if not hasattr(self, 'temp_index') or not self.temp_index:
                logger.error("❌ Temporary index not initialized. Call start_new_index_build() first.")
                return False
            
            logger.info(f"🔄 Starting complete re-embedding of {len(documents)} documents...")
            
            # Sanitize metadata for all documents
            logger.info("🧹 Sanitizing document metadata for ChromaDB compatibility...")
            for doc in documents:
                if hasattr(doc, 'metadata') and doc.metadata:
                    doc.metadata = self._sanitize_metadata(doc.metadata)
            
            # Use sentence splitter for better chunking
            parser = SentenceSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            
            # Parse documents into nodes
            nodes = parser.get_nodes_from_documents(documents)
            logger.info(f"📄 Created {len(nodes)} nodes from {len(documents)} documents")
            
            # Sanitize node metadata as well
            for node in nodes:
                if hasattr(node, 'metadata') and node.metadata:
                    node.metadata = self._sanitize_metadata(node.metadata)
            
            # Insert nodes - LlamaIndex auto-embeds them in batches via the
            # configured OpenAI embedding model (with built-in batching + retry).
            logger.info("🔄 Embedding nodes and adding them to the temporary index (this may take a while)...")
            self.temp_index.insert_nodes(nodes)
            
            # Update temporary index state
            for doc in documents:
                file_id = doc.metadata.get("file_id")
                if file_id:
                    self.temp_index_state[file_id] = {
                        "file_id": file_id,
                        "file_name": doc.metadata.get("source", "Unknown"),
                        "title": doc.metadata.get("title", doc.metadata.get("source", "Unknown")),
                        "modified_time": doc.metadata.get("modified_time"),
                        "chunk_count": len([n for n in nodes if n.metadata.get("file_id") == file_id])
                    }
            
            logger.info(f"✅ Added {len(documents)} documents with fresh embeddings to temporary index")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding documents to new index: {e}")
            import traceback
            traceback.print_exc()
            return False

    def rebuild_index_from_scratch(self, documents: List[Document]) -> bool:
        """Simple complete rebuild: Clear everything first, then rebuild from scratch"""
        try:
            logger.info("🚀 Starting COMPLETE INDEX REBUILD from scratch...")
            logger.info("🗑️ This will clear all existing data and rebuild everything fresh")
            
            # Step 1: Clear everything first
            logger.info("🧹 Clearing all existing index data...")
            if not self.clear_index():
                logger.error("❌ Failed to clear existing index")
                return False
            
            logger.info("✅ All existing data cleared")
            
            # Step 2: Sanitize metadata for all documents
            logger.info("🧹 Sanitizing document metadata for ChromaDB compatibility...")
            for doc in documents:
                if hasattr(doc, 'metadata') and doc.metadata:
                    doc.metadata = self._sanitize_metadata(doc.metadata)
            
            # Step 3: Use sentence splitter for better chunking
            logger.info("📄 Splitting documents into chunks...")
            parser = SentenceSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            
            # Parse documents into nodes
            nodes = parser.get_nodes_from_documents(documents)
            logger.info(f"📄 Created {len(nodes)} nodes from {len(documents)} documents")
            
            # Sanitize node metadata as well
            for node in nodes:
                if hasattr(node, 'metadata') and node.metadata:
                    node.metadata = self._sanitize_metadata(node.metadata)
            
            # Step 4: Insert nodes - LlamaIndex auto-embeds them in batches via the
            # configured OpenAI embedding model (with built-in batching + retry).
            logger.info("🔄 Embedding nodes and adding them to the index (this may take a while)...")
            self.index.insert_nodes(nodes)
            
            # Step 6: Update index state
            logger.info("💾 Updating index state...")
            self.index_state = {}
            for doc in documents:
                file_id = doc.metadata.get("file_id")
                if file_id:
                    self.index_state[file_id] = {
                        "file_id": file_id,
                        "file_name": doc.metadata.get("source", "Unknown"),
                        "title": doc.metadata.get("title", doc.metadata.get("source", "Unknown")),
                        "modified_time": doc.metadata.get("modified_time"),
                        "chunk_count": len([n for n in nodes if n.metadata.get("file_id") == file_id])
                    }
            
            # Save index state
            self.save_index_state()
            
            # Step 7: Setup query engine
            self._setup_query_engine()
            
            # Step 8: Persist index
            self.persist_index()

            # Invalidate any cached query results (the corpus changed)
            self._invalidate_query_cache()

            logger.info("✅ COMPLETE INDEX REBUILD successful!")
            logger.info(f"🎉 Rebuilt index with {len(documents)} documents and {len(nodes)} chunks")
            logger.info("🗑️ All old embeddings and .bin files have been completely replaced")
            return True

        except Exception as e:
            logger.error(f"❌ Error in complete index rebuild: {e}")
            import traceback
            traceback.print_exc()
            return False

    def incremental_update(self, drive_files: List[Dict[str, Any]]) -> dict:
        """Incrementally sync the index with Google Drive.

        Only re-embeds files that are new or whose ``modifiedTime`` changed since
        the last run, removes files that disappeared from Drive, and skips
        everything unchanged. Falls back to far less embedding cost than a full
        rebuild. Returns a summary dict.
        """
        from app.services.document_processor import document_processor

        summary = {"added": 0, "changed": 0, "skipped": 0, "deleted": 0, "failed": 0,
                   "added_files": [], "changed_files": [], "deleted_files": []}
        try:
            if self.index is None:
                self._load_or_create_index()

            current_ids = set()
            to_process = []  # files that are new or changed

            for f in drive_files:
                file_id = f.get("id")
                if not file_id:
                    continue
                current_ids.add(file_id)
                modified_time = f.get("modifiedTime", "Unknown")
                state = self.index_state.get(file_id)

                if state is None:
                    summary["added"] += 1
                    summary["added_files"].append(f.get("name", file_id))
                    to_process.append(f)
                elif str(state.get("modified_time")) != str(modified_time):
                    summary["changed"] += 1
                    summary["changed_files"].append(f.get("name", file_id))
                    to_process.append(f)
                else:
                    summary["skipped"] += 1

            # Files present in state but no longer in Drive -> delete
            deleted_ids = [fid for fid in list(self.index_state.keys()) if fid not in current_ids]
            for fid in deleted_ids:
                name = self.index_state.get(fid, {}).get("file_name", fid)
                if self.remove_document(fid):
                    summary["deleted"] += 1
                    summary["deleted_files"].append(name)

            # Changed files: remove the old version before re-inserting the new one
            changed_ids = {f.get("id") for f in to_process if self.index_state.get(f.get("id"))}
            for fid in changed_ids:
                self.remove_document(fid)

            logger.info(
                f"🔁 Incremental update: {summary['added']} added, {summary['changed']} changed, "
                f"{summary['skipped']} skipped, {summary['deleted']} deleted"
            )

            if to_process:
                documents = document_processor.process_drive_files(to_process)
                summary["failed"] = len(to_process) - len(documents)
                if documents:
                    # add_documents handles chunking, embedding, state update + persist
                    if not self.add_documents(documents):
                        logger.error("❌ Incremental update failed while adding documents")
                        summary["error"] = "add_documents failed"
            else:
                # Nothing new, but deletions may have changed the index -> persist + save
                if deleted_ids:
                    self.persist_index()
                self.save_index_state()

            self._invalidate_query_cache()
            return summary

        except Exception as e:
            logger.error(f"❌ Error during incremental update: {e}")
            import traceback
            traceback.print_exc()
            summary["error"] = str(e)
            return summary

    def test_embedding(self, text: str = "This is a test sentence.") -> dict:
        """Test the embedding model"""
        try:
            embed_model = LlamaSettings.embed_model
            embedding = embed_model._get_text_embedding(text)
            
            return {
                "status": "success",
                "text": text,
                "embedding_dim": len(embedding),
                "embedding_sample": embedding[:5],  # First 5 dimensions
                "model_name": settings.OPENAI_EMBED_MODEL,
                "api_type": "OpenAI",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
        

    def clear_index(self) -> bool:
        """Completely clear the RAG index and reset its state"""
        try:
            # Clear the ChromaDB collection
            chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            collection_name = settings.CHROMA_COLLECTION_NAME
            try:
                chroma_client.delete_collection(collection_name)
                logger.info(f"✅ Deleted ChromaDB collection: {collection_name}")
            except Exception as e:
                logger.info(f"⚠️ Could not delete collection: {e}. Creating new one anyway.")
            
            # Clear the ChromaDB persistent storage directory completely
            chroma_persist_dir = Path(settings.CHROMA_PERSIST_DIR)
            if chroma_persist_dir.exists():
                import shutil
                try:
                    shutil.rmtree(chroma_persist_dir)
                    logger.info(f"✅ Deleted ChromaDB storage directory: {chroma_persist_dir}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete ChromaDB storage directory: {e}")
                    # Try to delete individual collection folders
                    for item in chroma_persist_dir.iterdir():
                        if item.is_dir():
                            try:
                                shutil.rmtree(item)
                                logger.info(f"✅ Deleted ChromaDB collection folder: {item.name}")
                            except Exception as folder_error:
                                logger.warning(f"⚠️ Could not delete folder {item.name}: {folder_error}")
            
            # Ensure ChromaDB directory exists for recreation
            chroma_persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Clear the index storage directory (LlamaIndex storage)
            storage_dir = Path(settings.RAG_INDEX_DIR)
            if storage_dir.exists():
                import shutil
                try:
                    shutil.rmtree(storage_dir)
                    logger.info(f"✅ Deleted LlamaIndex storage at {storage_dir}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete LlamaIndex storage: {e}")
            
            # Ensure storage directory exists for recreation
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Reset index state file
            self.index_state = {}
            if self.index_state_file.exists():
                try:
                    self.index_state_file.unlink()
                    logger.info(f"✅ Deleted index state file: {self.index_state_file}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete index state file: {e}")
            
            # Reinitialize ChromaDB client and collection
            chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            chroma_collection = chroma_client.create_collection(collection_name, embedding_function=None)
            self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # Reinitialize the index components
            self.index = None
            self._load_or_create_index()
            
            # Save fresh index state (this will now create the directory if needed)
            self.save_index_state()

            # Drop any cached answers tied to the now-deleted corpus
            self._invalidate_query_cache()

            logger.info("✅ Index completely cleared and reset - all storage files deleted")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error clearing index: {e}")
            import traceback
            traceback.print_exc()
            return False

# Global instance
rag_service = LlamaIndexRAGService()