import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import json
import time
import concurrent.futures
from functools import lru_cache
import gc  # For memory management
import psutil

# Vector search
import faiss
from sentence_transformers import SentenceTransformer

# LLM integration - Updated for Groq
import aiohttp
import json as json_lib

# Local imports
from document_processor import DocumentChunk

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading
_embedding_model_cache = {}

@lru_cache(maxsize=1)
def get_embedding_model(model_name: str):
    """Cache embedding model globally for speed"""
    if model_name not in _embedding_model_cache:
        logger.info(f"📥 Loading embedding model: {model_name}")
        start_time = time.time()
        _embedding_model_cache[model_name] = SentenceTransformer(model_name)
        load_time = time.time() - start_time
        logger.info(f"✅ Embedding model cached in {load_time:.1f}s")
    return _embedding_model_cache[model_name]

@dataclass
class SearchResult:
    """Represents a search result with relevance score"""
    chunk: DocumentChunk
    score: float
    rank: int

class QueryEngine:
    """Handles embedding generation, vector search, and answer generation with memory optimization"""
    
    def __init__(self, 
                 embedding_model: str = None,
                 llm_model: str = None,
                 top_k: int = 2,
                 similarity_threshold: float = 0.3):
        """
        Initialize the query engine
        
        Args:
            embedding_model: Sentence transformer model for embeddings
            llm_model: Groq model for answer generation
            top_k: Number of top chunks to retrieve
            similarity_threshold: Minimum similarity score for relevance
        """
        # Load from environment variables if not provided
        self.embedding_model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "sonar-pro")
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        logger.info(f"🤖 Initializing QueryEngine with Perplexity model: {self.llm_model}")
        
        # Initialize embedding model with caching
        logger.info(f"📥 Loading embedding model: {self.embedding_model_name}")
        start_time = time.time()
        self.embedding_model = get_embedding_model(self.embedding_model_name)
        load_time = time.time() - start_time
        logger.info(f"✅ Embedding model loaded in {load_time:.1f}s")
        
        # Initialize Perplexity API configuration
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        self.perplexity_base_url = "https://api.perplexity.ai/chat/completions"
        
        if not self.perplexity_api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable is required")
        
        # Vector storage
        self.index = None
        self.chunks = []
        self.embeddings = None
        
    def log_memory_usage(self, stage: str):
        """Log current memory usage"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"💾 Memory usage at {stage}: {memory_mb:.1f} MB")
            return memory_mb
        except Exception as e:
            logger.warning(f"Could not log memory usage: {e}")
            return 0
        
    async def build_index(self, chunks: List[DocumentChunk]):
        """
        Build FAISS index from document chunks with memory optimization
        
        Args:
            chunks: List of document chunks to index
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return
            
        initial_memory = self.log_memory_usage("index_build_start")
        logger.info(f"Building index for {len(chunks)} chunks")
        
        # Store chunks
        self.chunks = chunks
        
        # Generate embeddings in parallel with memory management
        texts = [chunk.text for chunk in chunks]
        
        # Process embeddings in smaller batches for memory efficiency
        batch_size = 50  # Process 50 chunks at a time
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.info(f"📊 Processing embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                batch_embeddings = list(executor.map(self.embedding_model.encode, batch_texts))
            
            all_embeddings.extend(batch_embeddings)
            
            # Memory cleanup after each batch
            gc.collect()
            self.log_memory_usage(f"after_batch_{i//batch_size + 1}")
        
        self.embeddings = np.array(all_embeddings)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        # Memory cleanup after indexing
        gc.collect()
        final_memory = self.log_memory_usage("index_build_end")
        memory_delta = final_memory - initial_memory
        
        logger.info(f"Index built successfully with {self.index.ntotal} vectors (memory delta: {memory_delta:+.1f} MB)")
    
    async def search_relevant_chunks(self, query: str) -> List[SearchResult]:
        """
        Search for relevant chunks using semantic similarity
        
        Args:
            query: User query string
            
        Returns:
            List of relevant chunks with scores
        """
        if self.index is None or len(self.chunks) == 0:
            logger.warning("Index not built or no chunks available")
            return []
        
        logger.info(f"🔍 Searching for query: '{query[:50]}...'")
        logger.info(f"📚 Total chunks available: {len(self.chunks)}")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search for similar chunks
        scores, indices = self.index.search(query_embedding, self.top_k)
        
        # Debug logging
        logger.info(f" Top {self.top_k} similarity scores: {[f'{s:.3f}' for s in scores[0]]}")
        logger.info(f"🎯 Similarity threshold: {self.similarity_threshold}")
        
        # Create search results - use all chunks above threshold for accuracy
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            logger.info(f"   Rank {rank+1}: Score {score:.3f} - Chunk preview: '{self.chunks[idx].text[:100]}...'")
            
            # Include chunks above similarity threshold for better accuracy
            if score >= self.similarity_threshold:
                result = SearchResult(
                    chunk=self.chunks[idx],
                    score=float(score),
                    rank=rank
                )
                results.append(result)
                logger.info(f"   ✅ Including chunk {rank+1} (score: {score:.3f})")
            else:
                logger.info(f"   ❌ Skipping chunk {rank+1} (below threshold: {score:.3f})")
        
        logger.info(f"🎯 Returning {len(results)} chunks to LLM")
        return results
    
    async def answer_question(self, question: str) -> str:
        """
        Generate an answer to a question using retrieved context
        
        Args:
            question: User question
            
        Returns:
            Generated answer with source attribution
        """
        # Search for relevant chunks
        search_results = await self.search_relevant_chunks(question)
        
        if not search_results:
            return "I couldn't find relevant information in the provided documents to answer your question."
        
        # Prepare enhanced context from retrieved chunks with metadata
        context_parts = []
        source_info = []
        encoding = self.embedding_model.tokenizer if hasattr(self.embedding_model, 'tokenizer') else None
        try:
            import tiktoken
            encoding = encoding or tiktoken.get_encoding("cl100k_base")
        except Exception:
            pass
        system_prompt = "CRITICAL: DO NOT SHOW YOUR THINKING PROCESS. GIVE ONLY THE FINAL ANSWER."
        question_prompt = f"\n\nQuery: {question}\n"
        instructions = """\nINSTRUCTIONS:\n- Extract ALL relevant information from the document\n- Include ALL numbers, amounts, timeframes, and percentages when present\n- Include ALL applicable conditions or scenarios\n- Make decisions based on policy rules\n- Give complete answer with all relevant details\n- EVERY ESSENTIAL DETAIL and question asked by user should NOT remain unanswered if present in the document

IMPORTANT: Include ALL relevant numbers and amounts:
- Waiting periods (e.g., "24 months", "36 months")
- Sum insured amounts (e.g., "INR 5,00,000")
- Percentages (e.g., "20% co-payment")
- Grace periods (e.g., "15 days")
- Age limits (e.g., "65 years")
- Multiple scenarios if applicable

SIMPLIFICATION GUIDELINES:
You are a helpful assistant. Please simplify and shorten your insurance-related answers. Your goal is to rewrite them in plain English for a common user with no technical background.

Guidelines:
1. Each answer should be **1–3 lines** (max 60–80 words when multiple details are essential).
2. Avoid legal or policy jargon unless necessary. Use everyday language.
3. Summarize key facts only. Don't repeat clauses, references, or legal acts.
4. If conditions or limits exist, mention them **briefly but completely**.
5. Use bullet points or numbers only if truly helpful; prefer clear sentences.
6. Do not include citations like [1], [2], etc.
7. **CRITICAL**: Every essential detail and question asked by user should NOT remain unanswered if present in the document.
8. **ACCURACY OVER BREVITY**: If essential details require more words, prioritize completeness over word count.

FORMAT: Answer: [complete factual response with all numbers in plain English]

EXAMPLES:
- "Answer: Grace period is 30 days after due date to pay premium without losing benefits."
- "Answer: Pre-existing diseases covered after 36 months of continuous coverage."
- "Answer: Maternity covered after 24 months, limited to 2 deliveries per policy period."
- "Answer: Room rent capped at 1% of sum insured, ICU at 2% (Plan A only). No limits for PPN treatments."

CRITICAL: Include ALL relevant numbers and amounts from the document. Answer ALL parts of the question. NO reasoning process. Plain English only. ACCURACY IS PRIORITY.

Answer:"""
        # Token limit for prompt (reduced for memory efficiency)
        MAX_PROMPT_TOKENS = 3000  # Reduced from 4000
        # Assemble prompt and trim context if needed
        tokens = encoding.encode(system_prompt) + encoding.encode(question_prompt) + encoding.encode(instructions)
        context_parts_sorted = []
        # Sort by relevance if not already
        sorted_results = sorted(zip(search_results, range(len(search_results))), key=lambda x: x[0].score, reverse=True)
        for result, idx in sorted_results:
            chunk = result.chunk
            context_header = f"[Document Section {idx+1} - Relevance: {result.score:.3f}]\n"
            chunk_text = chunk.text.strip()
            part = context_header + chunk_text
            part_tokens = encoding.encode(part)
            if len(tokens) + len(part_tokens) > MAX_PROMPT_TOKENS:
                break
            tokens += part_tokens
            context_parts_sorted.append(part)
        context = "\n\n".join(context_parts_sorted)
        # Generate answer using LLM
        answer = await self._generate_llm_answer(question, context, source_info)
        return answer
    
    async def _generate_llm_answer(self, question: str, context: str, source_info: List[Dict]) -> str:
        """
        Generate answer using Perplexity Pro LLM with optimized performance and memory management
        """
        try:
            prompt = f"""CRITICAL: DO NOT SHOW YOUR THINKING PROCESS. GIVE ONLY THE FINAL ANSWER.

        Document Context:
        {context}

        Query: {question}

        INSTRUCTIONS:
        - Extract ALL relevant information from the document
        - Include ALL numbers, amounts, timeframes, and percentages when present
        - Include ALL applicable conditions or scenarios
        - Make decisions based on policy rules
        - Give complete answer with all relevant details
        - EVERY ESSENTIAL DETAIL and question asked by user should NOT remain unanswered if present in the document

        IMPORTANT: Include ALL relevant numbers and amounts:
        - Waiting periods (e.g., "24 months", "36 months")
        - Sum insured amounts (e.g., "INR 5,00,000")
        - Percentages (e.g., "20% co-payment")
        - Grace periods (e.g., "15 days")
        - Age limits (e.g., "65 years")
        - Multiple scenarios if applicable

        SIMPLIFICATION GUIDELINES:
        You are a helpful assistant. Please simplify and shorten your insurance-related answers. Your goal is to rewrite them in plain English for a common user with no technical background.

        Guidelines:
        1. Each answer should be **1–3 lines** (max 60–80 words when multiple details are essential).
        2. Avoid legal or policy jargon unless necessary. Use everyday language.
        3. Summarize key facts only. Don't repeat clauses, references, or legal acts.
        4. If conditions or limits exist, mention them **briefly but completely**.
        5. Use bullet points or numbers only if truly helpful; prefer clear sentences.
        6. Do not include citations like [1], [2], etc.
        7. **CRITICAL**: Every essential detail and question asked by user should NOT remain unanswered if present in the document.
        8. **ACCURACY OVER BREVITY**: If essential details require more words, prioritize completeness over word count.

        FORMAT: Answer: [complete factual response with all numbers in plain English]

        EXAMPLES:
        - "Answer: Grace period is 30 days after due date to pay premium without losing benefits."
        - "Answer: Pre-existing diseases covered after 36 months of continuous coverage."
        - "Answer: Maternity covered after 24 months, limited to 2 deliveries per policy period."
        - "Answer: Room rent capped at 1% of sum insured, ICU at 2% (Plan A only). No limits for PPN treatments."

        CRITICAL: Include ALL relevant numbers and amounts from the document. Answer ALL parts of the question. NO reasoning process. Plain English only. ACCURACY IS PRIORITY.

        Answer:"""

            # Optimized API call for speed using Perplexity
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.perplexity_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.llm_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,  # Slightly higher for faster processing
                        "max_tokens": 80,    # Much smaller for speed
                        "stream": False,
                        "search_recency_filter": "month",
                        "return_citations": False,
                        "return_images": False
                    },
                    timeout=aiohttp.ClientTimeout(total=8)  # 8 second timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        answer = result["choices"][0]["message"]["content"].strip()
                        
                        # Clean up the answer
                        if answer.startswith("Answer:"):
                            answer = answer[7:].strip()
                        
                        return answer
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ LLM API error {response.status}: {error_text}")
                        return f"Error generating answer: API returned {response.status}"
                        
        except asyncio.TimeoutError:
            logger.error("❌ LLM request timeout")
            return "Error: Response generation timed out"
        except Exception as e:
            logger.error(f"❌ LLM generation error: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def _format_source_attribution(self, source_info: List[Dict]) -> str:
        """Format source attribution information"""
        if not source_info:
            return ""
        
        attribution_parts = ["**Sources:**"]
        
        # Group by filename
        sources_by_file = {}
        for info in source_info:
            filename = info["source"]
            if filename not in sources_by_file:
                sources_by_file[filename] = []
            sources_by_file[filename].append(info)
        
        for filename, infos in sources_by_file.items():
            relevance_scores = [info["relevance_score"] for info in infos]
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            attribution_parts.append(f"- {filename} (relevance: {avg_relevance:.2f})")
        
        return "\n".join(attribution_parts)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        if self.index is None:
            return {"status": "not_built"}
        
        return {
            "status": "ready",
            "total_chunks": len(self.chunks),
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "embedding_model": self.embedding_model_name,
            "llm_model": self.llm_model
        }
