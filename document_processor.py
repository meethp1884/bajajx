import aiohttp
import asyncio
import tempfile
import os
import io
from typing import List, Dict, Any, Optional
import logging
from urllib.parse import urlparse
import hashlib
import gc  # For memory management
import psutil

# Document parsing libraries
import pdfplumber
import docx2txt
import email
from email import policy
from email.parser import BytesParser

# Text processing
import tiktoken
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    source_url: str
    filename: str
    page_number: Optional[int] = None
    chunk_index: int = 0

class DocumentProcessor:
    """Handles document ingestion, parsing, and chunking with memory optimization"""
    
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 200):  # Reduced for memory efficiency
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        
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
        
    async def process_document(self, blob_url: str) -> List[DocumentChunk]:
        """
        Download and process a document from blob URL with memory optimization
        
        Args:
            blob_url: URL to the document blob
            
        Returns:
            List of document chunks
        """
        initial_memory = self.log_memory_usage("doc_process_start")
        
        try:
            # Download document
            document_data = await self._download_document(blob_url)
            filename = self._extract_filename(blob_url)
            
            # Determine file type and extract text
            text = await self._extract_text(document_data, filename)
            
            if not text.strip():
                logger.warning(f"No text extracted from {filename}")
                return []
            
            # Create chunks with memory optimization
            chunks = self._create_chunks(text, blob_url, filename)
            
            # Memory cleanup
            del document_data
            gc.collect()
            
            final_memory = self.log_memory_usage("doc_process_end")
            memory_delta = final_memory - initial_memory
            logger.info(f"Successfully processed {filename}: {len(chunks)} chunks created (memory delta: {memory_delta:+.1f} MB)")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document {blob_url}: {str(e)}")
            raise
    
    async def _download_document(self, url: str) -> bytes:
        """Download document from URL with timeout and progress"""
        logger.info(f"⬇️ Starting download from: {url[:100]}...")
        
        timeout = aiohttp.ClientTimeout(total=60, connect=10)  # 60s total, 10s connect
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download document: HTTP {response.status}")
                    
                    # Get content length for progress
                    content_length = response.headers.get('Content-Length')
                    if content_length:
                        size_mb = int(content_length) / (1024 * 1024)
                        logger.info(f"📊 Document size: {size_mb:.1f} MB")
                        
                        # Memory check for large files
                        if size_mb > 50:
                            logger.warning(f"⚠️ Large document detected ({size_mb:.1f} MB), may impact memory usage")
                    
                    # Download with progress
                    data = await response.read()
                    logger.info(f"✅ Download completed: {len(data)} bytes")
                    return data
                    
            except asyncio.TimeoutError:
                logger.error(f"❌ Download timeout after 60 seconds")
                raise Exception("Document download timeout - file too large or network too slow")
            except Exception as e:
                logger.error(f"❌ Download failed: {str(e)}")
                raise
    
    def _extract_filename(self, url: str) -> str:
        """Extract filename from URL"""
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        if not filename:
            filename = f"document_{hashlib.md5(url.encode()).hexdigest()[:8]}"
        return filename
    
    async def _extract_text(self, document_data: bytes, filename: str) -> str:
        """Extract text based on file extension with memory optimization"""
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext == '.pdf':
            return await self._extract_text_from_pdf(document_data)
        elif file_ext in ['.docx', '.doc']:
            return await self._extract_docx_text(document_data)
        elif file_ext == '.eml':
            return await self._extract_eml_text(document_data)
        else:
            # Try to decode as plain text
            try:
                return document_data.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    return document_data.decode('latin-1')
                except:
                    raise Exception(f"Unsupported file type: {file_ext}")
    
    async def _extract_text_from_pdf(self, pdf_data: bytes) -> str:
        """Extract text from PDF with speed and memory optimizations"""
        try:
            # Use faster extraction with minimal processing
            with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
                text_parts = []
                # Process only first 30 pages for memory efficiency (reduced from 50)
                max_pages = min(30, len(pdf.pages))
                logger.info(f"📄 Processing {max_pages} pages for memory optimization")
                
                for i, page in enumerate(pdf.pages[:max_pages]):
                    if i % 10 == 0:  # Log progress every 10 pages
                        logger.info(f"📄 Processing page {i+1}/{max_pages}")
                    
                    # Extract text with minimal processing
                    page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if page_text:
                        # Quick cleanup - remove excessive whitespace only
                        cleaned_text = ' '.join(page_text.split())
                        text_parts.append(cleaned_text)
                    
                    # Memory cleanup every 5 pages
                    if i % 5 == 0:
                        gc.collect()
                
                full_text = '\n\n'.join(text_parts)
                logger.info(f"✅ PDF text extracted: {len(full_text)} characters from {max_pages} pages")
                return full_text
                
        except Exception as e:
            logger.error(f"❌ PDF extraction error: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    async def _extract_docx_text(self, docx_data: bytes) -> str:
        """Extract text from DOCX using docx2txt with memory cleanup"""
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp_file:
            tmp_file.write(docx_data)
            tmp_file.flush()
            tmp_path = tmp_file.name
            
        try:
            text = docx2txt.process(tmp_path)
            return text or ""
        finally:
            # Ensure file is properly closed before deletion
            try:
                os.unlink(tmp_path)
            except (OSError, PermissionError) as e:
                logger.warning(f"Could not delete temporary file {tmp_path}: {e}")
        
        return text or ""
    
    async def _extract_eml_text(self, eml_data: bytes) -> str:
        """Extract text from EML email files"""
        msg = BytesParser(policy=policy.default).parsebytes(eml_data)
        
        text_parts = []
        
        # Extract headers
        subject = msg.get('Subject', 'No Subject')
        from_addr = msg.get('From', 'Unknown Sender')
        to_addr = msg.get('To', 'Unknown Recipient')
        date = msg.get('Date', 'Unknown Date')
        
        text_parts.append(f"Subject: {subject}")
        text_parts.append(f"From: {from_addr}")
        text_parts.append(f"To: {to_addr}")
        text_parts.append(f"Date: {date}")
        text_parts.append("-" * 50)
        
        # Extract body
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_content()
                    if body:
                        text_parts.append(body)
        else:
            if msg.get_content_type() == "text/plain":
                body = msg.get_content()
                if body:
                    text_parts.append(body)
        
        return "\n".join(text_parts)
    
    def _create_chunks(self, text: str, source_url: str, filename: str) -> List[DocumentChunk]:
        """Split text into overlapping chunks with memory optimization"""
        logger.info(f"🔪 Starting text chunking: {len(text)} characters")
        
        if not text.strip():
            logger.warning("⚠️ Empty text provided for chunking")
            return []
        
        try:
            # Initialize tokenizer
            logger.info(f"📝 Initializing tokenizer...")
            encoding = tiktoken.get_encoding("cl100k_base")
            
            # Tokenize the full text
            logger.info(f"🔢 Tokenizing text...")
            tokens = encoding.encode(text)
            logger.info(f"✅ Text tokenized: {len(tokens)} tokens")
            
            chunks = []
            start = 0
            chunk_count = 0
            
            logger.info(f"🔄 Starting chunking process...")
            while start < len(tokens):
                chunk_count += 1
                if chunk_count % 50 == 0:  # Reduce logging frequency
                    logger.info(f"📦 Processing chunk {chunk_count}...")
                
                # Emergency brake - prevent runaway chunking (reduced limit for memory)
                if chunk_count > 500:  # Reduced from 1000
                    logger.error(f"🚨 EMERGENCY STOP: Too many chunks ({chunk_count}), breaking loop")
                    break
                
                # Calculate initial end position
                initial_end = min(start + self.chunk_size, len(tokens))
                
                # Prevent infinite loop - ensure we make progress
                if initial_end <= start:
                    logger.error(f"🚨 Invalid chunk boundaries: start={start}, end={initial_end}")
                    break
                
                # Smart boundary detection for insurance documents
                end = self._find_smart_boundary(tokens, start, initial_end, encoding)
                
                # Extract chunk tokens
                chunk_tokens = tokens[start:end]
                
                # Decode back to text
                chunk_text = encoding.decode(chunk_tokens)
                
                # Create chunk ID
                chunk_id = hashlib.md5(f"{source_url}_{chunk_count}".encode()).hexdigest()
                
                # Create chunk object
                chunk = DocumentChunk(
                    text=chunk_text,
                    metadata={
                        "token_count": len(chunk_tokens),
                        "start_token": start,
                        "end_token": end,
                        "total_chunks": None  # Will be set after all chunks are created
                    },
                    chunk_id=chunk_id,
                    source_url=source_url,
                    filename=filename,
                    chunk_index=chunk_count
                )
                
                chunks.append(chunk)
                
                # Move start position (with overlap) - FIXED LOGIC FOR LARGE OVERLAPS
                if self.chunk_overlap >= self.chunk_size:
                    # If overlap >= chunk_size, use smaller step to prevent infinite loops
                    step_size = max(self.chunk_size // 4, 50)  # Move by at least 50 tokens
                    start = start + step_size
                else:
                    # Normal case: move by (chunk_size - overlap)
                    step_size = self.chunk_size - self.chunk_overlap
                    start = start + step_size
                
                # Ensure we make progress
                if step_size <= 0:
                    start = end
                
                # Final safety check
                if start >= len(tokens):
                    break
                
                # Memory cleanup every 100 chunks
                if chunk_count % 100 == 0:
                    gc.collect()
            
            logger.info(f"✅ Text chunking completed: {len(chunks)} chunks created")
            
            # Update total chunks count
            for chunk in chunks:
                chunk.metadata["total_chunks"] = len(chunks)
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error during text chunking: {str(e)}")
            return []
    
    def _find_smart_boundary(self, tokens, start, initial_end, encoding):
        """
        Find smart chunk boundary to avoid splitting sentences/clauses mid-way.
        Optimized for insurance documents with clauses, numbered sections, and TABLES.
        """
        try:
            # If we're at the end of the document, return as-is
            if initial_end >= len(tokens):
                return initial_end
            
            # Look back up to 100 tokens for a good boundary (reduced from 150 for memory)
            search_window = min(100, initial_end - start)
            best_boundary = initial_end
            
            # Decode a larger segment to analyze for table structures
            segment_start = max(0, start)
            segment_end = min(len(tokens), initial_end + 30)  # Reduced from 50
            segment_tokens = tokens[segment_start:segment_end]
            segment_text = encoding.decode(segment_tokens)
            
            # TABLE-AWARE: Check if we're in the middle of a table
            table_indicators = ['|', '─', '┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼']
            pipe_count = segment_text.count('|')
            has_table_chars = any(char in segment_text for char in table_indicators)
            
            # If we detect table structure, try to find table boundaries
            if pipe_count >= 3 or has_table_chars:
                logger.debug(f"Table detected in chunk boundary area (pipes: {pipe_count})")
                table_boundary = self._find_table_boundary(tokens, start, initial_end, encoding)
                if table_boundary != initial_end:
                    logger.debug(f"Found table boundary at token {table_boundary}")
                    return table_boundary
            
            # Insurance document boundary markers (in order of preference)
            boundary_markers = [
                '\n\n',  # Paragraph break (highest priority for tables)
                '. ',  # Sentence end with space
                '.\n',  # Sentence end with newline
                '; ',  # Clause separator with space
                ')\n',  # End of numbered item with newline
                ': ',  # After section headers with space
                '\n',  # Line break
            ]
            
            # Check tokens in reverse order from initial_end
            for i in range(initial_end, max(start + 30, initial_end - search_window), -1):  # Reduced minimum
                # Decode a larger segment around this position to check for boundaries
                segment_start = max(0, i - 10)
                segment_end = min(len(tokens), i + 10)
                segment_tokens = tokens[segment_start:segment_end]
                segment_text = encoding.decode(segment_tokens)
                
                # Find the relative position within the segment
                relative_pos = i - segment_start
                
                # Check if any boundary marker appears at or near this position
                for priority, marker in enumerate(boundary_markers):
                    # Look for the marker in the text around our position
                    marker_positions = []
                    start_search = 0
                    while True:
                        pos = segment_text.find(marker, start_search)
                        if pos == -1:
                            break
                        marker_positions.append(pos)
                        start_search = pos + 1
                    
                    # Check if any marker position is close to our relative position
                    for marker_pos in marker_positions:
                        if abs(marker_pos - relative_pos) <= 3:  # Within 3 characters
                            # Found a good boundary, return position after the marker
                            boundary_pos = i + (marker_pos - relative_pos) + len(marker)
                            boundary_pos = min(boundary_pos, len(tokens))  # Don't exceed token limit
                            logger.debug(f"Found smart boundary at token {boundary_pos} with marker '{marker}' (priority {priority})")
                            return boundary_pos
            
            # If no good boundary found within search window, try a fallback
            # Look for any period or newline in a smaller window
            fallback_window = min(30, initial_end - start)  # Reduced from 50
            for i in range(initial_end, max(start + 20, initial_end - fallback_window), -1):
                segment_tokens = tokens[max(0, i-3):min(len(tokens), i+3)]
                segment_text = encoding.decode(segment_tokens)
                if '.' in segment_text or '\n' in segment_text:
                    logger.debug(f"Found fallback boundary at token {i}")
                    return i
            
            # Last resort: return initial position
            logger.debug(f"No smart boundary found, using initial position {initial_end}")
            return initial_end
            
        except Exception as e:
            logger.error(f"Error in smart boundary detection: {e}")
            return initial_end
    
    def _find_table_boundary(self, tokens, start, initial_end, encoding):
        """
        Find optimal boundary when tables are detected to avoid splitting tables.
        """
        try:
            # Look for table start/end patterns
            search_start = max(0, start - 30)  # Reduced from 50
            search_end = min(len(tokens), initial_end + 50)  # Reduced from 100
            search_tokens = tokens[search_start:search_end]
            search_text = encoding.decode(search_tokens)
            
            lines = search_text.split('\n')
            current_pos = 0
            table_start = -1
            table_end = -1
            
            # Find table boundaries by analyzing line patterns
            for i, line in enumerate(lines):
                line_start_pos = current_pos
                current_pos += len(line) + 1  # +1 for newline
                
                # Check if this line looks like a table row
                pipe_count = line.count('|')
                has_table_chars = any(char in line for char in ['─', '┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼'])
                
                if pipe_count >= 2 or has_table_chars:
                    if table_start == -1:
                        table_start = line_start_pos
                    table_end = current_pos
                elif table_start != -1 and line.strip() == '':
                    # Empty line after table might indicate table end
                    break
            
            # Convert text positions back to token positions
            if table_start != -1 and table_end != -1:
                # Try to position boundary before table start or after table end
                relative_initial = initial_end - search_start
                
                if relative_initial < table_start:
                    # We're before the table, try to end before it starts
                    boundary_tokens = encoding.encode(search_text[:table_start])
                    return search_start + len(boundary_tokens)
                elif relative_initial > table_end:
                    # We're after the table, try to start after it ends
                    boundary_tokens = encoding.encode(search_text[:table_end])
                    return search_start + len(boundary_tokens)
            
            return initial_end
            
        except Exception as e:
            logger.debug(f"Table boundary detection failed: {e}")
            return initial_end
