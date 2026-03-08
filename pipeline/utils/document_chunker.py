"""
Document Chunker - Chunks documents appropriately for embedding
"""
from typing import List, Dict
import re
import logging

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Chunks documents into appropriately-sized pieces for embedding
    Uses semantic chunking with overlap for context preservation
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        """
        Args:
            chunk_size: Target size in tokens (approximate)
            chunk_overlap: Overlap between chunks in tokens
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Chunk a document into smaller pieces
        
        Args:
            document: Document dict with text and metadata
            
        Returns:
            List of chunk dicts with text and metadata
        """
        # Get text to chunk
        text = self._get_text_to_chunk(document)
        
        if not text:
            logger.warning(f"Document {document.get('id')} has no text to chunk. Source type: {document.get('source_type')}")
            return []
        
        if len(text) < self.min_chunk_size:
            logger.warning(f"Document {document.get('id')} too short to chunk (length: {len(text)}, min: {self.min_chunk_size})")
            return []
        
        # Split into semantic chunks
        chunks = self._semantic_split(text)
        
        if not chunks:
            logger.warning(f"Document {document.get('id')} produced no chunks after splitting")
            return []
        
        # Create chunk documents
        chunk_docs = []
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text) < self.min_chunk_size:
                logger.debug(f"Skipping chunk {i} from {document.get('id')} - too short ({len(chunk_text)} < {self.min_chunk_size})")
                continue
            
            chunk_doc = {
                "id": f"{document.get('id', 'unknown')}_chunk_{i}",
                "text": chunk_text,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "metadata": self._build_chunk_metadata(document, i, len(chunks))
            }
            chunk_docs.append(chunk_doc)
        
        if chunk_docs:
            logger.info(f"Chunked document {document.get('id')} into {len(chunk_docs)} chunks")
        else:
            logger.warning(f"Document {document.get('id')} produced 0 valid chunks (all too short)")
        
        return chunk_docs
    
    def _get_text_to_chunk(self, document: Dict) -> str:
        """Extract text from document"""
        source_type = document.get("source_type", "")
        
        if source_type == "pubmed":
            # For PubMed: title + abstract
            title = document.get("title", "")
            abstract = document.get("abstract", "")
            text = f"{title}\n\n{abstract}"
            logger.debug(f"PubMed doc {document.get('id')}: title={len(title)} chars, abstract={len(abstract)} chars, total={len(text)} chars")
            return text
        
        elif source_type == "clinical_trial":
            # For clinical trials: title + summary
            title = document.get("title", "")
            summary = document.get("summary", "")
            text = f"{title}\n\n{summary}"
            logger.debug(f"Clinical trial {document.get('id')}: title={len(title)} chars, summary={len(summary)} chars, total={len(text)} chars")
            return text
        
        elif source_type in ["biorxiv", "medrxiv"]:
            # For preprints: title + abstract
            title = document.get("title", "")
            abstract = document.get("abstract", "")
            text = f"{title}\n\n{abstract}"
            logger.debug(f"{source_type} doc {document.get('id')}: title={len(title)} chars, abstract={len(abstract)} chars, total={len(text)} chars")
            return text
        
        else:
            # Generic: try to find text field
            logger.warning(f"Unknown source_type '{source_type}' for document {document.get('id')}")
            return document.get("text", "") or document.get("content", "")
    
    def _semantic_split(self, text: str) -> List[str]:
        """
        Split text into semantic chunks
        Tries to split on paragraph boundaries, then sentences
        """
        # First, try splitting by paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Estimate tokens (rough: 1 token ≈ 4 chars)
            para_tokens = len(para) // 4
            current_tokens = len(current_chunk) // 4
            
            if current_tokens + para_tokens <= self.chunk_size:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # Current chunk is full
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Start new chunk
                if para_tokens > self.chunk_size:
                    # Paragraph too long, split by sentences
                    sentence_chunks = self._split_by_sentences(para)
                    chunks.extend(sentence_chunks[:-1])
                    current_chunk = sentence_chunks[-1] if sentence_chunks else ""
                else:
                    current_chunk = para
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Add overlap
        chunks = self._add_overlap(chunks)
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_tokens = len(sentence) // 4
            current_tokens = len(current_chunk) // 4
            
            if current_tokens + sentence_tokens <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between chunks for context"""
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
                continue
            
            # Get last N tokens from previous chunk
            prev_chunk = chunks[i-1]
            prev_words = prev_chunk.split()
            overlap_words = prev_words[-self.chunk_overlap:] if len(prev_words) > self.chunk_overlap else prev_words
            overlap_text = " ".join(overlap_words)
            
            # Prepend overlap to current chunk
            overlapped_chunk = f"{overlap_text} {chunk}"
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def _build_chunk_metadata(self, document: Dict, chunk_index: int, total_chunks: int) -> Dict:
        """Build metadata for a chunk"""
        metadata = {
            "source_id": document.get("id", ""),
            "source_type": document.get("source_type", ""),
            "title": document.get("title", ""),
            "url": document.get("url", ""),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "industry_sponsored": document.get("industry_sponsored", False),
        }
        
        # Add source-specific metadata
        source_type = document.get("source_type", "")
        
        if source_type == "pubmed":
            metadata.update({
                "pmid": document.get("pmid", ""),
                "authors": document.get("authors", []),
                "publication_date": document.get("publication_date", ""),
                "journal": document.get("journal", ""),
                "doi": document.get("doi", ""),
            })
        
        elif source_type == "clinical_trial":
            metadata.update({
                "nct_id": document.get("nct_id", ""),
                "phase": document.get("phase", ""),
                "status": document.get("status", ""),
                "sponsor": document.get("sponsor", ""),
                "conditions": document.get("conditions", []),
            })
        
        elif source_type in ["biorxiv", "medrxiv"]:
            metadata.update({
                "doi": document.get("doi", ""),
                "authors": document.get("authors", []),
                "publication_date": document.get("publication_date", ""),
                "category": document.get("category", ""),
            })
        
        return metadata


# Example usage
if __name__ == "__main__":
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
    
    # Example document
    doc = {
        "id": "PMID:12345678",
        "source_type": "pubmed",
        "title": "Study on Diabetes Treatment",
        "abstract": "This is a long abstract that would be chunked..." * 50,
        "authors": ["John Doe", "Jane Smith"],
        "publication_date": "2023-01-15",
        "journal": "Medical Journal",
        "doi": "10.1234/example",
        "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
        "industry_sponsored": False,
    }
    
    chunks = chunker.chunk_document(doc)
    
    for chunk in chunks:
        print(f"Chunk {chunk['chunk_index'] + 1}/{chunk['total_chunks']}")
        print(f"Text length: {len(chunk['text'])}")
        print(f"Text preview: {chunk['text'][:100]}...")
        print("-" * 80)
