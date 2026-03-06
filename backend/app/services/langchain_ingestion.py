"""
LangChain Document Ingestion - Modern document processing pipeline
Phase 2 Feature #7
"""

from typing import List, Dict, Any, Optional
import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class LangChainIngestion:
    """
    LangChain-based document ingestion pipeline.
    Provides modern text splitting and metadata enrichment.
    """
    
    def __init__(self):
        self._initialize_splitters()
    
    def _initialize_splitters(self):
        """Initialize text splitters for different content types"""
        
        # General purpose splitter
        self.general_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Medical paper splitter (larger chunks)
        self.paper_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n## ", "\n\n", "\n", ". ", " ", ""]
        )
        
        # Abstract splitter (smaller chunks)
        self.abstract_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n", ". ", " ", ""]
        )
        
        logger.info("Initialized LangChain text splitters")
    
    def create_document(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> Document:
        """
        Create a LangChain Document
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            LangChain Document
        """
        return Document(
            page_content=text,
            metadata=metadata
        )
    
    def split_document(
        self,
        document: Document,
        content_type: str = 'general'
    ) -> List[Document]:
        """
        Split a document into chunks
        
        Args:
            document: LangChain Document
            content_type: Type of content ('general', 'paper', 'abstract')
            
        Returns:
            List of chunked Documents
        """
        # Select appropriate splitter
        if content_type == 'paper':
            splitter = self.paper_splitter
        elif content_type == 'abstract':
            splitter = self.abstract_splitter
        else:
            splitter = self.general_splitter
        
        # Split document
        chunks = splitter.split_documents([document])
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(chunks)
        
        logger.info(f"Split document into {len(chunks)} chunks")
        return chunks
    
    def enrich_metadata(
        self,
        document: Document,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """
        Enrich document metadata
        
        Args:
            document: LangChain Document
            additional_metadata: Additional metadata to add
            
        Returns:
            Document with enriched metadata
        """
        if additional_metadata:
            document.metadata.update(additional_metadata)
        
        # Add computed metadata
        document.metadata['text_length'] = len(document.page_content)
        document.metadata['word_count'] = len(document.page_content.split())
        
        return document
    
    def process_pubmed_article(
        self,
        article_data: Dict[str, Any]
    ) -> List[Document]:
        """
        Process a PubMed article into LangChain Documents
        
        Args:
            article_data: PubMed article data
            
        Returns:
            List of processed Documents
        """
        # Extract text
        title = article_data.get('title', '')
        abstract = article_data.get('abstract', '')
        text = f"{title}\n\n{abstract}"
        
        # Build metadata
        metadata = {
            'source_type': 'pubmed',
            'source_id': f"PMID:{article_data.get('pmid', '')}",
            'pmid': article_data.get('pmid', ''),
            'title': title,
            'url': f"https://pubmed.ncbi.nlm.nih.gov/{article_data.get('pmid', '')}/",
            'publication_date': article_data.get('publication_date', ''),
            'journal': article_data.get('journal', ''),
            'authors': article_data.get('authors', []),
            'doi': article_data.get('doi', ''),
        }
        
        # Create document
        doc = self.create_document(text, metadata)
        
        # Split into chunks
        chunks = self.split_document(doc, content_type='abstract')
        
        return chunks
    
    def process_clinical_trial(
        self,
        trial_data: Dict[str, Any]
    ) -> List[Document]:
        """
        Process a clinical trial into LangChain Documents
        
        Args:
            trial_data: Clinical trial data
            
        Returns:
            List of processed Documents
        """
        # Extract text
        title = trial_data.get('title', '')
        description = trial_data.get('description', '')
        text = f"{title}\n\n{description}"
        
        # Build metadata
        metadata = {
            'source_type': 'clinical_trial',
            'source_id': trial_data.get('nct_id', ''),
            'nct_id': trial_data.get('nct_id', ''),
            'title': title,
            'url': f"https://clinicaltrials.gov/study/{trial_data.get('nct_id', '')}",
            'phase': trial_data.get('phase', ''),
            'status': trial_data.get('status', ''),
            'sponsor': trial_data.get('sponsor', ''),
            'conditions': trial_data.get('conditions', []),
        }
        
        # Create document
        doc = self.create_document(text, metadata)
        
        # Split into chunks
        chunks = self.split_document(doc, content_type='general')
        
        return chunks
    
    def batch_process_documents(
        self,
        documents: List[Dict[str, Any]],
        source_type: str = 'general'
    ) -> List[Document]:
        """
        Process multiple documents in batch
        
        Args:
            documents: List of document data dicts
            source_type: Type of source ('pubmed', 'clinical_trial', 'general')
            
        Returns:
            List of processed LangChain Documents
        """
        all_chunks = []
        
        for doc_data in documents:
            try:
                if source_type == 'pubmed':
                    chunks = self.process_pubmed_article(doc_data)
                elif source_type == 'clinical_trial':
                    chunks = self.process_clinical_trial(doc_data)
                else:
                    # Generic processing
                    text = doc_data.get('text', '')
                    metadata = doc_data.get('metadata', {})
                    doc = self.create_document(text, metadata)
                    chunks = self.split_document(doc)
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                continue
        
        logger.info(f"Batch processed {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks
    
    def prepare_for_vectorstore(
        self,
        documents: List[Document]
    ) -> List[Dict[str, Any]]:
        """
        Prepare documents for vector store ingestion
        
        Args:
            documents: List of LangChain Documents
            
        Returns:
            List of dicts ready for vector store
        """
        prepared = []
        
        for doc in documents:
            prepared.append({
                'text': doc.page_content,
                'metadata': doc.metadata
            })
        
        return prepared


def get_langchain_ingestion() -> LangChainIngestion:
    """Create a LangChain ingestion instance"""
    return LangChainIngestion()
