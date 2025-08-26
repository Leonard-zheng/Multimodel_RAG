from langchain_chroma import Chroma
from langchain.retrievers import MultiVectorRetriever
from langchain_core.documents import Document
from langchain.storage import InMemoryStore
from .llm_manager import llm_manager
from .utils import handle_errors, logger
from .cache_manager import cache_manager
from pathlib import Path
import pickle

class DocumentManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.embeddings = llm_manager.get_embeddings()
        self.vector_store = Chroma(
            collection_name="multirag",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        self.docstore = InMemoryStore()
        # Persist original content to a pickle file; keep legacy JSON name as fallback
        self.docstore_file = "./docstore.pkl"
        
        # Load existing docstore data
        self._load_docstore()
        
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vector_store,
            docstore=self.docstore,
            id_key='doc_id'
        )
    
    def _load_docstore(self):
        """Load docstore data from file (supports legacy filename)."""
        path = self.docstore_file
        if not Path(path).exists():
            return

        # Prefer new path if both exist
        use_path = path
        try:
            with open(use_path, 'rb') as f:
                data = pickle.load(f)
                if data:
                    self.docstore.mset(list(data.items()))
                    logger.info(f"Loaded {len(data)} items from docstore")
        except Exception as e:
            logger.warning(f"Failed to load docstore from {use_path}: {e}")
    
    def _save_docstore(self):
        """Save docstore data to pickle file."""
        try:
            # Get all keys from docstore
            all_keys = list(self.docstore.store.keys()) if hasattr(self.docstore, 'store') else []
            if all_keys:
                data = {key: self.docstore.store[key] for key in all_keys}
                with open(self.docstore_file, 'wb') as f:
                    pickle.dump(data, f)
                logger.debug(f"Saved {len(data)} items to docstore ({self.docstore_file})")
        except Exception as e:
            logger.error(f"Failed to save docstore: {e}")

    def _add_content_type(self, contents: list[str], summaries: list[str], content_type: str) -> int:
        """
        Add content and summaries for a specific content type with deduplication
        
        Returns:
            Number of new documents added (after deduplication)
        """
        added_count = 0

        for i, (content, summary) in enumerate(zip(contents, summaries)):
            # Generate stable content-based ID
            if content_type=='text':
                content_id = cache_manager.generate_content_id(content.text)
            elif content_type=='table':
                content_id = cache_manager.generate_content_id(content.metadata.text_as_html)
            else:
                content_id = cache_manager.generate_content_id(content)
            
            # Check if this content already exists in vector store
            existing_docs = self.vector_store.get(where={"content_id": content_id})
            if existing_docs and len(existing_docs.get('ids', [])) > 0:
                logger.debug(f"Skipping duplicate {content_type} content: {content_id[:8]}...")
                continue
            
            # Store original content in docstore
            self.docstore.mset([(content_id, content)])
            
            # Store summary in vector store with metadata
            summary_doc = Document(
                page_content=summary,
                metadata={
                    "doc_id": content_id,
                    "content_id": content_id,
                    "content_type": content_type
                }
            )
            self.vector_store.add_documents([summary_doc], ids=[content_id])
            added_count += 1
            logger.debug(f"Added new {content_type} document: {content_id[:8]}...")
        
        return added_count

    @handle_errors("document storage")
    def add_documents(self, texts, text_summaries, tables, table_summaries, images, image_summaries):
        """Add documents with deduplication based on content hashing"""
        
        logger.info(f"Input counts - texts: {len(texts)}, tables: {len(tables)}, images: {len(images)}")
        logger.info(f"Summary counts - text_summaries: {len(text_summaries)}, table_summaries: {len(table_summaries)}, image_summaries: {len(image_summaries)}")
        
        # Add each content type with deduplication
        text_added = self._add_content_type(texts, text_summaries, "text")
        table_added = self._add_content_type(tables, table_summaries, "table") 
        image_added = self._add_content_type(images, image_summaries, "image")
        
        total_added = text_added + table_added + image_added
        logger.info(f"Added {total_added} new documents (texts: {text_added}, tables: {table_added}, images: {image_added})")
        
        # Save docstore after adding documents
        if total_added > 0:
            self._save_docstore()
        
        # Debug: Check total documents in vector store
        try:
            all_docs = self.vector_store.get()
            logger.info(f"Total documents in vector store: {len(all_docs.get('ids', []))}")
        except Exception as e:
            logger.warning(f"Could not get vector store count: {e}")

    @handle_errors("document retrieval")
    def call(self,query):
        result = self.retriever.invoke(query)
        logger.info(f"Retrieved {len(result)} documents")
        return result


