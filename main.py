from src.utils import setup_logging, logger
from src.partition import partition
from src.summaries import summarize, image_summarize
from src.vector_store import DocumentManager
from src.rag_pipeline import RAG
from src.config import settings

def is_document_processed(document_manager):
    """Check if documents are already processed in vector store"""
    try:
        all_docs = document_manager.vector_store.get()
        doc_count = len(all_docs.get('ids', []))
        logger.info(f"Found {doc_count} existing documents in vector store")
        return doc_count > 0
    except Exception as e:
        logger.warning(f"Could not check vector store: {e}")
        return False

def main():
    # Setup logging
    setup_logging()
    logger.info("Starting MultiRAG pipeline")
    
    # Build knowledge base
    document_manager = DocumentManager()
    
    # Check if we need to process the PDF
    if not is_document_processed(document_manager):
        logger.info("No existing documents found, processing PDF...")
        
        # Process document
        tables, texts, images = partition(settings.default_pdf_path)
        text_summaries = summarize(texts)
        table_summaries = summarize([table.metadata.text_as_html for table in tables])
        image_summaries = image_summarize(images)
        
        # Store in vector store
        document_manager.add_documents(texts, text_summaries, tables, table_summaries, images, image_summaries)
    else:
        logger.info("Using existing processed documents")
    
    # Query
    rag_instance = RAG(document_manager)
    query = "What is multihead?"
    result = rag_instance.call(query)
    
    print(f"Query: {query}")
    print(f"Answer: {result.get('response', result)}")
    
if __name__ == "__main__":
    main()
