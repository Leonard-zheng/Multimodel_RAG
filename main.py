from dotenv import load_dotenv
load_dotenv()
from partition import partition
from summaries import summarize,image_summarize
from vector_store import DocumentManager
from rag_pipeline import RAG

def main():
    tables,texts,images = partition(file_path= r"./content/attention-is-all-you-need.pdf")
    text_summaries = summarize(texts)
    table_summaries = summarize([table.metadata.text_as_html for table in tables])
    image_summaries = image_summarize(images)
    document_manager=DocumentManager()
    document_manager.add_documents(texts,text_summaries,tables,table_summaries,images,image_summaries)
    rag_instance = RAG(document_manager)
    query="What is multihead?"
    rag_instance.call(query)
