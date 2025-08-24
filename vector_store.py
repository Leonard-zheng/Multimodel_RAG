import uuid
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import MultiVectorRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

class DocumentManager:
    def __init__(self,persist_directory="./chroma_db"):
        self.embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.vector_store = Chroma(
            collection_name = "multirag",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        self.memory_store = InMemoryStore()
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vector_store,
            docstore=self.memory_store,
            id_key='doc_id'
        )

    def add_documents(self,texts,text_summaries,tables,table_summaries,images,image_summaries):
        #texts 的储存，doc 和 vector
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        self.memory_store.mset(list(zip(doc_ids,texts)))

        summary_text=[
            Document(page_content=summary,metadata={"doc_id":doc_ids[i]}) for i,summary in enumerate(text_summaries)
        ]
        self.vector_store.add_documents(summary_text)

        #tables的储存
        table_ids = [str(uuid.uuid4()) for _ in tables]
        self.memory_store.mset(list(zip(table_ids, tables)))

        summary_table = [
            Document(page_content=summary, metadata={"doc_id": table_ids[i]}) for i, summary in enumerate(table_summaries)
        ]
        self.vector_store.add_documents(summary_table)

        #images的储存
        image_ids = [str(uuid.uuid4()) for _ in images]
        self.memory_store.mset(list(zip(image_ids, images)))

        summary_image = [
            Document(page_content=summary, metadata={"doc_i d": image_ids[i]}) for i, summary in enumerate(image_summaries)
        ]
        self.vector_store.add_documents(summary_image)

    def call(self,query):
        result=self.retriever.invoke(query)
        return result




