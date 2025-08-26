from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from base64 import b64decode
from .vector_store import DocumentManager
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from .llm_manager import llm_manager
from .utils import handle_errors, logger, RAGError



class RAG:
    def __init__(self, document_manager: DocumentManager):
        self.document_manager = document_manager
        self.retriever = self.document_manager.retriever
        
        # Use cached LLM instance
        self.llm = llm_manager.get_llm()
        
        # Initialize chains as None - will be built on first use
        self.chain = None
        self.chain_with_sources = None
        
        logger.info("RAG pipeline initialized")

    def _parse_docs(self, docs):
        b64 = []
        text = []
        for doc in docs:
            try:
                b64decode(doc)
                b64.append(doc)
            except Exception:
                text.append(doc)
        return {"images": b64, "texts": text}

    def _build_prompt(self, kwargs):
        docs_by_type = kwargs["context"]
        user_question = kwargs["query"]

        context_text = ""
        if len(docs_by_type["texts"]) > 0:
            for text_e in docs_by_type["texts"]:
                context_text += text_e.text + "\n"

        prompt_template = f"""Answer the question based only on the following context, which can include text, tables, and the below image.
        Context: {context_text.strip()}
        Question: {user_question}"""
        
        prompt_content = [{"type": "text", "text": prompt_template}]

        if len(docs_by_type["images"]) > 0:
            for image in docs_by_type["images"]:
                prompt_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"}
                })

        return [HumanMessage(prompt_content)]


    def _ensure_chains_built(self):
        """Build chains only if not already built"""
        if self.chain is None or self.chain_with_sources is None:
            logger.info("Building RAG chains")
            self.chain = (
                {
                    "context": self.retriever | RunnableLambda(self._parse_docs),
                    "query": RunnablePassthrough()
                }
                | RunnableLambda(self._build_prompt)
                | self.llm
                | StrOutputParser()
            )
            
            self.chain_with_sources = (
                {
                    "context": self.retriever | RunnableLambda(self._parse_docs),
                    "query": RunnablePassthrough()
                }
                | RunnablePassthrough().assign(
                    response = (
                        RunnableLambda(self._build_prompt)
                        | self.llm
                        | StrOutputParser()
                    )
                )
            )

    @handle_errors("RAG query processing")
    def call(self, query: str):
        if not query.strip():
            raise RAGError("Empty query provided")
            
        self._ensure_chains_built()
        logger.info(f"Processing query: {query[:50]}...")
        
        result = self.chain_with_sources.invoke(query)
        return result