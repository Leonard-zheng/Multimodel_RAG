from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from base64 import b64decode
from vector_store import DocumentManager
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage



class RAG:
    def __init__(self,document_manager):
        self.document_manager=document_manager
        self.retriever=self.document_manager.retriever

        self.llm=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite',temperature=0)

    def _parse_docs(self,docs):
        """split images and texts"""
        b64=[]
        text=[]
        for doc in docs:
            try:
                b64decode(doc)
                b64.append(doc)
            except Exception as e:
                text.append(doc)
        return {"images":b64,"texts":text}

    def _build_prompt(self,kwargs):
        docs_by_type = kwargs["context"]
        user_question = kwargs["query"]

        context_text=""
        if len(docs_by_type["texts"])>0:
            for text_e in docs_by_type["texts"]:
                context_text+=text_e.text+"\n"

        prompt_template = f"""
        Answer the question based only on the following context, which can include text, tables, and the below image.
        Context: {context_text}
        Question: {user_question}
        """
        prompt_content=[{"type":"text","text":prompt_template}]

        if len(docs_by_type["images"])>0:
            for image in docs_by_type["images"]:
                prompt_content.append(
                    {"type":"image_url",
                     "image_url":{"url":f"data:image/jpeg;base64,{image}"}
                     }
                )

        return [HumanMessage(prompt_content)]


    def build_chain(self):
        # prompt = ChatPromptTemplate.from_messages([
        #     ("system", "You are a helpful AI assistant that can analyze text and images."),
        #     ("human", RunnableLambda(self._build_prompt))]
        # )
        self.chain=(
        {
            "context": self.retriever | RunnableLambda(self._parse_docs),
            "query": RunnablePassthrough()
        }
        | RunnableLambda(self._build_prompt)
        | self.llm
        | StrOutputParser()
        )
        self.chain_with_sources=(
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
        ))

    def call(self,query):
        self.build_chain()
        res=self.chain_with_sources.invoke(query)
        return res