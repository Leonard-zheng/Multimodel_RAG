from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import yaml


load_dotenv()
# llm = ChatOllama(
#     model="qwen2.5:7b-instruct",
#     temperature=0,
# )
def create_summary_chain(prompt_config_path='prompt.yml'):
    llm=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite',temperature=0)
    with open(prompt_config_path,'r',encoding='utf-8') as f:
        prompts=yaml.safe_load(f)['prompts']

    prompts = ChatPromptTemplate.from_messages([
        ('system',prompts['system_prompt']),
        ('human',prompts['user_template'])
    ])
    rag_chain=(
        {
            "element":RunnablePassthrough(),
        }
        | prompts
        | llm
        | StrOutputParser()
    )
    return rag_chain

def create_image_summary_chain():
    llm=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite',temperature=0)
    prompts = ChatPromptTemplate.from_messages([
        ("human",[
            {"type":"text",
             "text":"""Describe the image in detail. For context,the image is part of a research paper.
    Be specific about graphs, such as bar plots."""},
            {"type":"image_url",
             "image_url":{"url":"data:image/jpeg;base64,{image}"}}
        ]
         )
    ])
    rag_chain = {"image":RunnablePassthrough()} | prompts | llm | StrOutputParser()

    return rag_chain

def image_summarize(images):
    chain=create_image_summary_chain()
    summaries = chain.batch(images)
    return summaries

def summarize(data):
    chain=create_summary_chain()
    summaries = chain.batch(data,config={"max_concurrency":3})
    return summaries

