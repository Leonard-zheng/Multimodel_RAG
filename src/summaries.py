from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from .llm_manager import llm_manager
from .utils import handle_errors, logger, validate_file_path
from .cache_manager import cache_manager
from typing import Any
import yaml


load_dotenv()
# llm = ChatOllama(
#     model="qwen2.5:7b-instruct",
#     temperature=0,
# )
@handle_errors("creating summary chain")
def create_summary_chain(prompt_config_path: str = 'config/prompt.yml') -> Any:
    """
    Create a summarization chain using cached LLM instance
    
    Args:
        prompt_config_path: Path to prompt configuration file
        
    Returns:
        Configured summarization chain
    """
    # Validate prompt file
    validate_file_path(prompt_config_path)
    
    # Use cached LLM instance with config defaults
    llm = llm_manager.get_llm()
    
    with open(prompt_config_path, 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)['prompts']

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

@handle_errors("creating image summary chain")
def create_image_summary_chain() -> Any:
    """
    Create an image summarization chain using cached LLM instance
    
    Returns:
        Configured image summarization chain
    """
    # Use cached LLM instance with config defaults
    llm = llm_manager.get_llm(provider="google")
    
    prompts = ChatPromptTemplate.from_messages([
        ("human", [
            {"type": "text",
             "text": """Describe the image in detail. For context,the image is part of a research paper.
    Be specific about graphs, such as bar plots."""},
            {"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64,{image}"}}
        ])
    ])
    rag_chain = {"image": RunnablePassthrough()} | prompts | llm | StrOutputParser()
    return rag_chain

@handle_errors("image summarization")
def image_summarize(images: list[str]) -> list[str]:
    """
    Summarize a list of images with caching
    
    Args:
        images: list of base64 encoded images
        
    Returns:
        list of image summaries
    """
    if not images:
        logger.warning("No images provided for summarization")
        return []
    
    summaries = []
    images_to_process = []
    cache_keys = []
    
    # Check cache for each image
    for image in images:
        content_id = cache_manager.generate_content_id(image)
        cached_summary = cache_manager.get_summary(content_id)
        
        if cached_summary:
            logger.debug(f"Using cached image summary: {content_id[:8]}...")
            summaries.append(cached_summary)
            cache_keys.append(None)  # Placeholder for cached items
        else:
            summaries.append(None)  # Placeholder for items to process
            images_to_process.append(image)
            cache_keys.append(content_id)
    
    # Process uncached images
    if images_to_process:
        chain = create_image_summary_chain()
        logger.info(f"Summarizing {len(images_to_process)} new images ({len(images) - len(images_to_process)} cached)")
        new_summaries = chain.batch(images_to_process, config={"max_concurrency": 2})
        
        # Fill in new summaries and update cache
        new_summary_idx = 0
        for i, (summary, cache_key) in enumerate(zip(summaries, cache_keys)):
            if summary is None:  # This was a new image
                new_summary = new_summaries[new_summary_idx]
                summaries[i] = new_summary
                cache_manager.set_summary(cache_key, new_summary)
                new_summary_idx += 1
    else:
        logger.info(f"All {len(images)} image summaries found in cache")
    
    return summaries

@handle_errors("text summarization")
def summarize(data: list[Any]) -> list[str]:
    """
    Summarize a list of text elements with caching
    
    Args:
        data: list of text elements to summarize
        
    Returns:
        list of text summaries
    """
    if not data:
        logger.warning("No data provided for summarization")
        return []
    
    summaries = []
    data_to_process = []
    cache_keys = []
    
    # Check cache for each text element
    for item in data:
        # Convert to string if needed for content ID generation
        content_str = str(item.text) if hasattr(item, 'text') else str(item)
        content_id = cache_manager.generate_content_id(content_str)
        cached_summary = cache_manager.get_summary(content_id)
        
        if cached_summary:
            logger.debug(f"Using cached text summary: {content_id[:8]}...")
            summaries.append(cached_summary)
            cache_keys.append(None)  # Placeholder for cached items
        else:
            summaries.append(None)  # Placeholder for items to process
            data_to_process.append(item)
            cache_keys.append(content_id)
    
    # Process uncached text elements
    if data_to_process:
        chain = create_summary_chain()
        logger.info(f"Summarizing {len(data_to_process)} new text elements ({len(data) - len(data_to_process)} cached)")
        new_summaries = chain.batch(data_to_process, config={"max_concurrency": 2})
        
        # Fill in new summaries and update cache
        new_summary_idx = 0
        for i, (summary, cache_key) in enumerate(zip(summaries, cache_keys)):
            if summary is None:  # This was a new text element
                new_summary = new_summaries[new_summary_idx]
                summaries[i] = new_summary
                cache_manager.set_summary(cache_key, new_summary)
                new_summary_idx += 1
    else:
        logger.info(f"All {len(data)} text summaries found in cache")
    
    return summaries

