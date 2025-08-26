from unstructured.partition.pdf import partition_pdf
from .utils import handle_errors, validate_file_path, logger, DocumentProcessingError
from .config import settings  
from typing import Any

@handle_errors("PDF document partitioning")
def partition(file_path: str = None) -> tuple[list[Any], list[Any], list[str]]:
    """
    Extract and partition content from PDF document
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Tuple containing (tables, text_elements, images)
        
    Raises:
        DocumentProcessingError: If PDF processing fails
        FileNotFoundError: If file doesn't exist
    """
    # Use default path from settings when not provided
    if file_path is None:
        file_path = settings.default_pdf_path

    # Validate input file
    validate_file_path(file_path)
    
    try:
        #提取
        elements = partition_pdf(
        filename = file_path,
        infer_table_structure = True,
        strategy = 'hi_res',
        extract_images_in_pdf=True,
        extract_image_block_types=["Image"],
        # extract_image_block_output_dir= r"./content/pdf_images"
        extract_image_block_to_payload=True,
        chunking_strategy = "by_title",
        max_characters = 6000, #希望每个块不要超过多少字。库会尽量在合适的自然边界进行切分。
        new_after_n_chars = 10000, #设定一个硬性阈值。
        combine_text_under_n_chars = 2000, #如果某个块不足 2000 字符，会和相邻块合并。
        )
        
        if not elements:
            raise DocumentProcessingError(f"No content extracted from PDF: {file_path}")
        
        logger.info(f"Extracted {len(elements)} elements from PDF")
        
        #拆分，将提取到的文字、表格、图片分别放入三个列表
        tables = []
        text = []
        images = []
        
        for e in elements:
            text.append(e)
            if hasattr(e, 'metadata') and hasattr(e.metadata, 'orig_elements'):
                for orig in e.metadata.orig_elements:
                    if "Table" in str(type(orig)):
                        tables.append(orig)
                    if "Image" in str(type(orig)) and hasattr(orig.metadata, 'image_base64'):
                        images.append(orig.metadata.image_base64)
        
        logger.info(f"Processed: {len(text)} text elements, {len(tables)} tables, {len(images)} images")
        return tables, text, images
        
    except Exception as e:
        raise DocumentProcessingError(f"Failed to process PDF {file_path}: {str(e)}") from e

