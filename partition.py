from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
load_dotenv()
def partition(file_path= r"./content/attention-is-all-you-need.pdf",):
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

    #拆分，将提取到的文字、表格、图片分别放入三个列表
    tables=[]
    text=[]
    images=[]
    for e in elements:
        has_table=False
        text.append(e)
        for orig in e.metadata.orig_elements:
            if "Table" in str(type(orig)):
                has_table=True
                tables.append(orig)
            if "Image" in str(type(orig)):
                images.append(orig.metadata.image_base64)
    return tables,text,images


