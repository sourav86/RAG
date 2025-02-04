from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_pdf_text(pdf_docs):
    """
    Desc:
    Extract the text from requested pdf files.

    Args:
    pdf_docs (str) -> File path of the pdf file(s)

    Return:
     str
    """    
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    Desc:
    Convert extracted text into chunks for embedding process.

    Args:
    text (str) -> Extracted text

    Return:
     List[str]
    """       
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks