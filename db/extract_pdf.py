# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_data_from_pdf():
    loader = PyPDFLoader("data/dalieu.pdf")
    pages = loader.load()
    for i, page in enumerate(pages):
        if i == 0:
            print(page.page_content)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, 
                                                   chunk_overlap=200,
                                                   length_function=len,
                                                   separators=["\n\n", "\n", " "])
    print(text_splitter.split_text(pages))



# load_data_from_pdf()