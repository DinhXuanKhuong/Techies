import uuid
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions

# Khởi tạo text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Khởi tạo client
chroma_client = chromadb.PersistentClient(path="./medical_db")

# Khởi tạo embedding function (dùng Ollama để nhúng văn bản)
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    model_name="nomic-embed-text"
)

def load_to_chromadb(raw_texts):

    chunks = []
    for text in raw_texts:
        if not isinstance(text, str):
            raise ValueError("Mỗi phần tử trong raw_texts phải là string")
        chunks.extend(splitter.split_text(text))
    
    try:
        chroma_client.delete_collection("medical")
    except:
        pass  # Không có collection thì thôi

    collection = chroma_client.create_collection(
        name="medical",
        embedding_function=ollama_ef
    )

    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

    collection.add(
        documents=chunks,
        ids=ids
    )
    return chunks

