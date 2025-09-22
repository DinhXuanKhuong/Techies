from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from typing import List

# Cấu hình
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 400
# EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
RERANK_MODEL = "BAAI/bge-reranker-base"
COLLECTION_NAME = "medical_collection"
PERSIST_DIR = "./chroma_db"
NUM_RETRIEVER = 15
NUM_RERANKS = 10
NUM_RESULTS = 5

# Khởi tạo embedding và rerank model
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
rerank_model = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)

# Khởi tạo vector store
vector_store = Chroma(
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR
)

# Khởi tạo retriever với số lượng tài liệu ban đầu
retriever = vector_store.as_retriever(search_kwargs={"k": NUM_RETRIEVER})

# Khởi tạo ContextualCompressionRetriever với reranker
compressor = CrossEncoderReranker(model=rerank_model, top_n=NUM_RERANKS)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)


def retrieve_relevant_chunks(query_text: str) -> List[Document]:
    """
    Hàm lấy tài liệu liên quan từ vector store và rerank chúng.

    Args:
        query_text (str): Câu truy vấn từ người dùng.

    Returns:
        List[Document]: Danh sách các tài liệu đã được rerank.
    """
    try:
        print(f"[RAG] Retrieving documents for query: {query_text}")

        # Lấy tài liệu từ retriever
        candidate_docs = retriever.invoke(query_text)
        print(f"[RAG] Retrieved {len(candidate_docs)} candidate documents.")

        # Rerank tài liệu
        reranked_docs = compression_retriever.invoke(query_text)
        print(f"[RAG] Retrieved {len(reranked_docs)} documents after reranking.")

        # Giới hạn số lượng tài liệu trả về
        final_docs = reranked_docs[:NUM_RESULTS]
        print(f"[RAG] Returning {len(final_docs)} final documents.")

        return final_docs

    except Exception as e:
        print(f"[RAG] Error retrieving documents: {str(e)}")
        return []

# # Sử dụng hàm
# query = "Triệu chứng nổi mề đay là gì?"
# chunks = retrieve_relevant_chunks(query)

# # In nội dung các tài liệu
# for i, chunk in enumerate(chunks, 1):
#     print(f"Document {i}:")
#     print(chunk.metadata)
#     print(chunk.page_content)
#     print("-" * 50)