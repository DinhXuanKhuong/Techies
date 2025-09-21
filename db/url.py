from scrape import scrape_website, clean_body_content, extract_body_content
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# New update: from langchain-chroma import Chroma

import shutil
import os
from sklearn.metrics.pairwise import cosine_similarity

# Rerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 400
# EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
RERANK_MODEL = "BAAI/bge-reranker-base"
COLLECTION_NAME = "medical_collection"
PERSIST_DIR = "./chroma_db"
TOP_TITLES = 15
NUM_RETRIEVER = 500
NUM_RERANKS = 15
NUM_RESULTS = 10

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
rerank_model = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)

def load_embedding_model(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

def load_rerank_model(model_name: str = "BAAI/bge-reranker-base"):
    return HuggingFaceCrossEncoder(model_name=model_name)

def read_data_url(filename: str):
    list_url = []
    with open(filename, "r") as file:
        for line in file:
            url = line.strip()
            if url and url not in list_url:
                list_url.append(url)
    return list_url

def split_text_to_chunks(document_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = splitter.split_text(document_text)
    print(f"Number of chunks: {len(chunks)}")
    return chunks

def store_chunks_in_chroma(chunks, vector_store, title, url, persist_dir=PERSIST_DIR):
    metadatas = [{"title": title, "url": url, "index": title + "_" + str(i)} for i in range(len(chunks))]
    ids = [f"{title}_{i}" for i in range(len(chunks))]

    if vector_store is None:
        vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=embedding_model,
            collection_name=COLLECTION_NAME,
            persist_directory=persist_dir,
            metadatas=metadatas,
            ids=ids
        )
    else:
        if not chunks:
            return vector_store
        vector_store.add_texts(chunks, metadatas=metadatas, ids=ids)
    return vector_store

def get_top_titles(query_text, titles, top_k=TOP_TITLES):
    query_vec = embedding_model.embed_query(query_text)
    scores = []
    for title in set(titles):
        title_vec = embedding_model.embed_query(title)
        score = cosine_similarity([query_vec], [title_vec])[0][0]
        scores.append((title, score))
    top_titles = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    return [t[0] for t in top_titles]


def load_data_to_chromaDB(url_file):
    url_list = read_data_url(url_file)

    # Xóa chroma cũ
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    vector_store = None
    titles = []

    for url in url_list:
        print(f"Scraping url: {url}")
        html = scrape_website(url)
        body_content, title = extract_body_content(html)
        cleaned_content = clean_body_content(body_content)
        titles.append(title)
        chunks = split_text_to_chunks(cleaned_content)
        vector_store = store_chunks_in_chroma(chunks, vector_store, title, url)

    return vector_store, titles

def load_titles_from_chroma(vector_store):
    # lấy tất cả documents trong DB
    docs = vector_store.get(include=["metadatas"])
    titles = set()
    for meta in docs["metadatas"]:
        if meta and "title" in meta:
            titles.add(meta["title"])
    return list(titles)

# # Ideas: Top relevant chunks -> reranking -> filter chunks by top titles -> retrieve top chunks from filtered chunks
# def retrieve_relevant_chunks(vector_store, titles, query_text, num_results=NUM_RESULTS):
#     top_titles = get_top_titles(query_text, titles, top_k=TOP_TITLES)

#     base_retriever = vector_store.as_retriever(search_kwargs={"k": num_results * len(top_titles)})

#     # reranker
#     compressor = CrossEncoderReranker(model=rerank_model, top_n=NUM_RERANKS)
#     compression_retriever = ContextualCompressionRetriever(
#         base_compressor=compressor,
#         base_retriever=base_retriever
#     )

#     reranked_docs = compression_retriever.invoke(query_text) #doc = page_content + metadata

#     filtered_docs = [doc for doc in reranked_docs if doc.metadata.get("title") in top_titles]

#     seen = set()
#     # unique_chunks = []
#     unique_docs = []
#     for doc in filtered_docs:
#         if doc.page_content not in seen:
#             seen.add(doc.page_content)
#             unique_docs.append(doc)
#         # if len(unique_chunks) >= num_results:
#         if len(unique_docs) >= num_results:
#             break
#     return unique_docs

def retrieve_relevant_chunks(vector_store, titles, query_text, num_results=NUM_RESULTS):
    top_titles = get_top_titles(query_text, titles, top_k=TOP_TITLES)

    docs_and_scores = vector_store.similarity_search_with_score(query_text, k=NUM_RETRIEVER)
    filtered_docs = [doc for doc, score in docs_and_scores if score > 0.3]
    candidate_docs = [doc for doc in filtered_docs if doc.metadata.get("title") in top_titles]

    # reranker
    compressor = CrossEncoderReranker(model=rerank_model, top_n=NUM_RERANKS)
    reranked_docs = compressor.compress_documents(candidate_docs, query_text)

    seen = set()
    unique_docs = []
    for doc in reranked_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
        if len(unique_docs) >= num_results:
            break

    return unique_docs



# vector_store, titles = load_data_to_chromaDB(url_file="db/data/url_data_test.csv")

# vector_store = Chroma(
#     collection_name="medical_collection",
#     embedding_function=embedding_model,
#     persist_directory=PERSIST_DIR  
# )

# titles = load_titles_from_chroma(vector_store)

# user_input = "Cách điều trị bệnh thủy đậu là gì?"
# query = user_input
# chunks = retrieve_relevant_chunks(vector_store, titles, query)
# print("-------------------------------------------------------------------------------------")
# print(f"Query: {query}")

# print("--------------------------------------------------------------------------------------")
# print("Top chunks:")
# for chunk in chunks:
#     print(chunk.page_content)
#     print("-------------------------------------------------------------------------------------")  

import json
from typing import List, Dict, Any
from uuid import uuid4

def load_eval_dataset(filename: str) -> List[Dict[str, Any]]:
    """
    Load the evaluation dataset from a JSON file.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_precision(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    if not retrieved_ids:
        return 0.0
    true_positives = len(set(retrieved_ids) & set(relevant_ids))
    return true_positives / len(retrieved_ids)

def calculate_recall(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    if not relevant_ids:
        return 0.0
    true_positives = len(set(retrieved_ids) & set(relevant_ids))
    return true_positives / len(relevant_ids)

def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str], k: int = None) -> float:
    if k:
        retrieved_ids = retrieved_ids[:k]
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def evaluate_retrieval(vector_store, titles: List[str], eval_dataset: List[Dict[str, Any]], num_results) -> Dict[str, float]:
    precisions = []
    recalls = []
    mrrs = []

    for item in eval_dataset:
        query = item['query']
        relevant_ids = item['relevant_ids']
        
        # Retrieve chunks for the query
        chunks = retrieve_relevant_chunks(vector_store, titles, query, num_results=num_results)
        
        # Extract document IDs from retrieved chunks
        retrieved_ids = [chunk.metadata.get('index', f"{chunk.metadata.get('title')}_{uuid4()}") for chunk in chunks]
        
        # Calculate metrics
        precision = calculate_precision(retrieved_ids, relevant_ids)
        recall = calculate_recall(retrieved_ids, relevant_ids)
        mrr = calculate_mrr(retrieved_ids, relevant_ids)
        
        precisions.append(precision)
        recalls.append(recall)
        mrrs.append(mrr)
        
        with open("experiment_evaluation.csv", "a", encoding="utf-8") as f:
            f.write(f'"{query}", {precision}, {recall}, {mrr}\n')

        print(f"Query: {query}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, MRR: {mrr:.4f}")
        print("-------------------------------------------------------------------------------------")

    # Calculate average metrics
    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0

    return {
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_mrr': avg_mrr
    }


if __name__ == "__main__":
    # Load evaluation dataset
    eval_dataset = load_eval_dataset("eval_dataset.json")
    print(f"Loaded {len(eval_dataset)} evaluation queries.")
    # print(eval_dataset[0]) 
    # print(eval_dataset[0].get("query"))

    vector_store, titles = load_data_to_chromaDB(url_file="db/data/url_data.csv")

    # vector_store = Chroma(
    #     collection_name="medical_collection",
    #     embedding_function=embedding_model,
    #     persist_directory=PERSIST_DIR  
    # )
    # titles = load_titles_from_chroma(vector_store)

    results = evaluate_retrieval(vector_store, titles, eval_dataset, num_results=NUM_RESULTS)
    
    print("Average Metrics:")
    print(f"Average Precision: {results['avg_precision']:.4f}")
    print(f"Average Recall: {results['avg_recall']:.4f}")
    print(f"Average MRR: {results['avg_mrr']:.4f}")

    # query = "Viêm mô tế bào là gì?"
    # chunks, top_titles = retrieve_relevant_chunks(vector_store, titles, query)


    # print("-------------------------------------------------------------------------------------")
    # print(f"Query: {query}")
    # print("--------------------------------------------------------------------------------------")
    # print("Top title: ")
    # for t in top_titles:
    #     print(t)
    # print("--------------------------------------------------------------------------------------")
    # print("Top chunks:")
    # for chunk in chunks:
    #     print(chunk.page_content)
    #     print("-------------------------------------------------------------------------------------")


    
