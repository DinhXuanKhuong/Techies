

'''
- Tạo datasets = đánh dấu tài liệu liên quan
    + Lưu vào bộ datasets: query - index
    +  
- Hàm evaluate tính giá trị: Recall, Precision, MRR
    + Recall: chạy hàm xuất ra index của 
    + Precision: 
    + MRR: 

'''
 
import json
from typing import List, Dict, Any

from uuid import uuid4
from langchain.vectorstores import Chroma
from db.url import retrieve_relevant_chunks, load_titles_from_chroma, load_data_to_chromaDB
from langchain_huggingface import HuggingFaceEmbeddings

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 400
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "medical_collection"
PERSIST_DIR = "./chroma_db"
TOP_TITLES = 5
NUM_RETRIVER = 500
NUM_RERANKS = 50
NUM_RESULTS = 10

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_eval_dataset(filename: str) -> List[Dict[str, Any]]:
    """
    Load the evaluation dataset from a JSON file.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_precision(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Calculate precision: the fraction of retrieved documents that are relevant.
    """
    if not retrieved_ids:
        return 0.0
    true_positives = len(set(retrieved_ids) & set(relevant_ids))
    return true_positives / len(retrieved_ids)

def calculate_recall(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Calculate recall: the fraction of relevant documents that were retrieved.
    """
    if not relevant_ids:
        return 0.0
    true_positives = len(set(retrieved_ids) & set(relevant_ids))
    return true_positives / len(relevant_ids)

def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank: the reciprocal of the rank of the first relevant document.
    """
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def evaluate_retrieval(vector_store, titles: List[str], eval_dataset: List[Dict[str, Any]], num_results) -> Dict[str, float]:
    """
    Evaluate the retrieval system using precision, recall, and MRR.
    """
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

    # vector_store, titles = load_data_to_chromaDB(url_file="db/data/url_data_test.csv")

    vector_store = Chroma(
        collection_name="medical_collection",
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIR  
    )
    titles = load_titles_from_chroma(vector_store)

    # docs = retrieve_relevant_chunks(vector_store, titles, eval_dataset[0].get("query")) # return 10 list docs
    
    # true_positives = 0
    # for doc in docs:
    #     if doc.metadata.get("index") in eval_dataset[0].get("relevant_ids"):
    #         print(doc.metadata.get("index"))
    #         true_positives += 1
    
    # print(f"precision: {true_positives/len(docs)}")
    results = evaluate_retrieval(vector_store, titles, eval_dataset, num_results=NUM_RESULTS)
    
    print("Average Metrics:")
    print(f"Average Precision: {results['avg_precision']:.4f}")
    print(f"Average Recall: {results['avg_recall']:.4f}")
    print(f"Average MRR: {results['avg_mrr']:.4f}")
