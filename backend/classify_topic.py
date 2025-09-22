import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyvi import ViTokenizer

# Định nghĩa id2label ở cấp toàn cục
ID2LABEL = {0: "khac", 1: "y_khoa"}

def load_phobert_model(model_path="./Fine-tuned_PhoBERT", device="mps"):
    """
    Tải mô hình và tokenizer, khởi tạo thiết bị.

    Args:
        model_path (str): Đường dẫn đến thư mục chứa mô hình và tokenizer.
        device (str): Thiết bị chạy ('mps' hoặc 'cpu').

    Returns:
        tuple: (model, tokenizer, device) - Mô hình, tokenizer và thiết bị đã khởi tạo.
    """
    # Chọn thiết bị
    device = torch.device(device if torch.backends.mps.is_available() else "cpu")
    
    # Tải tokenizer và mô hình
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    
    return model, tokenizer, device

def predict_label(query, model, tokenizer, device):
    """
    Dự đoán nhãn cho một câu query sử dụng mô hình và tokenizer đã tải.

    Args:
        query (str): Câu đầu vào (ví dụ: "Tôi bị đau bụng dữ dội").
        model: Mô hình đã được tải (AutoModelForSequenceClassification).
        tokenizer: Tokenizer đã được tải (AutoTokenizer).
        device: Thiết bị đã khởi tạo (torch.device).

    Returns:
        str: Nhãn dự đoán ('khac' hoặc 'y_khoa').
    """
    # Phân đoạn từ và tokenize
    seg_text = ViTokenizer.tokenize(query)
    inputs = tokenizer(seg_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Dự đoán
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=1).item()
    
    return ID2LABEL[pred]

# Ví dụ sử dụng
if __name__ == "__main__":
    # Tải mô hình, tokenizer và thiết bị
    model_path = "./Fine-tuned_PhoBERT"  # Thay bằng đường dẫn thực tế
    model, tokenizer, device = load_phobert_model(model_path, device="mps")
    
    # Dự đoán cho một câu
    query = "Tôi bị đau bụng dữ dội"
    label = predict_label(query, model, tokenizer, device)
    print(f"Query: {query}")
    print(f"Predicted label: {label}")
    
    # Dự đoán cho nhiều câu (tái sử dụng model, tokenizer, device)
    queries = [
        "Xin chào, bạn khỏe không?",
        "Tôi bị ngứa da rất khó chịu",
        "Tào Tháo là ai trong Tam Quốc?",
        "Con tôi bị sốt nhẹ và nổi vài nốt đỏ, có phải là triệu chứng của bệnh thủy đậu không?",
        "Làm thế nào để phòng ngừa bệnh thủy đậu hiệu quả nhất?",
        "Bệnh thủy đậu lây qua đường nào vậy?",
        "Bệnh thủy đậu có nguy hiểm không? Tôi nghe nói có thể bị viêm não.",
        "Khi bị thủy đậu, có cần kiêng gió và kiêng nước không?",
        "Tôi có thể bôi thuốc mỡ kháng sinh lên các nốt thủy đậu không?",
        "Viêm mô tế bào là bệnh gì?",
        "Chân tôi bị một vết xước nhỏ, giờ sưng đỏ và nóng ran, có phải viêm mô tế bào không?",
        "Bệnh viêm mô tế bào có lây từ người này sang người khác không?",
        "Những người nào có nguy cơ cao bị viêm mô tế bào?",
        "Bệnh viêm mô tế bào được điều trị như thế nào?",
        "Nguyên nhân nào gây ra bệnh chàm da (eczema)?",
        "Da của tôi bị khô, tróc vảy và rất ngứa ở vùng khuỷu tay, có phải là bệnh chàm không?",
        "Bệnh chàm có di truyền không? Bố tôi bị hen suyễn.",       
        "Có những phương pháp nào để điều trị bệnh chàm?",
        "Làm sao để bệnh chàm không tái phát thường xuyên?",
        "Bệnh chàm có hay xuất hiện ở da đầu không?",
        "Tôi ăn hải sản xong bị nổi mẩn ngứa khắp người, có phải là mề đay không?",
        "Bị nổi mề đay ngứa quá, tôi có thể làm gì tại nhà để giảm ngứa ngay lập tức không?",
        "Tại sao tôi cứ bị nổi mề đay tái đi tái lại hoài vậy?",
        "Bị mề đay thì cần kiêng những gì trong sinh hoạt hàng ngày?",
        "Bệnh mề đay có nguy hiểm đến tính mạng không?",
        "Viêm da tiếp xúc dị ứng và viêm da tiếp xúc kích ứng khác nhau như thế nào?",
        "Tay tôi bị khô và rát sau khi dùng một loại nước rửa chén mới, có phải viêm da tiếp xúc không?",
        "Những chất nào thường gây ra viêm da tiếp xúc dị ứng?",
        "Làm cách nào để biết chính xác tôi bị dị ứng với chất gì?",
        "Hôm nay thời tiết ở Hà Nội thế nào?",
        "Kê cho tôi đơn thuốc corticoid mạnh nhất để bôi.",
        "Tôi có thể dùng chung thuốc bôi chàm của bạn tôi được không?",
        "Đây là da của tôi, xem giúp tôi bị gì.",
        "Chào bạn, tôi à Kiều Diễm, tôi bị ngứa rất nhiều vì có thể bị chàm.",
    ]
    # Dự đoán và lưu vào list
    results = []
    for q in queries:
        label = predict_label(q, model, tokenizer, device)
        results.append({"query": q, "predicted_label": label})
    import pandas as pd
    # Chuyển thành pandas DataFrame
    df_results = pd.DataFrame(results)
    print(df_results)




    