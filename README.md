# Techies

## Hướng dẫn cài đặt và chạy chương trình

### 1. Tạo môi trường ảo (Windows)

Mở PowerShell hoặc Command Prompt và chạy lệnh sau:

```powershell
python -m venv env
```

### 2. Kích hoạt môi trường ảo

```cmd
.\env\Scripts\activate
```

### 3. Cài đặt các thư viện cần thiết

```powershell
pip install -r requirements.txt
```

### 4. Chạy chương trình dự đoán

```powershell
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

```

### 5. Thay đổi đường dẫn ảnh hoặc mô hình (nếu cần)

- Sửa các biến `model_path` và `image_path` trong file `main.py` để phù hợp với dữ liệu nhập vào.

---
### 6. Chạy web frontend 

```powershell
cd frontend
npm install
npm run dev
```
**Lưu ý:**

- Python 3.10.x. `x` nào cũng được.
- Nếu gặp lỗi thiếu thư viện, kiểm tra lại bước cài đặt requirements.
- Các model .pth hoặc thư mục Fine-tuned cần clone qua Git LFS. Nếu chưa cài:
```powershell
git lfs install
git lfs pull
