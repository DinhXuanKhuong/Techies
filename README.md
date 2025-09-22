# Techies

## Hướng dẫn cài đặt và chạy chương trình

### Yêu cầu
Python ≥ 3.10.x (cho backend)

Node.js ≥ 18.x + npm (cho frontend)

Git LFS (nếu clone repo từ github)
### 1. Tạo môi trường ảo (Windows)

Mở PowerShell hoặc Command Prompt và chạy lệnh sau:

```powershell
python -m venv env
```

### 2. Kích hoạt môi trường ảo

```cmd
.\.venv\Scripts\activate
(Linux/macOS thì dùng: source .venv/bin/activate)
```

### 3. Cài đặt các thư viện Python cần thiết

```powershell
pip install -r requirements.txt
```

### 4. Chạy chương trình backend

```powershell
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

```


---
### 5. Chạy web frontend 

```powershell
cd frontend
npm install
npm run dev
```
**Lưu ý:**

- Nếu gặp lỗi thiếu thư viện, kiểm tra lại bước cài đặt requirements.
- Các model .pth hoặc thư mục Fine-tuned cần clone qua Git LFS(Nếu clone repo từ github). Nếu chưa cài:
```powershell
git lfs install
git lfs pull
