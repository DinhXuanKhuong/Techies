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

### 4. Cấu hình biến môi trường(Nếu clone repo qua Github hoặc chưa có sẵn các file .env và .env.local)

Chương trình cần các biến môi trường để kết nối với dịch vụ bên ngoài.  
Repo có sẵn các file mẫu:  

- `backend/.env.example`
- `frontend/.env.local.example`

#### Backend
Trong thư mục `backend`, tạo file `.env` dựa trên `.env.example`:  

```powershell
cd backend
cp .env.example .env
```
Nội dung tối thiểu của .env
```powershell
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
OPENAI_KEY=your_openai_key
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langchain_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=derm-agent
```


#### Frontend 
Trong thư mục `frontend`, tạo file `.env.local` dựa trên `.env.local.example`:

```powershell
cd frontend
cp .env.local.example .env.local
```
Nội dung tối thiểu của .env
```powershell
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=your_supabase_publishable_key
```
### 5. Chạy chương trình backend

```powershell
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

```


---
### 6. Chạy web frontend 

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
