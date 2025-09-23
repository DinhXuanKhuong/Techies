# Techies - AI Dermatology Assistant

Ứng dụng AI hỗ trợ chẩn đoán bệnh da liễu với FastAPI backend và Next.js frontend.

## 🚀 Chạy với Docker 

### Yêu cầu
- **Docker Desktop** hoặc **Docker Engine** + **Docker Compose**
- Git (để clone repository nếu có sẵn project folder thì khỏi)
- Git LFS (nếu clone từ GitHub)

### Các bước chi tiết

1. **Di chuyển vào thư mục dự án**
```bash
cd techies
# Hoặc tên thư mục dự án của bạn
```

2. **Kiểm tra file docker-compose.yml tồn tại**
```bash
ls docker-compose.yml
# Phải thấy file docker-compose.yml
```

3. **Chạy ứng dụng**
```bash
docker compose up --build
```

**Giải thích các flag:**
- `up`: Khởi động containers
- `--build`: Build lại images (bắt buộc lần đầu hoặc khi có thay đổi code)

### Truy cập ứng dụng
- **Frontend (Web UI)**: http://localhost:3000


### Dừng ứng dụng
```bash
# Cách 1: Nhấn Ctrl+C trong terminal đang chạy
# Cách 2: Từ terminal khác trong cùng thư mục:
docker compose down
```

### ⚠️ Lưu ý quan trọng
- **Phải chạy lệnh trong thư mục chứa file `docker-compose.yml`**
- Nếu có nhiều Docker projects khác, đảm bảo đang ở đúng thư mục
- Kiểm tra port 3000 và 8000 không bị chiếm bởi ứng dụng khác

---

## ⚙️ Cấu hình biến môi trường(nếu clone từ github, còn có sẵn project folder thì khỏi cần)

Ứng dụng cần các biến môi trường để kết nối với dịch vụ bên ngoài.

### Backend Environment
Tạo file `backend/.env` với nội dung:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
OPENAI_KEY=your_openai_key
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langchain_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=derm-agent
```

### Frontend Environment  
Tạo file `frontend/.env.local` với nội dung:
```env
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=your_supabase_publishable_key
```

**Lưu ý**: Các file `.env.example` và `.env.local.example` có sẵn để tham khảo.

---

## 🛠️ Development Mode (Tùy chọn)

Nếu muốn chạy trực tiếp mà không dùng Docker:

### Yêu cầu
- Python ≥ 3.10
- Node.js ≥ 18 + npm

### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend  
npm install
npm run dev
```

---

## 📁 Cấu trúc dự án

```
techies/
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env.example
│   └── main.py
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── .env.local.example
│   └── ...
└── README.md
```

---

## 🔧 Troubleshooting

### Lỗi thường gặp

**"No such file or directory: docker-compose.yml"**
```bash
# Kiểm tra bạn đang ở đúng thư mục
pwd
ls -la
# Phải thấy file docker-compose.yml
```

**Port đã được sử dụng (port 3000 hoặc 8000):**
```bash
# Dừng containers hiện tại
docker compose down

# Xem process nào đang dùng port
lsof -i :3000  # Linux/macOS
netstat -ano | findstr :3000  # Windows

# Hoặc thay đổi port trong docker-compose.yml
```

**Lỗi "Cannot connect to Docker daemon":**
```bash
# Đảm bảo Docker Desktop đang chạy
# Hoặc start Docker service trên Linux
sudo systemctl start docker
```

**Rebuild sau khi thay đổi code:**
```bash
docker compose down
docker compose up --build
```

**Xóa toàn bộ containers và images (reset hoàn toàn):**
```bash
docker compose down
docker system prune -a
# Cảnh báo: Lệnh này xóa TẤT CẢ images/containers không sử dụng
```

### Model Files (Git LFS)
Nếu clone từ GitHub và gặp lỗi thiếu model:
```bash
git lfs install
git lfs pull
```

---

## 📊 Tech Stack

- **Backend**: FastAPI, Python 3.12, LangChain, OpenAI/Groq
- **Frontend**: Next.js 15, React, TypeScript  
- **Database**: Supabase
- **Container**: Docker + Docker Compose
- **AI/ML**: OpenCV, Albumentations, PyTorch

---

## 👥 Đóng góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.