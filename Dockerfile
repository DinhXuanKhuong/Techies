# --- GIAI ĐOẠN 1: Build Frontend ---
# Sử dụng Node.js v22 để build code frontend
FROM node:22-alpine AS frontend-builder
WORKDIR /app

# Sao chép file package.json và cài đặt thư viện
COPY frontend/package*.json ./
RUN npm install

# Sao chép file .env.local để Next.js sử dụng trong quá trình build
COPY frontend/.env.local ./.env.local

# Sao chép toàn bộ code frontend và build ra các file tĩnh
COPY frontend/. .
RUN npm run build


# --- GIAI ĐOẠN 2: Build Image cuối cùng ---
# Sử dụng Python 3.12 làm nền
FROM python:3.12-slim

# Cài đặt Nginx và Supervisor vào bên trong image
RUN apt-get update && apt-get install -y nginx supervisor && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc chính
WORKDIR /app

# Cài đặt các thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép các file cấu hình Nginx và Supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY nginx.conf /etc/nginx/nginx.conf

# Sao chép TOÀN BỘ thư mục backend vào bên trong image
# Lệnh này sẽ bao gồm tất cả các file .py, .pth, .env, và các thư mục con
# như chroma_db, Fine-tuned_PhoBERT, medical_db, v.v.
COPY backend/ /app/backend/

# Sao chép KẾT QUẢ BUILD của frontend (từ GIAI ĐOẠN 1) vào image
COPY --from=frontend-builder /app/out /app/frontend/out

# Mở cổng 80 để bên ngoài có thể truy cập vào Nginx
EXPOSE 80

# Lệnh cuối cùng để khởi chạy Supervisor (nó sẽ tự động chạy Nginx và FastAPI)
CMD ["/usr/bin/supervisord"]