# Sử dụng base image Python 3.12
FROM python:3.12-slim

# Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# Sao chép file requirements và cài đặt thư viện
# Tối ưu hóa cache: chỉ cài lại khi requirements.txt thay đổi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ code của backend vào container
COPY . .

# Mở cổng 8000 để các container khác có thể giao tiếp
EXPOSE 8000

# Lệnh để chạy app (sử dụng reload để phát triển)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]