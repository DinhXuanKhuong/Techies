# Sử dụng base image Node.js 22
FROM node:22-alpine

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép file package.json và cài đặt thư viện
# Cũng tối ưu hóa cache
COPY package*.json ./
RUN npm install

# Sao chép toàn bộ code của frontend
COPY . .

# Mở cổng 3000
EXPOSE 3000

# Lệnh để chạy server dev của Next.js
CMD ["npm", "run", "dev"]