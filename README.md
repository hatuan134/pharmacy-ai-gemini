# AN TÂM — Hệ thống quản lý nhà thuốc tích hợp AI Gemini

**Đồ án quản lý nhà thuốc** | **Frontend:** React 19 + Vite 6 | **Backend:** Python FastAPI | **CSDL:** PostgreSQL | **AI:** Google Gemini

An Tâm hỗ trợ quản lý thuốc theo lô, nhập hàng, kiểm kê, bán hàng, hóa đơn, thống kê và tra cứu thông tin bằng AI. Giao diện tiếng Việt; nghiệp vụ và quyền truy cập được xử lý ở backend. Dự án có thể chạy trên Windows bằng Docker Desktop hoặc triển khai một dịch vụ Docker trên Render, kết nối PostgreSQL bên ngoài (ví dụ Neon).

> **Lưu ý:** README này mô tả mã nguồn `pharmacy-ai-gemini` được cung cấp, không mô tả dự án LIBRA (quản lý thư viện) từng có trong repository `DuAnTTCS_N5`. Hãy thay README khi mã nguồn nhà thuốc đã được đưa lên đúng repository/nhánh.

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Các nhóm chức năng](#2-các-nhóm-chức-năng)
3. [Công nghệ và kiến trúc](#3-công-nghệ-và-kiến-trúc)
4. [Cấu trúc thư mục](#4-cấu-trúc-thư-mục)
5. [Yêu cầu môi trường](#5-yêu-cầu-môi-trường)
6. [Cài đặt và chạy trên Windows](#6-cài-đặt-và-chạy-trên-windows)
7. [Cấu hình biến môi trường](#7-cấu-hình-biến-môi-trường)
8. [Chạy và kiểm tra thủ công](#8-chạy-và-kiểm-tra-thủ-công)

## 1. Giới thiệu

**An Tâm** là ứng dụng web phục vụ vận hành nhà thuốc, hỗ trợ nhiều vai trò nhân viên và tích hợp Google Gemini để hỏi đáp dữ liệu/quy trình trong phạm vi hệ thống. Mục tiêu là số hóa hoạt động thường ngày: danh mục thuốc, lô hàng, tồn kho, hóa đơn và báo cáo.

- **Đối tượng sử dụng:** Quản lý, dược sĩ, thu ngân.
- **Ứng dụng:** Giao diện web dùng trên máy tính hoặc trình duyệt di động.
- **Lưu trữ:** PostgreSQL; số lượng và giá trị tồn kho được quản lý theo lô, không chỉ theo một tổng số chung.
- **AI:** Gọi Gemini từ backend; tính sẵn sàng phụ thuộc API key, quyền truy cập model và hạn mức dịch vụ.

## 2. Các nhóm chức năng

| Nhóm chức năng | Nội dung |
|---|---|
| Tài khoản, phân quyền | Đăng nhập/đăng xuất, đổi mật khẩu, quản lý tài khoản nhân viên; phân vai trò quản lý, dược sĩ, thu ngân |
| Danh mục thuốc | Quản lý thuốc, nhóm thuốc, đơn vị tính, nhà cung cấp và trạng thái kinh doanh |
| Nhập hàng, quản lý lô | Tạo lô nhập, số lô, ngày nhập, hạn sử dụng, giá nhập/giá bán, số lượng tồn theo lô |
| Tồn kho, kiểm kê | Theo dõi tồn kho theo lô, điều chỉnh số lượng có lý do, ghi nhận biến động kho |
| Bán hàng, hóa đơn | Tìm thuốc, lựa chọn lô, giỏ hàng, ghi nhận phương thức thanh toán, lập/in hóa đơn, hủy hóa đơn theo quy tắc nghiệp vụ |
| Cảnh báo | Lô sắp hết hạn/hết hạn và thuốc dưới ngưỡng tồn |
| Tra cứu, thống kê | Tra cứu thuốc, lô; tổng hợp hóa đơn, doanh thu và báo cáo tồn |
| AI Gemini | Hỏi đáp thông tin/quy trình nhà thuốc, tổng hợp dữ liệu theo khả năng và quyền hạn được thực thi bởi phiên bản đang chạy |

**Giới hạn:** AI không thay thế tư vấn của dược sĩ/bác sĩ, không tự chẩn đoán hoặc kê đơn. Chức năng ghi nhận thanh toán chuyển khoản không đồng nghĩa tích hợp xác nhận giao dịch từ ngân hàng. Mức quyền chi tiết cần kiểm thử trên tài khoản thực tế trước khi đưa vào sử dụng.

## 3. Công nghệ và kiến trúc

| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| Frontend | React 19, Vite 6, JavaScript, CSS | Giao diện và tương tác người dùng |
| Backend | Python, FastAPI, Uvicorn | REST API và xử lý nghiệp vụ |
| ORM / dữ liệu | SQLAlchemy, PostgreSQL, Psycopg | Truy vấn và giao dịch CSDL |
| Xác thực | Mật khẩu băm, phiên/xác thực tại backend | Bảo vệ tài khoản và giới hạn truy cập |
| AI | Google Gemini API | Sinh câu trả lời từ nội dung/yêu cầu hệ thống |
| Local | Docker Compose (PostgreSQL) | Chạy CSDL tại máy lập trình |
| Deploy | Dockerfile nhiều giai đoạn, Render Web Service | Build React và chạy FastAPI trong cùng dịch vụ |
| Database production | PostgreSQL bên ngoài, ví dụ Neon | Lưu dữ liệu tách khỏi vòng đời dịch vụ Render |

```text
Trình duyệt
    |
    +--> Local: Vite :5173 --> FastAPI :8000 --> PostgreSQL Docker :5432
    |
    +--> Production: Render (FastAPI + React build)
                            |              |
                            v              v
                   PostgreSQL bên ngoài   Gemini API
```

Dockerfile hiện tại build React ở giai đoạn Node.js, sau đó chạy backend FastAPI và phục vụ frontend từ cùng origin. Vì vậy có thể triển khai **một** Render Web Service thay vì tạo hai website riêng.

## 4. Cấu trúc thư mục

```text
pharmacy-ai-gemini/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI endpoints
│   │   ├── db.py            # Kết nối PostgreSQL
│   │   ├── config.py        # Đọc cấu hình
│   │   ├── init_db.py       # Khởi tạo cơ sở dữ liệu / admin
│   │   ├── models.py        # Mô hình dữ liệu
│   │   ├── schemas.py       # Xác thực dữ liệu API
│   │   ├── services.py      # Xử lý nghiệp vụ
│   │   ├── auth.py          # Xác thực
│   │   └── ai.py            # Tích hợp Gemini
│   ├── requirements.txt
│   └── .env.example         # Ví dụ biến môi trường, không có secret thật
├── frontend/
│   ├── src/                 # Giao diện React
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
├── scripts/
│   ├── setup.py             # Khởi tạo file .env local
│   └── verify_gemini.py     # Công cụ kiểm tra Gemini
├── compose.yaml             # PostgreSQL 16 local, lưu bằng Docker volume
├── Dockerfile               # Build frontend + chạy backend
├── render.yaml              # Cấu hình deploy Render
└── README.md
```

> Danh sách trên rút gọn để dễ theo dõi. Một số tệp được sinh trong lúc chạy (`.env`, `.venv`, `node_modules`) không được đưa lên GitHub.

## 5. Yêu cầu môi trường

Khuyến nghị cho Windows 10/11:

- Git và VS Code (hoặc trình soạn thảo tương đương).
- **Python 3.12** để khớp image backend trong Dockerfile. Python 3.13 có thể chạy một số phần nhưng phụ thuộc vào mức tương thích của các gói trong `requirements.txt`.
- **Node.js 22+** cùng npm (Docker build dùng Node 22).
- Docker Desktop đã bật Docker Engine.
- Internet để tải thư viện, truy cập Gemini và deploy.

Kiểm tra phiên bản trên PowerShell:

```powershell
git --version
python --version
node -v
npm -v
docker --version
docker compose version
```

## 6. Cài đặt và chạy trên Windows

### 6.1. Lấy mã nguồn

Nếu repository đã chứa **mã nguồn An Tâm** trên nhánh `main`:

```powershell
git clone https://github.com/hatuan134/pharmacy-ai-gemini.git
cd 
```

Hoặc mở thư mục `pharmacy-ai-gemini` đã giải nén bằng VS Code. Không chạy lệnh setup trong thư mục dự án LIBRA cũ.

### 6.2. Tạo cấu hình local

Tại **thư mục gốc dự án**, chạy:

```powershell
python scripts/setup.py
```

Script tạo `.env` ở thư mục gốc (cho Docker Compose) và `backend/.env` (cho FastAPI) **nếu các file chưa tồn tại**. Nếu `backend/.env` đã có từ trước, script giữ nguyên; phải đối chiếu `DATABASE_URL` với PostgreSQL thực tế và tránh ghi đè bí mật đang dùng.

### 6.3. Khởi động PostgreSQL

Mở Docker Desktop, sau đó:

```powershell
docker compose up -d db
docker compose ps
```

Chờ dịch vụ `db` ở trạng thái `healthy`. Docker volume giữ dữ liệu local khi container khởi động lại. **Không chạy `docker compose down -v` nếu muốn giữ dữ liệu.**

### 6.4. Backend FastAPI

Mở terminal riêng tại thư mục gốc:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Kiểm tra kết nối dữ liệu trước khi khởi tạo:

```powershell
.\.venv\Scripts\python.exe -c "from app.db import engine; from sqlalchemy import text; print(engine.connect().execute(text('SELECT 1')).scalar())"
```

Nếu đây là **database mới hoặc đã sao lưu dữ liệu đang có**, mới chạy:

```powershell
.\.venv\Scripts\python.exe -m app.init_db
```

Bước này thiết lập bảng và tài khoản quản lý lần đầu; nếu được hỏi mật khẩu admin, hãy tự đặt và lưu an toàn.

Chạy server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

### 6.5. Frontend React

Mở terminal **khác** tại thư mục gốc:

```powershell
cd frontend
npm ci
npm run dev
```

Mở **http://localhost:5173** để dùng ứng dụng. API chạy tại **http://localhost:8000**; không đóng terminal backend/frontend khi đang sử dụng.

## 7. Cấu hình biến môi trường

**File `.env` ở thư mục gốc** được Docker Compose sử dụng:

```dotenv
POSTGRES_USER=pharmacy
POSTGRES_PASSWORD=YOUR_STRONG_LOCAL_PASSWORD
POSTGRES_DB=pharmacy
```

**File `backend/.env`** dùng khi chạy FastAPI trực tiếp:

```dotenv
DATABASE_URL=postgresql+psycopg://pharmacy:YOUR_STRONG_LOCAL_PASSWORD@localhost:5432/pharmacy
SECRET_KEY=REPLACE_WITH_A_RANDOM_SECRET_AT_LEAST_32_CHARACTERS
GEMINI_API_KEY=
GEMINI_MODEL=YOUR_SUPPORTED_GEMINI_MODEL
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
COOKIE_SECURE=false
```

Mật khẩu sau `DATABASE_URL` phải khớp mật khẩu thực tế của PostgreSQL. Nếu dùng Docker volume cũ, thay giá trị `POSTGRES_PASSWORD` trong `.env` **không tự đổi** mật khẩu của tài khoản đã được tạo trong CSDL.

## 8. Chạy và kiểm tra thủ công

### 8.1. Kiểm tra trạng thái dịch vụ

```powershell
# Ở thư mục gốc
docker compose ps

# Khi backend đã chạy
curl.exe -i http://localhost:8000/api/health

# Truy cập giao diện
# http://localhost:5173
```

API docs (nếu được bật): **http://localhost:8000/docs**.

### 8.2. Build và kiểm thử

```powershell
# Frontend
cd frontend
npm ci
npm test
npm run build
```

```powershell
# Backend: chạy ở thư mục backend, sau khi cài requirements
python -m pytest -q
```

Đây là các **lệnh đề xuất để tự chạy kiểm thử**, không phải cam kết mọi kiểm thử đều thành công trên mọi môi trường.

### 8.3. Danh sách kiểm tra nghiệp vụ

1. Đăng nhập tài khoản quản lý, dược sĩ và thu ngân; thử quyền truy cập theo từng vai trò.
2. Tra cứu và thêm/sửa thuốc; nhập lô, xem tồn và hạn sử dụng.
3. Bán hàng, tạo/in hóa đơn, kiểm tra tồn kho sau khi bán.
4. Kiểm tra báo cáo doanh thu và cảnh báo lô hết hạn.
5. Hỏi chatbot AI câu hỏi dữ liệu/quy trình, kiểm tra câu trả lời khi API key hoạt động.

