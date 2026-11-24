# Ứng dụng phân tích và dự đoán giá đất

Ứng dụng Streamlit phục vụ báo cáo học thuật, sử dụng hai thuật toán:

- **Linear Regression:** dự đoán giá đất từ diện tích, chiều ngang, chiều dài, độ rộng hẻm, tọa độ, phường/xã, vị trí, hướng và loại đường.
- **K-Means Clustering:** phân cụm tin đăng dựa trên giá, diện tích và giá trên mỗi mét vuông.

## Cấu trúc thư mục

```text
Streamlit_GiaDat/
├── app.py
├── requirements.txt
├── README.md
└── data/
    ├── Nhom2_GiaDat.xlsx
    ├── Nhom2_GiaDat_train.xlsx
    └── Nhom2_GiaDat_test.xlsx
```

## Đưa ứng dụng lên mạng để chia sẻ với giảng viên

Ứng dụng được chuẩn bị để triển khai bằng **Streamlit Community Cloud**. Khi đã triển khai, người xem chỉ cần mở đường link; họ không cần cài Python hay tải các file Excel về máy.

### 1. Chuẩn bị GitHub

1. Đăng nhập hoặc tạo tài khoản GitHub.
2. Tạo một repository mới, ví dụ `streamlit-gia-dat`.
3. Chọn **Private** nếu dữ liệu là dữ liệu nội bộ của nhóm hoặc chưa được phép công khai.
4. Giải nén gói ứng dụng; tải lên GitHub toàn bộ nội dung bên trong thư mục `Streamlit_GiaDat_ChiaSe` (gồm `app.py`, `requirements.txt`, `README.md` và thư mục `data`). Không tải file ZIP nguyên khối thay cho các file này.
5. Xác nhận `app.py` và `requirements.txt` nằm ở thư mục gốc của repository; thư mục `data` nằm cạnh `app.py`.

### 2. Triển khai bằng Streamlit Community Cloud

1. Mở https://share.streamlit.io/ và đăng nhập bằng GitHub.
2. Chọn **Create app** rồi chọn repository vừa tạo.
3. Chọn branch `main` (hoặc branch bạn đã tải mã nguồn lên).
4. Ở **Main file path**, nhập `app.py`.
5. Bấm **Deploy**. Streamlit sẽ cài thư viện theo `requirements.txt` và khởi động ứng dụng.

### 3. Chia sẻ riêng cho giảng viên

1. Sau khi app chạy thành công, mở app và nhấn **Share**.
2. Nhập email của giảng viên và nhấn **Invite**.
3. Giảng viên sẽ nhận email mời và cần đăng nhập/xác thực để xem ứng dụng riêng tư.

**Lưu ý quyền riêng tư:** Repository Public sẽ làm lộ mã nguồn và các file dữ liệu đã tải lên. Hãy dùng repository Private và chỉ mời giảng viên xem nếu dữ liệu chỉ dành cho nhóm/giảng viên. Không tải lên mật khẩu, khóa API hoặc thông tin cá nhân nhạy cảm.

## Chạy trên máy cá nhân (không bắt buộc khi dùng link online)

Cần cài Python 3.10 trở lên, sau đó chạy:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Các chức năng

- **Tổng quan:** quy mô dữ liệu, giá/diện tích trung vị và phân bố giá.
- **Dự đoán giá đất:** nhập thông tin lô đất để dự đoán bằng Linear Regression.
- **Phân cụm K-Means:** chọn số cụm và xem hồ sơ từng cụm.
- **Đánh giá Linear Regression:** MAE, RMSE, R² trên tập test và biểu đồ thực tế/dự đoán.
- **Khám phá dữ liệu:** xem trước dữ liệu, tỷ lệ thiếu, thống kê mô tả và tải CSV.

## Ghi chú phương pháp

- `Price` trong file Excel có đơn vị triệu đồng; giao diện quy đổi sang tỷ đồng khi hiển thị.
- Linear Regression sử dụng file train/test riêng; ngưỡng lọc ngoại lệ được tính từ train và áp dụng cho test.
- Giá được biến đổi `log(1 + Price)` khi huấn luyện; các biến số được điền thiếu bằng trung vị và biến phân loại được mã hóa One-Hot.
- K-Means sử dụng log-transform và chuẩn hóa ba đặc trưng: giá, diện tích, giá/m².
- Đây là ứng dụng học thuật dựa trên dữ liệu tin đăng, không phải kết quả thẩm định hoặc cam kết giá giao dịch.
