# Jetson backend của vehicle-counting
Cài đặt:
```
pip install -r requirement.txt
```
Tạo đường dẫn `model/trt_jetson`, tải các file model từ `https://drive.google.com/drive/folders/1ZjUminHQG91UDiiSGROcy6WHpg3_PQAy?usp=sharing` vào thư mục này. 

Tạo đường dẫn `imgs`, tải các file video từ `https://drive.google.com/open?id=1bnvPPvRSjC6ebiJrqp1i3EGBclgWxcMb&usp=drive_fs` vào thư mục này.

Chạy backend:
```
python src/app.py
```
Có thể chỉnh file mô hình trong `src/config.py`.