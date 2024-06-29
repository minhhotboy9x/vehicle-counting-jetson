# Jetson backend của vehicle-counting
Cài đặt:
```
pip install -r requirement.txt
```
Tải các file model từ `https://drive.google.com/drive/folders/1ZjUminHQG91UDiiSGROcy6WHpg3_PQAy?usp=sharing` vào trong thư mục `model/trt_jetson`

Chạy backend:
```
python src/app.py
```
Có thể chỉnh file mô hình trong `src/config.py`.