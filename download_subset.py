import os
from huggingface_hub import hf_hub_download

# Cấu hình đường dẫn lưu file zip
download_dir = "/home/tanht/EfficientLoFTR/data/megadepth"
os.makedirs(download_dir, exist_ok=True)

print("Đang tải tệp megadepth_test_1500.zip từ Hugging Face...")

# Tải trực tiếp file zip mà không cần qua git clone hay git-lfs
local_file = hf_hub_download(
    repo_id="shngjz/ce29d0e9486d476eb73163644b050222",
    filename="megadepth_test_1500.zip", # Hoặc tên file .tar/.zip chuẩn trong repo đó
    repo_type="dataset",
    local_dir=download_dir
)

print(f"Tải thành công! File được lưu tại: {local_file}")