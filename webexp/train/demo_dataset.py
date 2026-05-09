import os
import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def demo_data_pipeline():
    # Bước 1 & 2: Tải Dataset và Lưu offline
    print("1. Đang tải dataset từ HuggingFace 'apurvaga/go-browse-wa'...")
    # Chú ý: Ở đây lấy split 'train' và lấy 50 dòng đầu tiên để demo cho nhẹ
    dataset = load_dataset("apurvaga/go-browse-wa", split="train[:50]")

    save_dir = "./data_local/go-browse-wa"
    os.makedirs(save_dir, exist_ok=True)

    print(f"2. Đang lưu dataset về thư mục {save_dir}...")
    dataset.save_to_disk(save_dir)
    print("Đã lưu xong!")

    # Bước 3: Load lại Dataset từ thư mục local để giả lập
    print("\n3. Đang nạp dataset offline...")
    local_dataset = Dataset.load_from_disk(save_dir)

    # Hiển thị cấu trúc vài cột của dòng đầu tiên
    print("\n--- CẤU TRÚC RAW DATA (Dòng 0) ---")
    sample_raw = local_dataset[0]
    print("Các keys có trong dữ liệu gốc:", sample_raw.keys())

    # Giả lập: Lọc các phần tử cần thiết như code train
    # SFT code lọc: x['traj_reward'] > 0 và map x['step_data'].
    # Nhưng nếu format 'apurvaga/go-browse-wa' có khác, ta mổ xẻ ở nhánh 'messages' hoặc 'prompt/completion'.
    # Vì bộ này có thể cấu trúc khác một tí, ta viết mock functions để giả lập theo code của bạn:

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    # CÁC HÀM XỬ LÝ (Lấy từ sft_policy.py của bạn)
    def flatten_messages(sample):
        # Thông thường prompt là List các dict (role, content) và completion cũng vậy
        # Ta tạm giả lập cho list messages
        if 'messages' in sample:
            return {'flattened': sample['messages']}

        prompt = sample.get('prompt', [])
        completion = sample.get('completion', [])
        return {'flattened': prompt + completion}

    def formatting_prompts_func(sample):
        # sample['flattened'] chứa 1 batch các cuộc hội thoại
        convos = sample['flattened']
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return { "text" : texts }

    # Demo convert thử 1 dòng (đảm bảo dataset có cột messages hoặc prompt/comp)
    print("\n4. Chạy thử các hàm xử lý dữ liệu (Flatten & Apply Chat Template)...")
    try:
        # Lọc theo điều kiện giống sft_policy.py
        filtered_dataset = local_dataset.filter(lambda x: x.get('traj_reward', 0) > 0).map(lambda x: x['step_data'])
        filtered_dataset = filtered_dataset.filter(lambda x: 'prompt' in x and 'completion' in x and x['prompt'] and x['completion'])

        # Làm phẳng JSON array
        flattened_dataset = filtered_dataset.map(flatten_messages, remove_columns=['prompt', 'completion'])
        # Apply chat template biến list json chat thành THẾ GIỚI CHỮ TEXT
        processed_dataset = flattened_dataset.map(formatting_prompts_func, batched=True)

        print("\n--- KẾT QUẢ CUỐI CÙNG FEED VÀO MODEL (Cột 'text' của Dòng 0) ---")
        print(processed_dataset[0]['text'][:3000]) # In 1500 ký tự đầu tiên
        print("\n... [VĂN BẢN BỊ CẮT VÌ QUÁ DÀI]")
    except Exception as e:
        print("\nCảnh báo: Có thể format raw dataset này không khớp với trường prompt/completion. Lỗi:", e)

if __name__ == "__main__":
    demo_data_pipeline()
