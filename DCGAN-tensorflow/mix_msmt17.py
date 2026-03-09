import os
import shutil
import random
from PIL import Image

def mix_datasets(original_folder, new_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取原始数据集和新数据集中的所有文件
    original_files = [os.path.join(original_folder, f) for f in os.listdir(original_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    new_files = [os.path.join(new_folder, f) for f in os.listdir(new_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 合并两个列表
    combined_files = original_files + new_files

    # 打乱顺序
    random.shuffle(combined_files)

    # 将混合后的数据复制到输出文件夹
    for idx, file_path in enumerate(combined_files):
        # 获取文件名
        file_name = os.path.basename(file_path)
        # 构造新的文件路径
        new_file_path = os.path.join(output_folder, f"{idx}_{file_name}")
        # 复制文件
        shutil.copy(file_path, new_file_path)

    print(f"混合后的数据已保存到 {output_folder}")

# 调用函数
original_folder = "/mnt/data_hdd1/yangj/pbr/data/msmt17/MSMT17_V1/bounding_box_train"
new_folder = "/mnt/data_hdd1/yangj/pbr/other_augmented_data/augmented-msmt17"
output_folder = "/mnt/data_hdd1/yangj/pbr/other_mixed_data/msmt17/MSMT17_V1/bounding_box_train"
mix_datasets(original_folder, new_folder, output_folder)
