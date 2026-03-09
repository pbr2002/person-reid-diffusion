# person-reid-diffusion
基于扩散式数据增强的行人重识别方法研究与实现  
包含基线模型、扩散模型（DCAC）、GAN模型（DCGAN）的完整代码、实验流程、结果分析

## 1. 项目概述
### 1.1 研究目标
对比基线模型（Cluster-Contrast-ReID）、扩散模型（DCAC）、GAN模型（DCGAN-tensorflow）在行人重识别任务中的性能，验证扩散式数据增强对模型精度的提升效果。

### 1.2 核心依赖与环境
| 模型/模块               | Python版本 | 核心框架       | 虚拟环境名称 | 其他关键依赖       |
|-------------------------|------------|----------------|--------------|--------------------|
| 基线模型（cluster-contrast-reid） | 3.9        | PyTorch 1.8.0  | pbr          | 详见对应文件夹requirements.txt |
| 扩散模型（DCAC）| 3.10       | PyTorch 1.8.0  | DCAC         | 详见对应文件夹requirements.txt |
| GAN模型（DCGAN-tensorflow）| 3.7    | TensorFlow     | DCGAN        | 运行代码，根据所缺依赖包进行安装 |
| 通用环境                | -          | CUDA 12.4      | -            | -                  |

### 1.3 参考开源项目
- Cluster-Contrast-ReID：https://github.com/alibaba/cluster-contrast-reid
- DCAC（扩散模型）：https://github.com/RikoLi/DCAC
- DCGAN-tensorflow：https://github.com/carpedm20/DCGAN-tensorflow?tab=readme-ov-file

## 2. 前置准备
### 2.1 预训练权重下载（扩散模型专用）
DCAC 模型依赖 `stable-diffusion-v1-5` 预训练权重，需手动下载并放置到指定路径：
```bash
# 创建pretrained文件夹并下载权重
cd ../DCAC
mkdir pretrained && cd pretrained
wget https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt
cd ..
```

### 2.2 依赖安装
各模型的依赖包已放在对应文件夹的 `requirements.txt` 中，按需安装：
```bash
# 示例：安装基线模型依赖和虚拟环境
conda create -n pbr python=3.9 -y
conda activate pbr
cd ../cluster-contrast-reid
pip install -r requirements.txt
cd ..
# 关闭当前虚拟环境
conda deactivate

# 安装扩散模型依赖和虚拟环境
conda create -n DCAC python=3.10 -y
conda activate DCAC
cd ../DCAC
pip install -r requirements.txt
cd ..
# 关闭当前虚拟环境
conda deactivate

# 安装GAN模型依赖和虚拟环境
conda create -n DCGAN python=3.7 -y
conda activate DCGAN
cd ../DCGAN-tensorflow
# 该项目无固定 requirements.txt
# 先安装基础依赖，若运行时报错，再根据提示补装缺失包
pip install tensorflow numpy scipy pillow matplotlib
cd ..
# 关闭当前虚拟环境
conda deactivate
```

## 2.3 路径修改（非常重要！！！）

**⚠️ 重要提示：**  
代码中存在硬编码的数据集路径（本程序中所有数据集路径均在`/mnt/data_hdd1/yangj/pbr`下，包括原始数据集路径`data`、扩散模型增强数据集路径`augmented_data`、扩散模型混合数据集路径`mixed_data`、GAN模型增强数据集路径`other_augmented_data`、GAN模型混合数据集路径`other_mixed_data`）。这些路径会直接影响数据集的读取、生成与保存，**运行前必须全部替换为你本地的实际路径**，否则程序会因找不到数据集或者生成的数据集无法正确保存而报错。

请按以下步骤检查并修改：

1. 全文件搜索上述路径关键词`/mnt/data_hdd1/yangj/pbr`；
2. 将所有匹配到的路径改为你的相应数据集实际存储路径；
3. 确认修改后保存代码。

## 3. 实验流程（Market-1501数据集为例）
### 3.1 基线模型测试（无数据增强）
```bash
# 1. 进入基线模型目录，激活对应虚拟环境
cd /home/yangj/cluster-contrast-reid
conda activate pbr

# 2. 运行训练脚本（4卡训练，Market-1501数据集，--data-dir需修改成自己的market501数据集路径）
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py \
  -b 256 -a resnet50 -d market1501 \
  --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 \
  --data-dir /mnt/data_hdd1/yangj/pbr/data > nohup.out 2>&1 &

# 3. 查看测试结果（结果保存在nohup.out）
cat nohup.out

# 4. 重命名结果文件（避免被覆盖）
mv nohup.out market1501_baseline.out

# 5. 关闭虚拟环境
conda deactivate
```

### 3.2 扩散模型数据增强+测试
```bash
# 步骤1：运行扩散模型生成增强数据集（DCAC环境）
cd /home/yangj/DCAC
conda activate DCAC

# 增强Market-1501数据集（ms_to_ma）；增强MSMT17则改为ma_to_ms
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python train_dcac.py \
  --config_file alchemycat_configs/single-source/ms_to_ma/cfg.py > output.log 2>&1 &

# 步骤2：混合原始数据集与增强数据集
python mix_market501.py  # MSMT17数据集则运行mix_msmt17.py

# 步骤3：基于混合数据集训练基线模型（pbr环境，--data-dir需修改成自己的混合后的market501数据集路径）
cd /home/yangj/cluster-contrast-reid
conda activate pbr

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py \
  -b 256 -a resnet50 -d market1501 \
  --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 \
  --data-dir /mnt/data_hdd1/yangj/pbr/mixed_data > nohup.out 2>&1 &

# 步骤4：查看并保存结果
cat nohup.out
mv nohup.out mixed_market1501_diffusion.out

# 5. 关闭虚拟环境
conda deactivate
```

### 3.3 GAN模型数据增强+测试
```bash
# 步骤1：运行GAN模型生成增强数据集（DCGAN环境）
cd /home/yangj/DCGAN-tensorflow
conda activate DCGAN

# 增强Market-1501数据集（需修改自己的data_dir和sample_dir）
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
  --dataset=bounding_box_train \
  --data_dir /mnt/data_hdd1/yangj/pbr/data/market1501/Market-1501-v15.09.15 \
  --train=True \
  --sample_dir /mnt/data_hdd1/yangj/pbr/other_augmented_data/augmented-market1501 \
  --crop=True > output.log 2>&1 &

# 步骤2：混合原始数据集与增强数据集
python mix_market501.py  # MSMT17数据集则运行mix_msmt17.py

# 步骤3：基于混合数据集训练基线模型（pbr环境，--data-dir需修改成自己的混合后的market501数据集路径）
cd /home/yangj/cluster-contrast-reid
conda activate pbr

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py \
  -b 256 -a resnet50 -d market1501 \
  --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 \
  --data-dir /mnt/data_hdd1/yangj/pbr/other_mixed_data > nohup.out 2>&1 &

# 步骤4：查看并保存结果
cat nohup.out
mv nohup.out mixed_market1501_gan.out

# 5. 关闭虚拟环境
conda deactivate
```

### 3.4 结果对比
对比以下文件中的测试指标，验证数据增强对模型精度的提升效果：
- 基线模型：`market1501_baseline.out`
- 扩散模型增强：`mixed_market1501_diffusion.out`
- GAN模型增强：`mixed_market1501_gan.out`

## 4. 注意事项
### 4.1 数据集清理（避免数据冗余）
若数据集体积过大，需清空以下文件夹后重新执行实验：
```
/mnt/data_hdd1/yangj/pbr/data/augmented-market1501
/mnt/data_hdd1/yangj/pbr/mixed_data/market1501/Market-1501-v15.09.15/bounding_box_train
/mnt/data_hdd1/yangj/pbr/mixed_data/msmt17/MSMT17_V1/bounding_box_train
/mnt/data_hdd1/yangj/pbr/data/augmented
```

### 4.2 虚拟环境切换
不同模型依赖不同Python版本，需严格激活对应虚拟环境：
- 基线模型：`conda activate pbr`
- 扩散模型：`conda activate DCAC`
- GAN模型：`conda activate DCGAN`

### 4.3 日志文件防覆盖
每次实验后务必重命名 `nohup.out`/`output.log`，避免后续实验覆盖历史结果。

## 5. 常见问题
### Q1：运行时提示“路径不存在”？
A1：检查代码中硬编码的路径是否已替换为本地实际路径，确认数据集文件夹已创建且包含对应文件。

### Q2：CUDA out of memory？
A2：降低批量大小（`-b` 参数，如从256改为128/64），或减少使用的GPU数量（如 `CUDA_VISIBLE_DEVICES=0,1`）。

### Q3：预训练权重下载失败？
A3：直接访问 HuggingFace 链接（https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt）手动下载，再上传到 `pretrained` 文件夹。

## 6. 实验结果（部分实验结果）
| 模型/数据集       | Market-1501 mAP | Market-1501 Rank-1 | Market-1501 Rank-5 |
|-------------------|-----------------|--------------------|------------|
| 基线模型          | 82.0               | 92.2                  | 96.6          |
| 扩散模型增强      | 83.4               | 93.2                  | 97.2          |
| GAN模型增强       | 82.2               | 92.3                  | 97.2          |
```

