# person-reid-diffusion
基于扩散式数据增强的行人重识别方法研究与实现  
包含基线模型、扩散模型（DCAC）、GAN模型（DCGAN）的完整代码、实验流程、结果分析（本科毕业设计）

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
mkdir pretrained && cd pretrained
wget https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt
cd ..
```

### 2.2 依赖安装
各模型的依赖包已放在对应文件夹的 `requirements.txt` 中，按需安装：
```bash
# 示例：安装基线模型依赖
cd cluster-contrast-reid
pip install -r requirements.txt

# 安装扩散模型依赖
cd ../DCAC
pip install -r requirements.txt

# 安装GAN模型依赖（无requirements.txt文件。注：若运行时提示缺少依赖，直接根据报错信息安装即可。）

### 2.3 路径修改（关键！）
代码中硬编码的数据集路径（如 `/mnt/data_hdd1/yangj/pbr/data`）需替换为你的本地路径：
1. 全文件搜索上述路径关键词；
2. 将所有匹配到的路径改为你的数据集实际存储路径；
3. 确认修改后保存代码。

## 3. 实验流程（Market-1501数据集为例）
### 3.1 基线模型测试（无数据增强）
```bash
# 1. 进入基线模型目录，激活对应虚拟环境
cd /home/yangj/cluster-contrast-reid
conda activate pbr

# 2. 运行训练脚本（4卡训练，Market-1501数据集）
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py \
  -b 256 -a resnet50 -d market1501 \
  --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 \
  --data-dir /mnt/data_hdd1/yangj/pbr/data > nohup.out 2>&1 &

# 3. 查看测试结果（结果保存在nohup.out）
cat nohup.out

# 4. 重命名结果文件（避免被覆盖）
mv nohup.out market1501_baseline.out
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

# 步骤3：基于混合数据集训练基线模型（pbr环境）
cd /home/yangj/cluster-contrast-reid
conda activate pbr

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py \
  -b 256 -a resnet50 -d market1501 \
  --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 \
  --data-dir /mnt/data_hdd1/yangj/pbr/mixed_data > nohup.out 2>&1 &

# 步骤4：查看并保存结果
cat nohup.out
mv nohup.out mixed_market1501_diffusion.out
```

### 3.3 GAN模型数据增强+测试
```bash
# 步骤1：运行GAN模型生成增强数据集（DCGAN环境）
cd /home/yangj/DCGAN-tensorflow
conda activate DCGAN

# 增强Market-1501数据集（MSMT17需修改data_dir和sample_dir）
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
  --dataset=bounding_box_train \
  --data_dir /mnt/data_hdd1/yangj/pbr/data/market1501/Market-1501-v15.09.15 \
  --train=True \
  --sample_dir /mnt/data_hdd1/yangj/pbr/other_augmented_data/augmented-market1501 \
  --crop=True > output.log 2>&1 &

# 步骤2：混合原始数据集与增强数据集
python mix_market501.py  # MSMT17数据集则运行mix_msmt17.py

# 步骤3：基于混合数据集训练基线模型（pbr环境）
cd /home/yangj/cluster-contrast-reid
conda activate pbr

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py \
  -b 256 -a resnet50 -d market1501 \
  --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 \
  --data-dir /mnt/data_hdd1/yangj/pbr/other_mixed_data > nohup.out 2>&1 &

# 步骤4：查看并保存结果
cat nohup.out
mv nohup.out mixed_market1501_gan.out
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

## 6. 实验结果（毕设补充区）
| 模型/数据集       | Market-1501 mAP | Market-1501 Rank-1 | 精度提升率 |
|-------------------|-----------------|--------------------|------------|
| 基线模型          | -               | -                  | -          |
| 扩散模型增强      | -               | -                  | -          |
| GAN模型增强       | -               | -                  | -          |
> 注：请将实验得到的具体数值填入上表，用于结果分析与答辩展示。
```

### 关键说明
1. 文档全程采用纯 Markdown 语法编写，包含标题、表格、代码块、列表、引用等核心格式，可直接复制到 GitHub 仓库的 `README.md` 文件中使用；
2. 保留所有实验核心逻辑，补充了虚拟环境名称、格式优化、毕设专属模块，符合 GitHub 开源规范和毕设评审要求；
3. 代码块、表格、层级标题等格式在 GitHub 上会自动渲染，可读性强，使用者可直接复制命令运行。

### 使用方式
1. 将上述内容全选复制，替换你仓库中原有 `README.md` 的内容；
2. 补充“6. 实验结果”表格中的具体数值（根据你的实验数据填写）；
3. 提交并推送更新：
   ```bash
   git add README.md
   git commit -m "更新README为规范Markdown格式，补充虚拟环境名称"
   git push origin main
   ```

路径问题代码运行之后会报错，根据报错修改，或者直接全文件搜索文件路径如/mnt/data_hdd1/yangj/pbr/data，找到有这个地址的代码位置根据自己的地址进行修改


一、使用market1501数据集来运行行人重识别模型测试结果：

1.cd /home/yangj/cluster-contrast-reid,并进入pbr虚拟环境

2.运行下面python脚本，其中-d 后面填写的market1501是数据集名称，--data-dir /mnt/data_hdd1/yangj/pbr/data是数据集所在路径

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --data-dir /mnt/data_hdd1/yangj/pbr/data > nohup.out 2>&1 &

3.代码模型测试结果在nohup.out文件中，打开nohup.out文件查看测试结果

4.将nohup.out文件改名成market501.out，运行其他数据集时测试结果被覆盖

二、使用使用扩散模型增强后的数据集（mixed_market1501数据集）来运行行人重识别模型测试结果：

1.cd /home/yangj/DCAC，并进入DCAC虚拟环境

2.运行下面python脚本代码，其中如果要增强market501数据集则路径改为ms_to_ma，增强msmt17数据集则改为ma_to_ms

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python train_dcac.py --config_file alchemycat_configs
/single-source/ms_to_ma/cfg.py > output.log 2>&1 &

3.在/home/yangj/DCAC路径下，运行这个python脚本，把原有数据集与增强数据集混合,msmt17数据集就改为mix_msmt17.py
python mix_market501.py

4.cd /home/yangj/cluster-contrast-reid,并进入pbr虚拟环境

5.运行下面python脚本，--data-dir /mnt/data_hdd1/yangj/pbr/mixed_data把数据集路径进行修改

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --data-dir /mnt/data_hdd1/yangj/pbr/mixed_data > nohup.out 2>&1 &

6.代码模型测试结果在nohup.out文件中，打开nohup.out文件查看测试结果

7.将nohup.out文件改名成mixed_market501.out，运行其他数据集时测试结果被覆盖


三、使用使用GAN模型增强后的数据集（mixed_market1501数据集）来运行行人重识别模型测试结果：

1.cd /home/yangj/DCGAN-tensorflow，并进入DCGAN虚拟环境

2.运行下面python脚本代码，其中如果要增强market501数据集则路径改为--data_dir /mnt/data_hdd1/yangj/pbr/data/market1501/Market-1501-v15.09.15，增强数据集保存路径改为--sample_dir /mnt/data_hdd1/yangj/pbr/other_augmented_data/augmented-market1501，增强msmt17数据集则改为--data_dir /mnt/data_hdd1/yangj/pbr/data/msmt17/MSMT17_V1，增强数据集保存路径改为--sample_dir /mnt/data_hdd1/yangj/pbr/other_augmented_data/augmented-msmt17，


nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset=bounding_box_train --data_dir /mnt/data_hdd1/yangj/pbr/data/market1501/Market-1501-v15.09.15 --train=True --sample_dir /mnt/data_hdd1/yangj/pbr/other_augmented_data/augmented-market1501 --crop=True > output.log 2>&1 &

3.在/home/yangj/DCGAN-tensorflow路径下，运行这个python脚本，把原有数据集与增强数据集混合,msmt17数据集就改为mix_msmt17.py

python mix_market501.py

4.cd /home/yangj/cluster-contrast-reid,并进入pbr虚拟环境

5.运行下面python脚本，--data-dir /mnt/data_hdd1/yangj/pbr/other_mixed_data把数据集路径进行修改

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --data-dir /mnt/data_hdd1/yangj/pbr/other_mixed_data > nohup.out 2>&1 &

6.代码模型测试结果在nohup.out文件中，打开nohup.out文件查看测试结果

7.将nohup.out文件改名成other_mixed_market501.out，运行其他数据集时测试结果被覆盖

四、对比market501.out和mixed_market501.out的测试结果，检测模型精度是否有提升

注意：如果数据集过大，记得把四个文件夹下的图片清空，之后重新进行实验流程即可

/mnt/data_hdd1/yangj/pbr/data/augmented-market1501

/mnt/data_hdd1/yangj/pbr/mixed_data/ market1501/Market-1501-v15.09.15/bounding_box_train

/mnt/data_hdd1/yangj/pbr/mixed_data/ msmt17/MSMT17_V1/bounding_box_train

/mnt/data_hdd1/yangj/pbr/data/augmented_msmt17

