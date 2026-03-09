# person-reid-diffusion
基于扩散式数据增强的行人重识别方法研究与实现，包含代码、实验报告和结果分析

环境设置

基线模型（cluster-contrast-reid）：python3.9 原始链接：https://github.com/alibaba/cluster-contrast-reid

扩散模型（DCAC）：python3.10 原始链接：https://github.com/RikoLi/DCAC

GAN模型（DCGAN-tensorflow）：python3.7 原始链接：https://github.com/carpedm20/DCGAN-tensorflow?tab=readme-ov-file

CUDA 12.4以及PyTorch 1.8.0

其他需要的环境可以直接代码运行查看提示有什么需要的再安装

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

