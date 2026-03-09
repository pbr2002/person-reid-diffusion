from alchemy_cat.dl_config import Config

# default configs
cfg = Config(caps=('alchemycat_configs/base/model.yaml',
                   'alchemycat_configs/base/input.yaml',
                   'alchemycat_configs/base/test.yaml',
                   'alchemycat_configs/base/data/dataloader.yaml',
                   'alchemycat_configs/base/data/dataset.yaml',
                   'alchemycat_configs/base/solver/adam.yaml',
                   'alchemycat_configs/pcl/pcl.yaml',
                   'alchemycat_configs/stable_diffusion/stable_diffusion.yaml'))

cfg.MODEL.ID_LOSS_WEIGHT = 1.0
cfg.MODEL.PCL_LOSS_WEIGHT = 1.0
cfg.MODEL.DDPM_LOSS_WEIGHT = 1.0

cfg.SD.FINETUNE_MODE = 'lora'
cfg.SD.UNET.LORA_RANK = 32

cfg.INPUT.DIFFUSION_NOISE_AUG = True
cfg.DATASETS.NAMES = ('msmt17',)
cfg.DATASETS.EVAL_DATASET = 'market1501'
cfg.DATASETS.ROOT_DIR = '/mnt/data_hdd1/yangj/pbr/data'
cfg.DATALOADER.NUM_INSTANCE = 4  # 或其他合适的整数值
cfg.DATALOADER.NUM_WORKERS = 4  # 或其他合适的整数值
#用于保存增强后的数据集的路径
cfg.DATASETS.AUGMENTED_DATA_PATH = "/mnt/data_hdd1/yangj/pbr/augmented_data/augmented-market1501"