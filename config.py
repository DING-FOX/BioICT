# config.py

class Config:
    # 数据相关参数
    DATA_DIR = '/root/BioICT'
    SAMPLE_FRAC = 0.05  # 数据采样比例
    TRAIN_VAL_TEST_SPLIT = [0.7, 0.2, 0.1]  # 训练集、验证集、测试集比例
    MAX_SEQ_LENGTH = 512  # 序列最大长度
    RANDOM_SEED = 42  # 随机种子

    # 模型架构参数
    HIDDEN_DIM = 256  # Transformer隐藏层维度
    NUM_LAYERS = 3  # Transformer层数
    NUM_HEADS = 8  # 注意力头数
    DROPOUT = 0.1  # Dropout比例

    # 训练参数
    BATCH_SIZE = 128  # 批次大小
    NUM_EPOCHS = 10  # 训练轮数
    LEARNING_RATE = 0.001  # 学习率
    SCHEDULER_STEP_SIZE = 3  # 学习率调整步长
    SCHEDULER_GAMMA = 0.5  # 学习率调整因子
    GRAD_CLIP = 1.0  # 梯度裁剪阈值

    # 设备参数
    USE_CUDA = True
    CUDA_DEVICE = 0
    NUM_WORKERS = 4  # 数据加载线程数
    PIN_MEMORY = True  # 数据加载是否使用锁页内存

    # 输出和保存参数
    PRINT_FREQ = 10  # 打印频率（每多少个batch打印一次）
    SAVE_FREQ = 1  # 保存频率（每多少个epoch保存一次）
    MODEL_SAVE_PATH = 'protein_generator_best.pth'


# 加速训练的推荐参数配置
class FastConfig(Config):
    # 减小数据量
    SAMPLE_FRAC = 0.02

    # 简化模型结构
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    NUM_HEADS = 4

    # 增大批次大小以提高GPU利用率
    BATCH_SIZE = 256

    # 减少训练轮数
    NUM_EPOCHS = 5

    # 提高学习率以加快收敛
    LEARNING_RATE = 0.005

    # 优化数据加载
    NUM_WORKERS = 8
    PIN_MEMORY = True

    # 减少打印频率
    PRINT_FREQ = 20


# 对参数的详细说明
PARAMETER_DESCRIPTIONS = {
    'DATA_DIR': '数据目录路径',
    'SAMPLE_FRAC': '数据采样比例，减小此值可以加快训练速度但可能影响模型效果',
    'TRAIN_VAL_TEST_SPLIT': '数据集划分比例',
    'MAX_SEQ_LENGTH': '序列的最大长度，超过此长度的序列将被截断',
    'HIDDEN_DIM': 'Transformer模型的隐藏层维度，影响模型容量和训练速度',
    'NUM_LAYERS': 'Transformer编码器和解码器的层数，更多层意味着更强的模型能力但训练更慢',
    'NUM_HEADS': '多头注意力机制中的头数，通常是8个',
    'DROPOUT': 'Dropout比例，用于防止过拟合',
    'BATCH_SIZE': '训练批次大小，更大的批次可以提高训练速度但需要更多显存',
    'NUM_EPOCHS': '训练轮数',
    'LEARNING_RATE': '学习率，更大的学习率可能加快收敛但也可能导致不稳定',
    'SCHEDULER_STEP_SIZE': '学习率调整的步长',
    'SCHEDULER_GAMMA': '学习率调整的缩放因子',
    'GRAD_CLIP': '梯度裁剪阈值，用于防止梯度爆炸',
    'NUM_WORKERS': '数据加载的工作进程数，增加此值可能提高数据加载速度',
    'PIN_MEMORY': '是否使用锁页内存，对GPU训练有加速效果',
    'PRINT_FREQ': '打印训练状态的频率'
}


def save_parameter_descriptions():
    """保存参数说明到文件"""
    with open('parameter_descriptions.txt', 'w', encoding='utf-8') as f:
        f.write("参数配置说明：\n")
        f.write("=" * 50 + "\n\n")
        for param, desc in PARAMETER_DESCRIPTIONS.items():
            f.write(f"{param}:\n    {desc}\n\n")

        f.write("\n推荐的加速训练参数配置：\n")
        f.write("=" * 50 + "\n\n")
        for attr in dir(FastConfig):
            if not attr.startswith('__'):
                value = getattr(FastConfig, attr)
                if attr in PARAMETER_DESCRIPTIONS:
                    f.write(f"{attr} = {value}\n    {PARAMETER_DESCRIPTIONS[attr]}\n\n")


