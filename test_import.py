import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 测试导入
try:
    from dataset import SST2Dataset
    print("dataset.py 导入成功")
except ImportError:
    print("dataset.py 导入失败！")

try:
    from models.transceiver import DeepSC
    print("models.transceiver 导入成功")
except ImportError:
    print("models.transceiver 导入失败！")

try:
    from utils import *
    print("utils.py 导入成功")
except ImportError:
    print("utils.py 导入失败！")