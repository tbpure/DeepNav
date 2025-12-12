import os

import numpy as np


def get_subfolders(parent_dir:str)->list[str]:
    """
    获取指定文件夹下的所有子文件夹（不递归）

    :param parent_dir: 父文件夹路径
    :return: 子文件夹路径列表
    """
    subfolders = [os.path.join(parent_dir, f)
                  for f in os.listdir(parent_dir)
                  if os.path.isdir(os.path.join(parent_dir, f))]
    return subfolders


def pad(arr, target_len):
    padded = np.full((target_len, arr.shape[1]), np.nan)
    padded[:len(arr)] = arr
    return padded
