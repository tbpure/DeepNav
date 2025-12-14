import os
import sys
from pathlib import Path


def get_current_project_folder() -> Path:
    # 判断操作系统
    if sys.platform == "darwin":
        print("当前系统是 Mac")
    elif sys.platform == "win32":
        print("当前系统是 Windows")
    else:
        print("当前系统是其他")

    # 获取当前 Python 项目的文件夹路径
    current_project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("当前项目的文件夹路径:", current_project_folder)
    return Path(current_project_folder)
