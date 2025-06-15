#!/usr/bin/env python3
"""
ALQuery3D GUI应用启动脚本
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 启动GUI应用
from src.gui.main_window import main

if __name__ == "__main__":
    main() 