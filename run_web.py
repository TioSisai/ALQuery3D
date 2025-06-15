#!/usr/bin/env python3
"""
ALQuery3D Web应用启动脚本
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 启动Web应用
from src.web.app import app

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ALQuery3D Web应用启动中...")
    print("=" * 60)
    print("📍 访问地址: http://localhost:5000")
    print("📍 或者: http://0.0.0.0:5000")
    print("=" * 60)
    print("💡 使用说明:")
    print("   1. 在浏览器中打开上述地址")
    print("   2. 选择类别数量（1-10个）")
    print("   3. 调节每个类别的参数")
    print("   4. 设置全局参数")
    print("   5. 点击生成按钮查看3D可视化")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True) 