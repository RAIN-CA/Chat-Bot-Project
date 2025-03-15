#!/bin/bash
# 切换到脚本所在的目录（即项目根目录）
cd "$(dirname "$0")"

# 使用 -m 选项以模块方式运行训练脚本
python -m scripts.train