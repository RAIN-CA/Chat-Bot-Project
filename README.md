# 简单聊天机器人项目

这是一个基于 Python 和 PyTorch 构建的简单 Seq2Seq 聊天机器人示例项目。项目包含：

- **数据集**：`data/small_chat_dataset.json` – 一个简单的对话数据集。
- **模型**：使用 LSTM 构建的简单编码器-解码器结构。
- **训练脚本**：`scripts/train.py` – 加载数据、构建词汇表、训练模型，并保存模型权重到 `model/trained_model.pth`。
- **推理脚本**：`scripts/chat.py` – 加载训练好的模型，与用户进行交互。

## 使用方法

1. **安装依赖**

   ```bash
   pip install -r requirements.txt