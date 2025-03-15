import torch
import torch.nn as nn

# 定义特殊符号
SOS_token = 0
EOS_token = 1

# 词汇表类：负责词与索引之间的转换
class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.n_words = 0

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

# 将句子转换为 tensor（用于训练或推理）
def tensor_from_sentence(vocab, sentence):
    indexes = [vocab.word2index.get(word, 0) for word in sentence.split()] + [EOS_token]
    return torch.tensor(indexes, dtype=torch.long)

# 编码器：使用 LSTM 架构
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

# 解码器：同样使用 LSTM，并通过线性层输出词概率分布
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden