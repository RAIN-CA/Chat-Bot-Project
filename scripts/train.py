import os
import json
import random
import torch
import torch.optim as optim
import torch.nn as nn

from src.model_arch import Vocabulary, EncoderRNN, DecoderRNN, tensor_from_sentence, SOS_token, EOS_token

# 单个训练样本的训练函数
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio=0.5):
    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # 编码过程
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([SOS_token])
    decoder_hidden = encoder_hidden

    loss = 0
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # 使用教师强制
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
            decoder_input = target_tensor[di].unsqueeze(0)
    else:
        # 自己预测下一个词
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().unsqueeze(0)
            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def main():
    # 加载数据集
    dataset_path = os.path.join("data", "small_chat_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 构建词汇表，并加载训练样本对
    vocab = Vocabulary()
    # 添加特殊符号
    vocab.add_word("SOS")
    vocab.add_word("EOS")
    pairs = []
    for pair in data:
        input_sentence = pair["input"]
        target_sentence = pair["response"]
        vocab.add_sentence(input_sentence)
        vocab.add_sentence(target_sentence)
        pairs.append((input_sentence, target_sentence))

    hidden_size = 256
    encoder = EncoderRNN(vocab.n_words, hidden_size)
    decoder = DecoderRNN(hidden_size, vocab.n_words)

    learning_rate = 0.01
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    n_iters = 1000
    print_every = 100

    for iter in range(1, n_iters + 1):
        pair = random.choice(pairs)
        input_tensor = tensor_from_sentence(vocab, pair[0])
        target_tensor = tensor_from_sentence(vocab, pair[1])
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        if iter % print_every == 0:
            print(f"迭代次数：{iter}，当前损失：{loss:.4f}")

    # 保存模型及词汇表
    os.makedirs("model", exist_ok=True)
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "vocab": vocab.__dict__
    }, os.path.join("model", "trained_model.pth"))
    print("模型已保存到 model/trained_model.pth")

if __name__ == '__main__':
    main()