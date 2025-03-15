import os
import torch
from src.model_arch import Vocabulary, EncoderRNN, DecoderRNN, tensor_from_sentence, SOS_token, EOS_token

# 评估函数：使用贪心搜索生成回复
def evaluate(encoder, decoder, vocab, sentence, max_length=10):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(vocab, sentence)
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.init_hidden()

        # 编码过程
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([SOS_token])
        decoder_hidden = encoder_hidden

        decoded_words = []
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(vocab.index2word.get(topi.item(), "<unk>"))
            decoder_input = topi.squeeze().detach().unsqueeze(0)
        return " ".join(decoded_words)

def main():
    # 加载模型和词汇表
    checkpoint = torch.load(os.path.join("model", "trained_model.pth"), map_location=torch.device('cpu'))
    vocab_data = checkpoint["vocab"]
    vocab = Vocabulary()
    vocab.__dict__.update(vocab_data)
    hidden_size = 256
    encoder = EncoderRNN(vocab.n_words, hidden_size)
    decoder = DecoderRNN(hidden_size, vocab.n_words)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    encoder.eval()
    decoder.eval()

    print("欢迎使用聊天机器人，输入 'quit' 退出对话")
    while True:
        sentence = input("你: ")
        if sentence.lower() == "quit":
            break
        output_sentence = evaluate(encoder, decoder, vocab, sentence)
        print("机器人:", output_sentence)

if __name__ == '__main__':
    main()