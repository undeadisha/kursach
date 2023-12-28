import torch
import torchvision as tv
import os
import pickle
from torchvision import transforms as T
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Загрузка словаря (vocab.pkl), который содержит отображение слов к их идентификаторам
with open('vocab.pkl', 'rb') as f:
    words = pickle.load(f)

# Параметры модели
feature_dim = 576
lstm_dim = 1024
embed_dim = 1024
attention_dim = 2048
num_hidden = 256
num_steps= 20
dict_length=len(words)
batch_size = 100

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_attn = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoder's output
        self.decoder_attn = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_attn = nn.Linear(attention_dim, 1)

    def forward(self, image_features, decoder_hidden):
        # image_features = mobilenet.features(image) # [1,576,7,7]
        # decoder_hidden = LSTM_hidden nn.LSTM(input_dim, output_dim) # [B, 1, output_dim]
        # Q, K, V  V=image_features, Q=decoder_hidden, K=image_features
        attn1 = self.encoder_attn(image_features)  # (batch_size, num_pixels, attention_dim)
        attn2 = self.decoder_attn(decoder_hidden)  # (batch_size, attention_dim)
        attn = self.full_attn(F.relu(attn1 + attn2.unsqueeze(1)))  # (batch_size, num_pixels, 1)

        # apply softmax to calculate weights for weighted encoding based on attention
        alpha = F.softmax(attn, dim=1)  # (batch_size, num_pixels, 1) num_pixels = 7 (width) * 7 (height)= 49
        attn_weighted_encoding = (image_features * alpha).sum(dim=1)  # (batch_size, encoder_dim)
        alpha = alpha.squeeze(2)  # (batch_size, num_pixels)
        return attn_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, dict_size, encoder_dim=2048, dropout=0.5):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = dict_size
        self.dropout = dropout

        self.embed = nn.Embedding(dict_size, embed_dim)  # embedding layer
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.fc = nn.Linear(decoder_dim, dict_size)  # linear layer to find scores over vocabulary

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.dropout = nn.Dropout(p=dropout)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)  # (batch_size, encoder_dim)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)  # (batch_size, decoder_dim)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lens):
        # encoder_out = mobilenet.features(image) # [1,7,7,576]

        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1) * encoder_out.size(2)
        encoder_dim = encoder_out.size(-1)
        # flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (1, 49, 576)
        num_pixels = encoder_out.size(1)

        # embedding
        embeddings = self.embed(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # initialize lstm state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        decode_lens = caption_lens.tolist()  # (caption_lens - 1).tolist()

        # create tensors to hold word prediction scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lens), self.fc.weight.size(0)).to(device)
        alphas = torch.zeros(batch_size, max(decode_lens), num_pixels).to(device)

        # decode_lens = [10,8,3], encoder_out[B(3),num_pixels (49), f_dim(576)]
        # When t = 0,1,2,3.... t == 3:
        # decode_lens = [10,8,3]

        # decode_lens = [10,8], encoder_out[B(2),num_pixels (49), f_dim(576)]
        for t in range(max(decode_lens)):  # max(decode_lens) = 20

            batch_size_t = sum([l > t for l in decode_lens])

            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])

            gate = torch.sigmoid(self.f_beta(h[:batch_size_t]))  # sigmoid gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding

            h, c = self.lstm_cell(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                # <word_t-1, image, h, c> -> word_t, c_t
                (h[:batch_size_t], c[:batch_size_t])
            )  # (batch_size_t, decoder_dim)

            # output - [B, 1, FI + FC]
            # get the next word prediction
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)

            # save the prediction and alpha for every time step
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lens, alphas