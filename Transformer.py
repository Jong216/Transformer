import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import copy


# -------------------- Transformer Encoder-Decoder --------------------
# 克隆
def clones(module, N):
    "Produce N identical layers"

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 生成结果序列
class Generator(nn.Module):
    def __init__(self, d_model, tgt_vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)  
    


# 词嵌入是为了将离散序列变为更有连续性
# 连续性可以体现在语义相近的词在向量空间中更接近
# 降维，减少计算复杂度

# 嵌入层
class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)  # 乘以权重


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=40):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 定义位置索引表
        # unsqueeze(idx)为在idx处增加维度，squeeze为减少维度
        position = torch.arange(0, max_len).unsqueeze(1)  # 每个索引为一个[1]的子张量

        # 定义pe表
        pe = torch.zeros(max_len, d_model)  # 每一个pos都有一个d_model维的向量

        # 分母分量计算
        denominators = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # 计算位置编码
        # 奇数位和偶数位只是sin和cos的区别

        pe[:, 0::2] = torch.sin(position * denominators)
        pe[:, 1::2] = torch.cos(position * denominators)

        pe = pe.unsqueeze(0)  # 给pe增加一个维度
        self.register_buffer("pe", pe)  # 为 pe 注册一个缓冲区，表示无需计算梯度的张量

    
    def forward(self, x):
        # 广播机制会自动统一维度
        # x为Embedding层的输出
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)  # 无需计算梯度
        return self.dropout(x)


# 多头注意力机制为多个自注意力机制的组合
def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):

    "Compute scaled dot-product attention"
    '''
    Args:
        query: (batch_size, num_heads, seq_len, d_k) 结构
    '''
    d_k = query.size(-1)  # 获取头的维度 d_k
    
    # attention_score
    # query通过keys计算与自身的相关性，通过attention矩阵在values中提取对应权重的信息（语义信息）
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 交换最后两维类似于转置

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # 掩码值为0时，表示不应被考虑
    p_attn = scores.softmax(dim=-1)  # 计算权重矩阵，带有注意力成分的，最后与value相乘
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


# 多头注意力机制
class MultiHeadAttention(nn.Module):
    "多头注意力机制"
    "多头注意力模块在于让每个注意力头注意各自的特征部分"
    "最终将每个注意力头的输出结果组合起来作为输出"
    def __init__(self, h, d_model, dropout=0.1):
        '''
        Args:
            h: 注意力头数量
            d_model: 隐藏层维度，词嵌入的输出维度
        '''
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0  # 确保输入维度可以被每个注意力头平分

        self.d_k = d_model // h  # 每个注意力头对应的维度
        self.h = h
        
        self.linears = clones(nn.Linear(d_model, d_model), 4)

        self.attn = None  # 自注意力计算值
        self.dropout = nn.Dropout(p=dropout)  # 创建dropout


    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"

        nbatches = query.size(0)
        # 1) 通过线性变化层将q k v 的维度变换为 h * d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) 
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) 计算attention值
        x, self.attn = scaled_dot_product_attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) contact
        x = (
            # 显示确保张量在内存中是连续的
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)  # -1表示一个占位符，表示-1维 == 自动计算剩余的维度大小
        )

        # 4) 最后一层Linear将维度转回d_model
        return self.linears[-1](x)


# FNN 前馈神经网络
# Linear + ReLU + Linear
class PositionwiseFeedforward(nn.Module):
    "Position-wise Feed-Forward Networks"
    # 升维+ReLU引入非线性 的目的为增加模型表达能力和捕捉更复杂的结构信息

    def __init__(self, d_model, d_ff, dropout=0.1):  
        super(PositionwiseFeedforward, self).__init__()
        '''
        Args:
            d_model: 输入输出的维度
            d_ff: 前馈神经网络的隐藏层维度
        '''
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        Arg:
            x: 句子序列
        '''
        return self.w2(self.dropout(self.w1(x).relu()))
     

# 归一化层
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))  # 控制缩放
        self.b_2 = nn.Parameter(torch.zeros(features))  # 用于偏移归一化后的特征
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # 保持维度不变
        std = x.std(-1, keepdim=True) # 求解标准差

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# 子层连接
class SublayerConnection(nn.Module):
    "A residual connection followed by a layer norm"
    "两个子层的输出必经之路"
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        "norm(x + subLayer(x))"
        return x + self.dropout(sublayer(self.norm(x)))
    

# 编码层
class EncoderLayer(nn.Module):
    "self-attn + feed forward"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.attn = self_attn
        self.feed_forward = feed_forward
        self.subConnections = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        output1 = self.subConnections[0](x, lambda x: self.attn(x, x, x, mask))  # 自注意力，输入同时作为q、k、v
        return self.subConnections[1](output1, self.feed_forward)


# 解码层
class DecoderLayer(nn.Module):
    "self_attn_masked + self_attn + feed forward"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()

        self.size = size
        self.src_attn = src_attn  # 关注编码器结果
        self.self_attn = self_attn  # masked
        self.feed_forward = feed_forward
        self.subConnections = clones(SublayerConnection(size, dropout), 3)

    def forward(self, memory, x, src_mask, tgt_mask):
        m = memory
        x = self.subConnections[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.subConnections[1](x, lambda x: self.src_attn(x, m, m, src_mask))

        return self.subConnections[2](x, self.feed_forward)


# 编码器
class Encoder(nn.Module):
    "N Encoderlayers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)


# 解码器
class Decoder(nn.Module):
    "N DecoderLayers"
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        
        # 加速收敛，防止梯度消失/爆炸
        return self.norm(x)  


class Transformer(nn.Module):
    "由编码器和解码器组成"
    def __init__(self, src_vocab_size, tgt_vocab_size, N, d_model, d_ff, h, dropout):
        '''
        Args:
            src_vocab_size: 源序列词典大小
            tgt_vocab_size: 目标序列词典大小
            N: 注意力头数目
            d_model: 模型维度
            d_ff: FFN的隐藏层数
        '''
        super(Transformer, self).__init__()
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        ff = PositionwiseFeedforward(d_model, d_ff)

        # Encoder
        encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
        self.encoder = Encoder(encoder_layer, N)

        # Decoder
        decoder_layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
        self.decoder = Decoder(decoder_layer, N)

        # src_embedding
        self.src_embed = nn.Sequential(
            Embeddings(src_vocab_size, d_model),
            c(PositionalEncoding(d_model, dropout))
        )

        # tgt_embedding
        self.tgt_embed = nn.Sequential(
            Embeddings(tgt_vocab_size, d_model),
            c(PositionalEncoding(d_model, dropout))
        )

        self.generator = Generator(d_model, tgt_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(tgt, memory, src_mask, tgt_mask)


def subsequent_mask(size):
    "Mask out subsequent positions."
    "[T F F]"
    "[T T F]"
    "[T T T]"
    "每一行只能关注T部分, F部分会被置为负无穷"
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


if __name__ == "__main__":
    src_vocab = 5000
    tgt_vocab = 5000
    N = 6
    h = 8
    d_model = 512
    d_ff = 2048
    dropout = 0.1
    batch_size = 64
    seq_len = 40

    # make_model
    model = Transformer(src_vocab, tgt_vocab, N, d_model, d_ff, h, dropout)

    print("Transformer architecture:")
    print(model)

    # pass
    src = torch.randint(0, src_vocab, (batch_size, seq_len))
    tgt = torch.randint(0, tgt_vocab, (batch_size, seq_len))

    src_mask = torch.ones(batch_size, 1, 1, seq_len)
    tgt_mask = subsequent_mask(seq_len).unsqueeze(0)

    output = model(src, tgt, src_mask, tgt_mask)
    print(f"output shape: {output.shape}")

    # generator
    result = model.generator(output)
    print(f"Final output shape: {result.shape}")
