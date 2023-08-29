import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


             # Encoder_input    Decoder_input (训练的时候需要，看看那个流程图就知道了)      Decoder_output
sentences = [['我 是 学 生 P' , 'S I am a student'   , 'I am a student E'],         # S: 开始符号
             ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],      # E: 结束符号
             ['我 是 男 生 P' , 'S I am a boy'       , 'I am a boy E']]             # P: 占位符号，如果当前句子不足固定长度用P占位

src_vocab = {'P':0, '我':1, '是':2, '学':3, '生':4, '喜':5, '欢':6,'习':7,'男':8}   # 词源字典  字：索引
src_idx2word = {src_vocab[key]: key for key in src_vocab} #键映射到值，把语言映射到索引，构成了语言之间的对应
src_vocab_size = len(src_vocab)                                                     # 字典字的个数
tgt_vocab = {'P':0, 'S':1, 'E':2, 'I':3, 'am':4, 'a':5, 'student':6, 'like':7, 'learning':8, 'boy':9}
idx2word = {tgt_vocab[key]: key for key in tgt_vocab}                               # 把目标字典转换成 索引：字的形式
tgt_vocab_size = len(tgt_vocab)                                                     # 目标字典尺寸
src_len = len(sentences[0][0].split(" "))                                           # Encoder输入的最大长度
tgt_len = len(sentences[0][1].split(" "))                                           # Decoder输入输出最大长度

#把sentences转换成成字典的形式
def pre_data(sentences):
    enc_inputs,dec_inputs,dec_outputs=[],[],[]
    for i in range (len(sentences)):
        enc_input=[[src_vocab[n] for n in sentences[i][0].split()]]
        dec_input=[[tgt_vocab[n] for n in sentences[i][1].split()]]
        dec_output=[[tgt_vocab[n] for n in sentences[i][2].split()]]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    return torch.tensor(enc_inputs,dtype=torch.long),torch.tensor(dec_inputs,dtype=torch.long),torch.tensor(dec_outputs,dtype=torch.long)

enc_inputs,dec_inputs,dec_outputs=pre_data(sentences)

#为什么还是要定义数据集呢，因为dataset、dataloader再到model就是一个完整的流程，他里面有自己定义的封装所以要按它的格式来
class mydataset(Data.Dataset):
    def __init__(self,enc_inputs,dec_inputs,dec_outputs):
        super(mydataset,self).__init__()
        self.enc_inputs=enc_inputs
        self.dec_inputs=dec_inputs
        self.dec_outputs=dec_outputs

    def __len__(self):
         return self.enc_inputs.shape[0] #模型只需要知道输入的长度，这样才可以把不同长度的句子进行padding从而训练。这里的enc是因为数据集很小所以自己知道enc就行
    
    def __getitem__(self,idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx] #这里不重写也行

dataset=mydataset(enc_inputs,dec_inputs,dec_outputs)

loader=Data.DataLoader(dataset,2,True)

#hyper parameters
d_model = 512   # 字 Embedding 的维度
d_ff = 2048     # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度 
n_layers = 6    # 有多少个encoder和decoder
n_heads = 8     # Multi-Head Attention设置为8

#positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout) 
        pos_table = np.array([
        [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
        if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])                  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])                  # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table)               # enc_inputs: [seq_len, d_model]

    def forward(self, enc_inputs):                                         # enc_inputs: [batch_size, seq_len, d_model]
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs)
    
#mask掉没有意义的占位符
def get_attn_pad_mask(seq_q,seq_k): #这里的kq都是已经进行和数据进行相乘以后得到的序列，而非只有参数的矩阵
    batch_size,len_q=seq_q.size()
    batch_size,len_k=seq_k.size()
    pad_attn_mask=seq_k.data.eq(0).unsqueeze(1) #eq是用来逐元素比较张量是否相等，然后返回一个bool
    return pad_attn_mask.expand(batch_size,len_q,len_k) #把它扩展成这样才可以和v进行相乘，其中的0相乘之后就可以降低无关量的注意力效果

#decoder的mask
def get_attn_subsequence_mask(seq):
    attn_shape=[seq.size(0),seq.size(1),seq.size(1)]
    subsequence_mask=np.triu(np.ones(attn_shape),k=1) #k表示主对角线之上的第几条对角线
    subsequence_mask=torch.from_numpy(subsequence_mask).byte() #因为bool只有一个字节所以用byte省空间
    return subsequence_mask
#这不是一种按时间序列的一步一步遮罩，而是已经知道了整个seq，所以用上三角一步到位

#敲完这个走人
class ScaledDotProduceAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProduceAttention,self).__init__()

    def forward(self,Q,K,V,attn_mask): #这里的Q，K是已经和数据乘完的
        scores=torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(d_k)
        scores.masked_fill(attn_mask,-1e9) 
        attn=nn.Softmax(dim=-1)(scores) #表示在最后一个维度上进行操作
        context=torch.matmul(attn,V)
        return context,attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention,self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads,bias=False) #原来进行的是全连接层的操作
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self,input_Q,input_K,input_V,attn_mask):
        residual,batch_size=input_Q,input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size,-1,n_heads,d_k).transpose(1,2) #把1 2 维度数据换一下
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)              # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        context, attn = ScaledDotProduceAttention()(Q, K, V, attn_mask)          # context: [batch_size, n_heads, len_q, d_v]
                                                                                 # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)                                                # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model)(output + residual), attn
    #seq_len 和 seq_v长度一般相等，我真服了
    #到底是谁想出来的，这么多维度转换我真服了

#前馈神经网络
class FeedForwardNet(nn.Module):
    def __init__(self):
        super(FeedForwardNet,self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(d_model,d_ff,bias=False), #转向更高维度，允许模型学习更复杂的特征映射
            nn.ReLU(),
            nn.Linear(d_ff,d_model,bias=False)
        )

    def forward(self,inputs):
        residual=inputs
        output=self.fc(inputs)
        return nn.LayerNorm(d_model)(output+residual) #一个句子上进行的归一化，batchnorm 每句话的相同位置归一化
    
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer,self).__init__()
        self.enc_attn=MultiHeadAttention()
        self.pos_ffn=FeedForwardNet()

    def forward(self,enc_inputs,enc_self_attn_mask):
        enc_outputs,attn=self.enc_attn(enc_inputs,enc_inputs,enc_inputs,enc_self_attn_mask)
        enc_outputs=self.pos_ffn(enc_outputs)
        return enc_outputs,attn
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.src_emb=nn.Embedding(src_vocab_size,d_model) #用于将离散的整数标识映射到连续的向量空间中
        self.pos_emb=PositionalEncoding(d_model)
        self.layers=nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
    
    def forward(self,enc_inputs):
        enc_outputs=self.src_emb(enc_inputs)
        enc_outputs=self.pos_emb(enc_outputs)
        print(enc_inputs.size())
        enc_self_attn_mask=get_attn_pad_mask(enc_inputs,enc_inputs)
        enc_self_attns=[]
        for layer in self.layers:
            enc_outputs,enc_self_attn=layer(enc_outputs,enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs,enc_self_attns

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer,self).__init__()
        self.dec_self_attn=MultiHeadAttention()
        self.dec_enc_attn=MultiHeadAttention()
        self.pos_ffn=FeedForwardNet()

    def forward(self,dec_inputs,enc_outputs,dec_self_attn_mask,dec_enc_attn_mask):
        dec_outputs,dec_self_attn=self.dec_self_attn(dec_inputs,dec_inputs,dec_inputs,dec_self_attn_mask)
        dec_outputs,dec_enc_attn=self.dec_enc_attn(dec_outputs,enc_outputs,enc_outputs,dec_enc_attn_mask)
        dec_outputs=self.pos_ffn(dec_outputs)  #跟着流程图来看还蛮好理解的
        return dec_outputs,dec_self_attn,dec_enc_attn
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.tgt_emb=nn.Embedding(tgt_vocab_size,d_model)
        self.pos_emb=PositionalEncoding(d_model)
        self.layers=nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self,dec_inputs,enc_inputs,enc_outputs):
        dec_outputs=self.tgt_emb(dec_inputs)
        dec_outputs=self.pos_emb(dec_outputs)
        dec_self_attn_pad_mask=get_attn_pad_mask(dec_inputs,dec_inputs)
        dec_self_attn_subsequence_mask=get_attn_subsequence_mask(dec_inputs)
        dec_self_attn_mask=torch.gt((dec_self_attn_pad_mask+dec_self_attn_subsequence_mask),0) #gt用于逐元素比较大小
        dec_enc_attn_mask=get_attn_pad_mask(dec_inputs,enc_inputs)
        dec_self_attns,dec_enc_attns=[],[]

        for layer in self.layers:
            dec_outputs,dec_self_attn,dec_enc_attn=layer(dec_outputs,enc_outputs,dec_self_attn_mask,dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs,dec_self_attns,dec_enc_attns
    
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer,self).__init__()
        self.Encoder=Encoder()
        self.Decoder=Decoder()
        self.projection=nn.Linear(d_model,tgt_vocab_size,bias=False)

    def forward(self,enc_inputs,dec_inputs):
        enc_outputs,enc_self_attns=self.Encoder(enc_inputs)
        dec_outputs,dec_self_attns,dec_enc_attns=self.Decoder(dec_inputs,enc_inputs,enc_outputs)
        dec_logits=self.projection(dec_outputs)
        return dec_logits.view(-1,dec_logits.size(-1)),enc_self_attns,dec_self_attns,dec_enc_attns
    

model=Transformer()
criterion=nn.CrossEntropyLoss(ignore_index=0) #忽略占位符
optimizer=optim.SGD(model.parameters(),lr=1e-3,momentum=0.99)

for epoch in range(50):
    for enc_inputs,dec_inputs,dec_outputs in loader:
        #enc_inputs=enc_inputs.squeeze()
        outputs,enc_self_attns,dec_self_attns,dec_enc_attns=model(enc_inputs, dec_inputs)

        loss=criterion(outputs,dec_outputs.view(-1))
        print('Epoch:','%04f'%(epoch+1),'loss=','{:.6f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
