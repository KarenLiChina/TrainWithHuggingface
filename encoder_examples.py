import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-chinese', cache_dir=None,
                                          force_download=False)  # 输入模型名字或者huggingface中模型的路径,模型下载后的缓存路径cache_dir NONE就是默认c盘，force_download 为false时，没有缓存才下载，true时，每次都下载
sentences = ['国破山河在',
             '城春草木深',
             '感时花溅泪',
             '恨别鸟惊心',
             '烽火连三月',
             '家书抵万金',
             '白头搔更短',
             '浑欲不胜簪']

out = tokenizer.encode(
    text=sentences[0],  # 指定地一个句子
    text_pair=sentences[1],  # bert需要一个句子对，用来处理两个相关的文本，而不是进行简单的拼接
    # 句子太长就阶段到max_length
    truncation=True,
    # 句子不够长就padding到max_length，用特殊符号补全
    padding='max_length',
    add_special_tokens=True,  # 句子中加特殊符号，表示句子开始结束
    max_length=25,
    return_tensors=None  # 返回列表
)
# 把数字转换成字符串
decode_out = tokenizer.decode(out)
print(decode_out)

# 进阶版本的编码函数
decode_out_plus = tokenizer.encode_plus(
    text=sentences[0],
    text_pair=sentences[1],
    truncation=True,  # 句子太长就截断
    padding='max_length',
    max_length=25,
    add_special_tokens=True,  # 句子中加特殊符号，表示句子开始结束,对应训练数据来说是要加的
    return_tensors=None,  # 返回的数据类型，默认返回列表，可以返回Tensorflow, pytorch 的tensor
    return_token_type_ids=True,  # 返回值是0或者1，我们传的是句子对，type id是是0 是第一个句子，type id是1，是第二个句子
    return_attention_mask=True,  # 特殊字符的标记，是特殊字符（包括起止和pad）的为1， 其他为0
    return_special_tokens_mask=True,  # 需要attention 的数值是1，不需要attention是 0
    return_length=True
)

for k, v in decode_out_plus.items():
    print(k, v)

# 批量编码
batch_out = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[(sentences[0], sentences[1]), (sentences[2], sentences[3])],
    # 参数可以是句子对，如果要对单句进行批量编码，就不需要括号
    truncation=True,  # 句子太长就截断
    padding='max_length',
    max_length=25,
    add_special_tokens=True,  # 句子中加特殊符号，表示句子开始结束,对应训练数据来说是要加的
    return_tensors=None,
    return_token_type_ids=True,  # 返回值是0或者1，我们传的是句子对，type id是是0 是第一个句子，type id是1，是第二个句子
    return_attention_mask=True,  # 特殊字符的标记，是特殊字符（包括起止和pad）的为1， 其他为0
    return_special_tokens_mask=True,  # 需要attention 的数值是1，不需要attention是 0
    return_length=True
)
for k, v in batch_out.items():
    print(k, ':', v)  # 句子对放到了列表中

# 字典的操作，也叫做词表

vocab = tokenizer.get_vocab()  # 获取预训练模型的字典
print(len(vocab))

# bert-base-chinese 是把每个中文当做一个词
# 增加新词
tokenizer.add_tokens(new_tokens=['山河', '草木'])

# 增加特殊字符
tokenizer.add_special_tokens({'eos_token': '[EOS]'})
for word in ['山河', '草木', '[EOS]']:
    print(tokenizer.get_vocab()[word])

new_out=tokenizer.encode(text='国破山河在[EOS]',
                 text_pair=None,
                 truncation=True,
                 padding='max_length',
                 add_special_tokens=True,
                 return_tensors=None,
                 max_length=10)
print(new_out)
print(tokenizer.decode(new_out)) # [CLS] 国 破 山河 在 [EOS] [SEP] [PAD] [PAD] [PAD] 此时 '山河' 被当作一个词
# 总结： 编码器的流程：定义字典，句子预处理，分词，编码