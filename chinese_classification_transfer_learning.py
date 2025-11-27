import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import BertTokenizer, BertModel
from transformers.optimization import get_scheduler
from torch.optim import AdamW
from datasets import load_from_disk
import torch

# 加载编码器工具，编码器和模型是对应的
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
print(tokenizer)

## 试编码，观察输出
output = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=['明月几时有', '把就问青天'],
    truncation=True,
    padding='max_length',
    max_length=12,
    return_tensors='pt',
    return_length=True
)
for k, v in output.items():
    print(k, v.shape)
    print(v)
# 输出如下：
# input_ids torch.Size([2, 12])
# token_type_ids torch.Size([2, 12])
# length torch.Size([2])
# attention_mask torch.Size([2, 12])
# 把编码还原成句子：
print(tokenizer.decode(output['input_ids'][0]))


##

### 定义数据集

# 想用pytorch的数据集进行处理，pytorch的数据集需要继承torch下自己的数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        load_dataset = load_from_disk('./data/ChnSentiCorp_htl_all')
        load_dataset = load_dataset['train'].filter(lambda example: example['review'] is not None).train_test_split(test_size=0.3)  # 数据集中只有训练数据，分成训练数据集和测试数据集
        self.dataset = load_dataset[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['review']
        label = self.dataset[idx]['label']
        return text, label


dataset = Dataset('train')
print(dataset[0])

## 定义计算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


## 数据整理函数
def collate_fn(batch):
    sentences = [i[0] for i in batch]
    labels = [i[1] for i in batch]
    # 编码
    batch = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sentences,
                                        truncation=True,
                                        padding='max_length',
                                        max_length=512,
                                        return_tensors='pt',
                                        return_length=True)
    # input_ids:编码之后的数字
    # attention_mask:0的位置是不需要计算attention的，1的位置表示要计算attention
    # token_type_ids: token的类型，0表示第一个句子，1表示第二个句子
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    token_type_ids = batch['token_type_ids']
    labels = torch.LongTensor(labels)
    # 把数据拷贝到计算设备上
    # 这步操作也可以在训练的时候做
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    labels = labels.to(device)
    return input_ids, attention_mask, token_type_ids, labels


#### 测试下整理函数
# 先模拟下一批数据
data = [
    ('明月几时有', 1),
    ('把酒问青天', 0),
    ('不知天上宫阙', 1),
    ('今昔是何年', 0)
]
result = collate_fn(data)
print(result)
####

## 创建数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=16,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)  # 最后一批数据如果不够，就丢弃

len(loader)
## 查看数据样例
for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
    print(input_ids.shape)
    print(attention_mask.shape)
    print(token_type_ids.shape)
    print(labels.shape)
    break

# 加载预训练模型
pretrain = BertModel.from_pretrained('bert-base-chinese')

# 计算一下参数量:102267648
print(sum(i.numel() for i in pretrain.parameters()))

# 冻结参数
for param in pretrain.parameters():
    param.requires_grad_(False)  # 带下划线的方法是修改原本变量的值，设置为False，原始变量都不可以求导，即冻结参数

# 预训练模型试算
pretrain.to(device)

out = pretrain(input_ids=input_ids, attention_mask=attention_mask,
               token_type_ids=token_type_ids)  # 模型最终层的容器对象BaseModelOutputWithPoolingAndCrossAttentions
print(out.last_hidden_state.shape)  # BERT 模型最后一层输出的隐藏状态序列。它包含了输入序列中每一个 token 的高维、上下文相关的向量表示


# torch.Size([16, 512, 768]) BERT 模型最后一层输出格式


# 定义下游模型，用huggingface比较难写，用pytorch比较容易写
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=768, out_features=2)  # 下游任务模型，定义一层神经网络，输入为768，BERT模型的输出为512，二分类输出为2

    # 定义反向传播
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 使用预训练模型BERT抽取数据特征，然后用全连接网络进行计算
        with torch.no_grad():  # 原本模型不需要反向传播
            out = pretrain(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # 对抽取的特征只抽取第一个字的结果做分类。bert种第一个字是《cls》
        out = self.fc1(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out


model = Model()
# 拷贝到GPU上
model.to(device)

# 在训练前先进行试算
result = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
print(result.shape)  # 输出结果为 torch.Size([16, 2])


# 训练过程
def train():
    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=5e-4)  # lr 学习率，先定义小一点
    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 定义学习率调节器
    scheduler = get_scheduler(name='linear',
                              optimizer=optimizer,
                              num_warmup_steps=0,  # 预热一开始就调整
                              num_training_steps=len(loader))
    # 切换到训练模式
    model.train()
    # 按批次遍历训练集的数据
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
        # 模型计算
        out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = criterion(out, labels)  # 计算损失
        loss.backward()  # loss 做反向传播
        optimizer.step()  # 做更新
        scheduler.step()
        optimizer.zero_grad()  # 每个批次都要梯度清零，否则批次会累计

        if i % 10 == 0:
            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print(i, loss.item(), accuracy, lr)


train()  # 训练一次，如果需要训练多次，可以循环调用


# 测试过程
def test():
    loader_test = torch.utils.data.DataLoader(dataset=Dataset('test'),
                                              batch_size=32,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)
    # 下游模型切换到测试模式
    model.eval()
    correct = 0
    total = 0
    # 按批次遍历测试集种的数据
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_test):
        with torch.no_grad():
            out = model(input_ids, attention_mask, token_type_ids)
        # 计算准确率
        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)
    print("准确率：",correct / total)

test()