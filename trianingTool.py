import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import evaluate

tokenizer = AutoTokenizer.from_pretrained('hfl/rbt3')  # huggingface上的模型都可以加载

dataset = load_from_disk('./data/ChnSentiCorp_htl_all')

# 缩小数据规模, 便于测试.
print(dataset)
dataset = dataset['train'].train_test_split(test_size=0.3)  # 数据集中只有训练数据，分成训练数据集和测试数据集
dataset['train'] = dataset['train'].shuffle().select(range(2000))
dataset['test'] = dataset['test'].shuffle().select(range(600))


# 将中文编码转换为数字编码
def f_encoder(data, tokenizer):
    return tokenizer.batch_encode_plus(data['review'], truncation=True)  # batch_encode_plus会返回input_ids结果


# filter掉空数据后再用map进行编码
dataset = dataset.filter(lambda example: example['review'] is not None).map(f_encoder, batched=True,
                                                                            batch_size=1000,
                                                                            remove_columns=['review'],
                                                                            fn_kwargs={'tokenizer': tokenizer})

print(dataset['train'][0])


# # 删掉太长的句子，原模型预训练时，最长的句子就是512
def f_remove_long(data):
    return [len(i) <= 512 for i in data['input_ids']]


dataset = dataset.filter(f_remove_long, batched=True, batch_size=1000)
# 加载模型，可以根据任务类型加载模型，如果不清楚模型类型，也可以直接用AutoModel.***，但是需要更多的配置
model = AutoModelForSequenceClassification.from_pretrained('hfl/rbt3', num_labels=2)  # 基于BERT的中文预训练模型，二分类
# 统计模型参数量
print(sum([i.nelement() for i in model.parameters()]))

### 模型试算
# 一条模拟数据
data = {
    'input_ids': torch.ones(4, 10, dtype=torch.long),  # 每次4个样本，每个样本10个tokens
    'token_type_ids': torch.ones(4, 10, dtype=torch.long),
    'attention_mask': torch.ones(4, 10, dtype=torch.long),
    'labels': torch.ones(4, dtype=torch.long)
}

out = model(**data)  # **data 一个字典传过来，加了** 相当于把字典解包
## 模型的返回值预测类别的得分
print(out)
print(out['loss'])
print(out['logits'])

#####
metric = evaluate.load('accuracy')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = logits.argmax(axis=1)  # 取得分最高的为预测结果
    ## 不用metric,也可以手算 下面两种方式等价
    # return {'accuracy': (pred == labels).mean().item()}
    return metric.compute(predictions=pred, references=labels)


# 定义训练参数,
args = TrainingArguments(
    output_dir='./output',  # 定义临时数据代表保存路径
    eval_strategy='steps',
    # 旧版本的名字是evaluation_strategy定义测试执行的策略，可选取no 不保存，epoch 每个epoch保存一次，steps 每隔eval_steps次step保存一次
    eval_steps=30,
    save_strategy='steps',  # 定义模型保存策略，no不保存，epoch 每个epoch保存一次，steps 每隔eval_steps次step保存一次
    save_steps=30,
    num_train_epochs=1,  # 定义藏歌训练集的轮次
    learning_rate=1e-4,  # 定义学习率
    weight_decay=1e-2,  # 加入参数权重衰减，防止过拟合
    # 定义训练和测试时候的批次大小
    per_device_eval_batch_size=16,
    per_device_train_batch_size=16,
    no_cuda=False  # 定义是否使用GPU
)

# 定义训练器
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)  # 数据整理函数，把长短不一的数据整理成同样长度
)

### 测试数据整理函数

data_collater = DataCollatorWithPadding(tokenizer=tokenizer)
# 获取一批数据
data = dataset['train'][:5]
print(data)  # 此时每个句子的长度不同
for i in data['input_ids']:
    print(len(i))
data = data_collater(data)  # 调用数据整理函数进行整理，给不够长的数据补0
for i in data['input_ids']:
    print(len(i))
#####

# 训练和测试

# 训练前，看下测试结果
result = trainer.evaluate()
print(result)
# 输出如下：
# {'eval_loss': 0.9054994583129883, 'eval_model_preparation_time': 0.0013, 'eval_accuracy': 0.3281786941580756, 'eval_runtime': 2.0323, 'eval_samples_per_second': 286.378, 'eval_steps_per_second': 18.206}

# 进行训练
trainer.train()

result = trainer.evaluate()
print(result)
# 训练好的输出如下：
# {'eval_loss': 0.2847978174686432, 'eval_model_preparation_time': 0.0014, 'eval_accuracy': 0.8811544991511036, 'eval_runtime': 1.6172, 'eval_samples_per_second': 364.203, 'eval_steps_per_second': 22.879, 'epoch': 1.0}

# 保存模型，在output中已经自动保存了checkpoint阶段的模型参数

trainer.save_model(output_dir='./output_model')  # 可以手动保存模型到指定目录，旧版本会生成pytorch_model.bin， 新版本生成model.safetensors
# 旧版本生成pytorch_model.bin文件可以用load_state_dict 去load 模型
# model.load_state_dict(torch.load('./output_model/pytorch_model.bin'))
# 新版本生成model.safetensors文件，需要用AutoModelForSequenceClassification 去加载
new_model = AutoModelForSequenceClassification.from_pretrained('./output_model')
print(new_model)
# 如果训练一半被中断（停电/断网）
trainer.train(resume_from_checkpoint='./output/checkpoint-90')

## 训练模型的预测

model.eval()
# 取一批数据做预测
for i, data in enumerate(trainer.get_eval_dataloader()):
    break;
for k, v in data.items():
    # 模型在GPU上训练，需要把数据拷贝到GPU上
    data[k] = v.to('cuda')

# 预测
out = model(**data)
pred = out['logits'].argmax(dim=1)
print(pred)
for i in range(16):
    print(tokenizer.decode(data['input_ids'][i], skip_special_tokens=True))
    print('label=',data['labels'][i].item())
    print('predict=',pred[i].item())

