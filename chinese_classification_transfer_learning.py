import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import BertTokenizer
from datasets import load_from_disk

# 加载编码器工具，编码器和模型是对应的
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
print(tokenizer)
dataset = load_from_disk('./data/ChnSentiCorp_htl_all')

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
