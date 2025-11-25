import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset, load_from_disk

# https://hf-mirror.com/datasets/dirtycomputer/ChnSentiCorp_htl_all
# 输入huggingface的路径

dataset_path = './data/ChnSentiCorp_htl_all'

if not os.path.exists(dataset_path):
    dataset = load_dataset('dirtycomputer/ChnSentiCorp_htl_all')
    dataset.save_to_disk(dataset_path)  # 把数据保存到本地。
else:
    dataset = load_from_disk(dataset_path)  # 也可以从本地加载数据

print(dataset)

# 数据集的基本操作
train_dataset = dataset['train']  # 获取训练数据集

# 查看数据内容
for i in range(10):
    print(train_dataset[i])

# 获取label的信息
print(train_dataset['label'][:10])

# 根据label 排序，不会改变原有数据集，返回新的结构
sorted_dataset = train_dataset.sort('label')
print(sorted_dataset['label'][:10])
print(sorted_dataset['label'][-10:])

shuffle_data = sorted_dataset.shuffle(seed=10)  # 重新打乱shuffle 洗牌\
print(shuffle_data['label'][:10])

# 数据抽样，可以实现数据抽样
train_dataset.select([0, 10, 20, 30, 40, 50])  # 抽取第0，10，20，30，40，50的数据
# 选择连续的索引范围
print(train_dataset.select(range(100, 200)))  # 抽取第100-199的数据


# 数据过滤
def f(data):
    print(data)
    print(data['review'])
    return data['review'] is not None and data['review'].startswith('非常好')


print(train_dataset.filter(f))  # 用方法f中的过滤条件去过滤

# 训练测试集的划分
split_result = train_dataset.train_test_split(test_size=0.1)  # 设置测试集比例，训练集90%，测试集10%，返回结果是DatasetDict
split_train = split_result['train']
split_test = split_result['test']

print(split_train)
print(split_test)

# 数据分桶：把数据均匀的分成N份
train_dataset.shard(num_shards=10, index=0)  # shard 碎片，分成4分，取索引是0的一份，需要index，按照原本数据的进行分桶，不会打乱顺序，当数据不能整除时，前面的会比后面的多一个样本

# 重命名字段
train_dataset.rename_column('review', 'text')  # 可以给原本的数据的列进行重命名

# 删除字段
train_dataset.remove_columns(['review', 'label'])  # 删除某些多余的列，不会改变原有数据集，返回新的数据集


# 映射函数
def map_function(data):
    if data['review'] is not None:
        data['review'] = "My sentence: " + data['review']
    return data


map_dataset = train_dataset.map(map_function)  # 所有的dataset 都被map加了前缀

print(train_dataset['review'][:10])
print(map_dataset['review'][:10])


# 批处理加速
def batch_process(data):
    text = data['review']
    if text is not None:
        text = ['My sentence' + i for i in text]
        data['review'] = text
    return data


if __name__ == '__main__':
    batch_map_dataset = train_dataset.filter(lambda example: example['review'] is not None).map(function=batch_process,
                                                                                                batched=True,
                                                                                                # 批处理为true
                                                                                                batch_size=1000,
                                                                                                # 批处理1000个样本
                                                                                                num_proc=4)  # 4个线程同时处理, windows中使用多进程时，需要放到main方法中
    print(batch_map_dataset['review'][:10])

# 设置数据格式, 会直接修改原始数据
train_dataset.set_format(type='torch', columns=['label'], output_all_columns=True)
print(train_dataset[20])  # 原始数据中的label转换为tensor，文本格式的数据不能直接转换为tensor，需要根据用分词器或者词汇表转换为数字格式之后，再转换为tensor

# 保存为其他格式
train_dataset.to_csv(path_or_buf='./data/ChnSentiCorp.csv')  # 保存为csv格式

# 也可从csv 文件中加载数据
csv_dataset = load_dataset(path='csv', data_files='./data/ChnSentiCorp.csv', split='train')

print(csv_dataset[10])

# 保存为json文件
train_dataset.to_json('./data/ChnSentiCorp.json')

# 读取json 数据
json_dataset= load_dataset(path='json', data_files='./data/ChnSentiCorp.json',split='train')
print(json_dataset[10])