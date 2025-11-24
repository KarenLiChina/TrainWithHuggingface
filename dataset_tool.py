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
