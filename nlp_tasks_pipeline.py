import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import pipeline

# 文本分类，传入参数task 任务类型，后续的model是可以选的， 不传model的时候，会自动从huggingface中去找对应的模型
# 中文分类一般
classifier = pipeline('sentiment-analysis')  # 情感分析，用于文本分类，建议输入模型
print(classifier("我喜欢你"))
print(classifier("我讨厌你"))
print(classifier("I love you"))
print(classifier("I hate you"))
print(classifier("聪明"))
print(classifier("大聪明"))
print(classifier("大智若愚"))
