## python环境要求
```bash
pip install -r requirements.txt
```

### 编码器的流程：定义字典，句子预处理，分词，编码
encoder_examples.py

### 从huggingface 加载数据集，并保持数据集，数据集中常见的一些工具
dataset_tool.py

### 评价指标
evaluation.py

### 使用管道工具pipeline完成一些自然语言处理npl的任务
#### text-classification/sentiment-analysis 为文本序列分配标签文本分类/情感分类
#### question-answering 根据给定的上下文和问题，从文本中提取答案
#### fill-mask 预测并填充序列中被掩盖的词语
#### text-generation 根据给定的提示（Prompt）生成新的文本
#### token-classification/ner 为序列中的每个词元分配标签（如命名实体识别）
#### summarization 为长文本或文档生成简洁的摘要
#### translation ，可以指定模型
#### pipeline中指定模型，tokenizer，可以进行其他的翻译
nlp_tasks_pipeline.py

## 基于预训练模型 进行微调，将微调后的模型进行存储，加载，从中间checkpoint处加载，使用模型进行预测
trianingTool.py