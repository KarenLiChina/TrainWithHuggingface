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
#### 文本分类/情感分类
nlp_tasks_pipeline.py