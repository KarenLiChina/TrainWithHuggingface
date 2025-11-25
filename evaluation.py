import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import evaluate

# 评价指标已经从datasets中转到evaluate 包中
all_metrics = evaluate.list_evaluation_modules()
print(all_metrics)

# # 加载的是专门用于评估句子对语义等价性判断任务的指标，主要关注：
# 准确率：整体分类正确率
# F1 分数：平衡精确率和召回率的综合指标
metric = evaluate.load(path='glue', config_name='mrpc')  # path是数据集，config_name 是任务，指定数据集和任务，会返回对应的评价指标
print(metric)

predictions = [0, 1, 0]  # 预测值
references = [0, 1, 1]  # 真实值
# 输出准确率和f1 分数
print(metric.compute(predictions=predictions, references=references))  # 参数列表*号之后的参数必须写关键词参数，不能写位置参数

# 单纯使用准确率
metric = evaluate.load('accuracy')
# 输出准去率
print(metric.compute(predictions=predictions, references=references))
