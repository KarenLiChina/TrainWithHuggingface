import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 文本分类，传入参数task 任务类型，后续的model是可以选的， 不传model的时候，会自动从huggingface中去找对应的模型
# 中文分类一般
classifier = pipeline('sentiment-analysis')  # text-classification/sentiment-analysis情感分析，用于文本分类，建议输入模型
print(classifier("我喜欢你"))
print(classifier("我讨厌你"))
print(classifier("I love you"))
print(classifier("I hate you"))
print(classifier("聪明"))
print(classifier("大聪明"))
print(classifier("大智若愚"))

# 阅读理解
question_answerer = pipeline('question-answering')  # 根据给定的上下文和问题，从文本中提取答案

context = r"""SAN FRANCISCO — ChatGPT maker OpenAI announced a major victory on Tuesday, gaining the blessing of the attorneys general in California and Delaware to complete its controversial, multi-billion dollar business restructuring after months of intense public scrutiny.
But the dominant, San Francisco-based AI company, which is valued at $500 billion, still faces potential hurdles with continued protest from influential civil society groups and an ongoing lawsuit from former business partner turned rival Elon Musk.
"""
result = question_answerer(question="what the challenge does OpenAI face?", context=context)
print(result)
result = question_answerer(question="What's the OpenAI valuation?", context=context)
print(result)

# 完形填空
sentence = "HuggingFace is creating a <mask> that the community uses to solve NLP tasks."
unmasker = pipeline("fill-mask")  # 预测并填充序列中被掩盖的词语。
result = unmasker(sentence)  # 返回一个可能的填入词的list，包含每个词的评分
print(result)

# 文本生成
text_gc = pipeline("text-generation")  # 默认模型是GPT2
result = text_gc("When I was young, I listen to the radio",  # 提示词
                 max_length=50,  # 最大长度
                 do_sample=False)  # 不作为样本
print(result)

# 命名实体识别，在文本中找到实体，如人名、地名、公司名。。。
ner_pipe = pipeline('token-classification')  # token-classification/ner 为序列中的每个词元分配标签（如命名实体识别）
result = ner_pipe(context)
for entity in result:
    print(entity)

# 文本摘要
article = """
Don't miss out on our latest stories. Add PCMag as a preferred source on Google.
OpenAI generally doesn't share your ChatGPT conversations with third parties. However, an analytics firm has discovered a way to capture users' prompts, which can reveal queries about sensitive topics such as prostitution, medical conditions, and immigration status. 
New York-based Profound has been selling access to the queries through a service called Prompt Volumes, which launched earlier this year. It can help companies identify what users are asking major chatbot providers, including ChatGPT, Google Gemini, and Anthropic's Claude, though ChatGPT data dominates.
"""
summarizer = pipeline('summarization') #为长文本或文档生成简洁的摘要
result = summarizer(article)
print(result)

# 翻译任务
translater = pipeline('translation', model="Helsinki-NLP/opus-mt-en-zh") # 英语翻译成中文，可以指定模型
result = translater(context)
print(result)

# 替换模型执行任务
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
new_trans = pipeline("translation", model=model, tokenizer=tokenizer)
result = new_trans(context)
print(result)
