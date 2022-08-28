# noting by nakaizura, AUG, 2022
# 源码给的注释已经很多，个人继续细化了一些。

# 在这个脚本中，你将学习
# 1. 如何在 openprompt 中使用集成 huggingface 数据集工具
# 在不同的数据集中启用快速学习。
# 2. 如何使用模板语言实例化模板
# 3. 模板如何将输入示例包装成模板化示例。
# 4. 我们如何隐藏 PLM 标记化细节并提供简单的标记化
# 5. 如何使用一个/多个标签词构建语言器
# 5. 如何像传统的预训练模型一样训练提示。


# 载入数据集
from datasets import load_dataset
# raw_dataset = load_dataset('super_glue', 'cb', cache_dir="../datasets/.cache/huggingface_datasets") #用这条命令会自动下载，前提是有网
# raw_dataset['train'][0]
from datasets import load_from_disk
raw_dataset = load_from_disk("/home/hushengding/huggingface_datasets/saved_to_disk/super_glue.cb") #从下载好的disk直接载入
# 请注意，如果您在 GPU 集群中运行此脚本，您可能无法直接连接到 huggingface 网站。
# 在这种情况下，我们建议您在某些具有 Internet 连接的机器上运行 `raw_dataset = load_dataset(...)`。
# 然后使用 `raw_dataset.save_to_disk(path)` 方法保存到本地路径。
# 第三次将保存的内容上传到集群中的机器中。
# 然后使用 `load_from_disk` 方法加载数据集。

from openprompt.data_utils import InputExample

dataset = {}
for split in ['train', 'validation', 'test']: #切分数据集
    dataset[split] = []
    for data in raw_dataset[split]:
        input_example = InputExample(text_a = data['premise'], text_b = data['hypothesis'], label=int(data['label']), guid=data['idx'])
        dataset[split].append(input_example)
print(dataset['train'][0])

# 你可以通过调用来加载openprompt提供的plm相关模型：
from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-base") #直接加载T5模型

# 构建模板
# 模板可以通过 yaml 配置构建，也可以通过直接传递参数来构建。
from openprompt.prompts import ManualTemplate #这里调用手工模版
template_text = '{"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"mask"}.' #输入是text_a，mask是prompt
mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

# 为了更好地理解模板是如何包装示例的，我们可视化一个实例。

wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
print(wrapped_example)


# 现在，包装好的示例已准备好传递给分词器，从而为语言模型生成输入。
# 您可以使用分词器自己对输入进行分词，但我们建议使用我们的包装分词器，它是一个用于 InputExample 的包装分词器。
# 如果你使用我们的`load_plm`函数，包装器已经给出，否则，你应该根据你的情况选择合适的包装器
# `openprompt.plms.__init__.py` 中的配置。
# 注意当使用t5进行分类时，我们只需要将<pad> <extra_id_0> <eos>传给decoder即可。
# 损失在 <extra_id_0> 处计算。 因此通过 decoder_max_length=3 节省空间
wrapped_t5tokenizer = WrapperClass(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head") #两种分词器可选
# or
from openprompt.plms import T5TokenizerWrapper
wrapped_t5tokenizer= T5TokenizerWrapper(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")

# 你可以看到一个标记化的例子是什么样子的
tokenized_example = wrapped_t5tokenizer.tokenize_one_example(wrapped_example, teacher_forcing=False)
print(tokenized_example)
print(tokenizer.convert_ids_to_tokens(tokenized_example['input_ids']))
print(tokenizer.convert_ids_to_tokens(tokenized_example['decoder_input_ids']))

# 现在是时候将整个数据集转换为输入格式了！
# 只需遍历数据集即可实现！

model_inputs = {}
for split in ['train', 'validation', 'test']: #遍历数据集进行分词
    model_inputs[split] = []
    for sample in dataset[split]:
        tokenized_example = wrapped_t5tokenizer.tokenize_one_example(mytemplate.wrap_one_example(sample), teacher_forcing=False)
        model_inputs[split].append(tokenized_example)


# 我们提供了一个 `PromptDataLoader` 类来帮助你完成上述所有事情，并将它们包装到一个 `torch.DataLoader` 样式的迭代器中。
from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
    batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")
# next(iter(train_dataloader)) #这个迭代器设计在使用起来会更方便


# 定义Verbalizer（[Mask]位置的输出是一个单词，需要把这些单词映射成"yes","no"标签）
# 在分类中，你需要定义你的verbalizer，它是从词汇表上的logits到最终标签概率的映射。 让我们看一下的详细信息：
from openprompt.prompts import ManualVerbalizer
import torch

# 例如Verbalizer在每个类中包含多个标签词
myverbalizer = ManualVerbalizer(tokenizer, num_classes=3,
                        label_words=[["yes"], ["no"], ["maybe"]]) #定义了三类label空间

print(myverbalizer.label_words_ids)
logits = torch.randn(2,len(tokenizer)) # 从 plm 创建一个伪输出，并且
print(myverbalizer.process_logits(logits)) # 看看Verbalizer做了什么


# 虽然可以手动将plm、template、verbalizer组合在一起，但我们提供了一个pipeline
# 模型从 PromptDataLoader 中获取批处理数据并生成一个类 logits

from openprompt import PromptForClassification

use_cuda = True
# 这里会将前面几步构建的模板(promptTemplate)、预训练模型(plm)、输出映射(promptVerbalizer)进一步组成promptModel
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()

# 开始训练
from transformers import  AdamW, get_linear_schedule_with_warmup #载入huggingface中的一些支持库
loss_func = torch.nn.CrossEntropyLoss() #交叉熵
no_decay = ['bias', 'LayerNorm.weight']
# 对偏差和 LayerNorm 参数设置不衰减是一个经验
optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4) #优化器

for epoch in range(10): #开始循环10个epoch
    tot_loss = 0
    for step, inputs in enumerate(train_dataloader): #从dataloader迭代数据
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs) #得到映射到label的概率
        labels = inputs['label']
        loss = loss_func(logits, labels) #计算loss
        loss.backward() #方向传播
        tot_loss += loss.item()
        optimizer.step() #梯度更新
        optimizer.zero_grad()
        if step %100 ==1:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)

# 载入评估集
validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
    batch_size=4,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

allpreds = []
alllabels = []
for step, inputs in enumerate(validation_dataloader): #同样迭代评估数据
    if use_cuda:
        inputs = inputs.cuda()
    logits = prompt_model(inputs) #预测分类概率
    labels = inputs['label']
    alllabels.extend(labels.cpu().tolist())
    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds) #计算准确率
print(acc)
