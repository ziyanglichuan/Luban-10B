
## 鲁班大模型（Luban）介绍

**作者：刘祥根、郭彦、吕建成**

我们推出了一个用于工业领域的预训练语言模型——鲁班大模型（Luban）。Luban 是一个强大的语言模型，专为解决工业领域的复杂问题而设计。目前，我们已经发布了包含 10B 参数的模型，并且正在进一步训练，以提升其性能和应用范围。

### 模型参数和特点

- **Luban-10B-v1.0**：拥有 10B 参数，已经发布并可供使用。适用于一般工业领域的自然语言处理任务。该模型支持长文本处理，能够在处理大段文本时保持高效的性能，同时快速生成高质量的输出。

### 项目特点

1. **预训练模型**：Luban 在大规模工业语料库上进行预训练，能够理解和生成与工业领域相关的高质量文本，特别适合处理长文本内容。
2. **灵活生成**：提供简单易用的生成测试代码，方便用户快速验证模型效果，能够迅速生成符合需求的文本。

### 项目结构

- **生成测试**：快速测试模型的生成能力，支持自定义问题。

我们相信，Luban 将成为工业领域的有力工具，帮助用户在各种自然语言处理任务中获得更高的效率和精度。无论是用于技术文档的生成、产品描述的编写，还是复杂技术问题的问答，Luban 都能提供卓越的支持，特别是在需要处理长文本的场景中表现优异。

欢迎大家下载和使用Luban，并期待在未来发布更多改进和增强版本。

## 快速开始

### 环境搭建

```bash
pip install -r requirements.txt
```

### 预训练模型下载

| **模型名称** | **模型参数** | **下载地址** |
|--------------|---------------|--------------|
| **Luban-10B-v1.0** | - **max_seq_len**: 2048<br>- **dim**: 4096<br>- **n_layers**: 48<br>- **n_heads**: 32 | [百度云盘下载](https://pan.baidu.com/s/1FkhcfS6CPreLZSgcZToBUg) 提取码：onk2|

### 测试模型生成

```bash
# 测试Luban模型
# 长文本快速生成
python python Luban-10B_generate.py --activate_selective --model_name_or_path ./model/Luban-10B
# 传统生成方式
python Luban-10B_generate.py --model_name_or_path ./model/Luban-10B
```

### 示例

```bash
# 示例一：Input：‘制造工艺设计是’
Luban-10B-v1.0 response：‘生产过程中的重要环节，它和工艺过程密切相关。在工艺设计中，必须考虑生产工艺的可行性、经济性，力求使工艺方案能经济、合理、有效地满足产品技术要求。在工艺设计中，应尽可能避免产生各种工艺设计缺陷，使工艺设计满足工艺要求。’

# 示例二：Input：‘挤塑工艺是’
Luban-10B-v1.0 response：‘塑料板材经加热，挤出成中空的长条，然后经冷却、固化而制成各种中空板材，其技术特征是板材在生产过程中采用挤包机对板材进行压出包边、压痕和开槽等工序，使板材成为中空塑料制品。’
```