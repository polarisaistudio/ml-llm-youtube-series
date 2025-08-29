# 视频 2：传统机器学习与现代AI - 代码示例
**目标时长：** 15分钟  
**格式：** 带清楚解释的现场编程  
**目标观众：** 有基础Python知识的初学者

## 开场（0:00-0:30）
"在上一个视频中，我们介绍了概念。现在让我们看看实际代码。我将向你展示两个工作示例——一个传统ML，一个现代AI——你可以跟着做。

在这个视频结束时，你将运行你的第一个机器学习模型，并理解为什么现代AI的设置更复杂。"

**[视觉效果：分屏显示代码编辑器和终端]**

## 第一部分：传统ML设置（0:30-2:30）
"让我们从传统ML开始，因为它需要零设置——我们需要的一切都随Python一起提供。

**[屏幕录制：打开VS Code/Python环境]**

我使用Python 3.9，但任何最新版本都可以。传统ML的美妙之处在于这些库通常是预安装的：

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
```

如果你遇到错误，只需运行：
```bash
pip install scikit-learn
```

**[现场演示导入工作]**

现在，传统ML方法总是遵循相同的模式：
1. 获取数据
2. 分割数据
3. 训练模型
4. 测试模型

让我们用一个真实的例子来看看这个。"

## 第二部分：传统ML示例（2:30-6:00）
"我们要构建一个花朵分类器——它学习从花瓣和萼片测量值识别花朵种类。

**[带解释的现场编程]**

```python
# 步骤1：获取数据（150个花朵样本，每个4个测量值）
data, labels = load_iris(return_X_y=True)
print(f"数据形状: {data.shape}")  # (150, 4)
print(f"独特种类: {len(set(labels))}")  # 3个种类

# 步骤2：分割数据 - 对测试至关重要！
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.3, random_state=42
)
print(f"训练样本: {len(X_train)}")  # 105
print(f"测试样本: {len(X_test)}")    # 45
```

**[暂停解释训练/测试分割概念]**

"这种分割至关重要——我们在70%的数据上训练，在其他30%上测试。模型在训练期间从未见过测试数据，所以它显示真实世界的性能。

```python
# 步骤3：训练模型
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("模型已训练！")

# 步骤4：测试性能
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"训练准确率: {train_accuracy:.1%}")  # 通常100%
print(f"测试准确率: {test_accuracy:.1%}")    # 通常约97%
```

**[现场运行代码，显示实际输出]**

注意重要的一点——训练准确率是完美的（100%），但测试准确率稍低（97%）。这是正常和健康的。如果它们都是100%，我会怀疑过拟合。"

## 第三部分：理解传统ML结果（6:00-7:30）
"让我们理解刚才发生了什么：

**[视觉效果：显示混淆矩阵或预测]**

```python
# 让我们看一些实际预测
sample_data = X_test[:5]  # 前5个测试样本
predictions = model.predict(sample_data)
actual = y_test[:5]

for i in range(5):
    print(f"预测: {predictions[i]}, 实际: {actual[i]}")
```

**[显示输出]**

模型学会了诸如以下模式：
- 如果花瓣长度 > 4.5 且花瓣宽度 > 1.5 → 种类2
- 如果萼片长度 < 5.0 → 种类0

它基于从训练数据中推导出的数学规则做决策。

**这里显示的传统ML优势：**
✅ 快速训练（即时）
✅ 可预测的性能
✅ 可以解释决策（特征重要性）
✅ 适用于小数据集（150个样本）"

## 第四部分：现代AI设置挑战（7:30-9:30）
"现在让我们尝试现代AI。这就是事情变得更复杂的地方。

**[显示设置过程的屏幕录制]**

对于现代AI，我们需要：
1. API密钥（花钱）
2. 互联网连接
3. 外部服务依赖

```bash
# 安装所需库
pip install openai python-dotenv

# 设置环境变量
export OPENAI_API_KEY="你的密钥在这里"
```

**[显示实际API密钥设置过程——密钥模糊处理]**

这已经突出了一个关键区别——传统ML在本地运行且免费。现代AI通常需要云服务。

让我们构建一个情感分析器，理解文本是积极的、消极的还是中性的：

```python
from openai import OpenAI
import os

# 初始化客户端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_sentiment(text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "分析情感。只回复'积极'、'消极'或'中性'"
                },
                {
                    "role": "user", 
                    "content": text
                }
            ],
            max_tokens=10,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"错误: {e}"
```

**[每个参数的现场编程和解释]**"

## 第五部分：现代AI示例（9:30-12:00）
"让我们测试我们的情感分析器：

**[现场演示]**

```python
# 测试各种示例
examples = [
    "我爱这个产品！它超出了我的期望！",
    "这是我买过的最差的东西。",
    "物品按时到达。它按预期工作。",
    "天哪，这真是太棒了！！！🔥🔥🔥",
    "呃，还好吧我想。"
]

for text in examples:
    sentiment = analyze_sentiment(text)
    print(f"文本: {text}")
    print(f"情感: {sentiment}")
    print("-" * 50)
```

**[显示实际API响应]**

注意正在发生的事情——AI理解：
- 上下文和细微差别
- 情感语言
- 甚至表情符号和俚语
- 中性和消极之间的微妙差异

这对传统ML来说极其困难。你需要为以下内容手动工程特征：
- 积极/消极词汇计数
- 标点符号模式
- 大小写
- 表情符号含义
- 上下文关系

现代AI从大量文本数据集中自动学习了所有这些。"

## 第六部分：比较和成本（12:00-13:30）
"让我们比较我们刚才看到的：

**[视觉效果：并排比较表]**

**传统ML（花朵分类器）：**
- 设置时间：30秒
- 训练时间：即时
- 每次预测成本：$0
- 所需数据：150个样本就足够了
- 可解释性：高（可以显示决策规则）

**现代AI（情感分析）：**
- 设置时间：10分钟（API密钥等）
- 训练时间：已经训练（几个月的预训练）
- 每次预测成本：约$0.001-0.01
- 所需数据：在数十亿文本样本上预训练
- 可解释性：低（黑箱）

**[现场演示检查OpenAI API成本]**

```python
# 每次API调用都花钱
print("每次请求成本：约$0.001")
print("如果你处理1000条评论：约$1")
print("传统ML：初始设置后$0")
```

这种成本差异在规模上很重要。"

## 第七部分：当每种方法失败时（13:30-14:30）
"让我们看看当我们使用错误方法时会发生什么：

**[现场演示]**

```python
# 传统ML能处理创意文本吗？
# 让我们尝试在文本上使用我们的花朵分类器...
# （这显然会失败）

text_example = "我喜欢这个产品"
# 我们甚至不能将文本传递给我们的花朵模型！
# model.predict(text_example)  # 这会崩溃

print("传统ML不能处理非结构化文本，除非进行大量预处理")
```

现代AI的局限性：

```python
# 现代AI能高效预测房价吗？
house_data = "3间卧室，1500平方英尺，建于1990年"
# 我们可以使用GPT，但它将是：
# - 每次预测都很昂贵
# - 结果不一致
# - 不保证准确性
# - 对结构化数据来说是杀鸡用牛刀

print("现代AI对于简单的结构化预测来说是过度杀伤且昂贵的")
```

**关键洞察：为工作使用正确的工具。**"

## 结尾和下一视频（14:30-15:00）
"你现在已经看到了两种方法的实际应用。传统ML示例你现在就可以免费运行。现代AI示例需要设置但显示出令人难以置信的语言理解。

下一个视频：我将准确展示如何为你遇到的任何问题决定使用哪种方法。我们将涵盖真实的商业场景并构建一个决策框架。

自己尝试运行这些代码——链接在描述中。如果你遇到困难，请在评论中留言，我会帮你调试。

订阅这个系列的其余部分——明天我们将深入实际决策制定。"

**[带有代码仓库链接和订阅按钮的结束画面]**

---

## 制作注意事项：

**代码仓库结构：**
```
/day-01-video-2/
  ├── traditional_ml_example.py
  ├── modern_ai_example.py  
  ├── requirements.txt
  └── README.md
```

**屏幕录制设置：**
- 移动查看的大字体
- 更好对比度的深色主题
- 终端和代码编辑器并排
- 光标高亮以便跟随

**要解决的常见初学者问题：**
- Python版本兼容性
- 库安装问题
- API密钥设置困惑
- 不同操作系统上的环境变量设置

**参与元素：**
- 在7:00暂停提问
- "自己尝试"时刻
- 代码下载链接
- 故障排除帮助提供