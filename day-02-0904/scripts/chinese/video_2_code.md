# 视频 2：Python Fundamentals for ML - 代码示例（15分钟）

**所需视觉素材：**
- day-02-0904/assets/images/video/04_step_by_step.png
- 代码执行的屏幕录制
- 终端/IDE设置演示

## 开场（0:00-0:30）
[视觉效果：分屏 - 代码编辑器和终端]

"让我们动手实践Python Fundamentals for ML。我将展示两个你可以跟着做的示例。"

## 第一部分：设置与环境（0:30-2:00）
[视觉效果：安装过程的屏幕录制]

"首先，让我们确保你的环境准备就绪..."

### 安装命令：
```bash
# [屏幕显示]
pip install numpy pandas matplotlib scikit-learn

# 验证
python -c "import pandas; print('设置成功！')"
```

[视觉效果：显示成功安装输出]

## 第二部分：简单示例（2:00-7:00）
[视觉效果：day-02-0904/assets/images/video/04_step_by_step.png 作为背景参考]

"让我们从一个演示核心概念的简单示例开始..."

```python
# [在屏幕上实时输入]
# Simple Python Fundamentals for ML example
# CUSTOMIZE: Add topic-specific simple example
import pandas as pd
import numpy as np

# Basic implementation
def simple_example():
    # Add your simple example here
    print(f"This is a placeholder - customize for Python Fundamentals for ML")
    return True

# Run example
result = simple_example()
print(f"Result: {result}")
```

[视觉效果：显示代码执行和输出]

## 第三部分：实际案例（7:00-12:00）
[视觉效果：在代码和架构图之间切换]

"现在让我们看看这如何与真实数据一起工作..."

```python
# [在屏幕上实时输入]
# Realistic Python Fundamentals for ML example with real data
# CUSTOMIZE: Add topic-specific realistic example
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def realistic_example():
    # CUSTOMIZE: Add realistic example for Python Fundamentals for ML
    print(f"This is a placeholder - customize for Python Fundamentals for ML")
    
    # Include proper error handling
    try:
        # Your realistic implementation here
        pass
    except Exception as e:
        print(f"Error: {e}")
        print("Common solutions: Check data format, verify imports")
    
    return None

# Run with error handling
result = realistic_example()
```

[视觉效果：显示结果并解释输出]

## 第四部分：常见问题排查（12:00-14:30）
[视觉效果：day-02-0904/assets/images/video/07_common_mistakes.png]

"以下是初学者面临的最常见问题..."

[自定义：添加常见错误和解决方案]

## 结尾（14:30-15:00）
"下期视频：如何决定何时使用这种方法..."

---

## 制作注意事项：
- 代码使用大字体（最小16pt）
- 包含错误示例和修复方法
- 在描述中提供GitHub仓库链接
- 显示成功和失败的运行结果
