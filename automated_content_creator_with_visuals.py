#!/usr/bin/env python3
"""
Enhanced Automated Content Creator with Visual Generation and Bilingual Support
Includes image generation for video content and Chinese/English script versions
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

class EnhancedContentCreator:
    """Creates content with visual aids and bilingual support for video production"""
    
    def __init__(self, day_number: int, topic: str, target_audience: str = "beginners"):
        self.day_number = day_number
        self.topic = topic
        self.target_audience = target_audience
        self.date = datetime.now().strftime("%m%d")
        self.folder_name = f"day-{day_number:02d}-{self.date}"
        self.base_path = Path(self.folder_name)
        
        # Visual elements for video
        self.required_visuals = [
            ("01_concept_overview", "Main concept overview diagram"),
            ("02_comparison", "Before/after or A vs B comparison"),
            ("03_architecture", "System architecture or flow diagram"),
            ("04_step_by_step", "Step-by-step process visualization"),
            ("05_decision_tree", "Decision framework flowchart"),
            ("06_real_example", "Real-world application example"),
            ("07_common_mistakes", "Common mistakes illustration"),
            ("08_learning_path", "Learning progression timeline")
        ]
        
        # Set up matplotlib style for consistent visuals
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
    def create_folder_structure(self):
        """Create complete folder structure including visuals"""
        folders = [
            self.base_path / "scripts" / "english",
            self.base_path / "scripts" / "chinese", 
            self.base_path / "video-scripts",
            self.base_path / "code" / "demos",
            self.base_path / "assets" / "images" / "video",
            self.base_path / "blog-posts" / "medium",
            self.base_path / "blog-posts" / "linkedin",
            self.base_path / "blog-posts" / "github-pages"
        ]
        
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Created structure for Day {self.day_number}: {self.topic}")
    
    def generate_visual_aids(self) -> Dict[str, str]:
        """Generate visual aids for video content"""
        output_dir = self.base_path / "assets" / "images" / "video"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_images = {}
        
        # 1. Concept Overview
        fig, ax = plt.subplots(figsize=(12, 8))
        self._create_concept_overview(ax)
        image_path = output_dir / "01_concept_overview.png"
        plt.savefig(image_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        generated_images["concept_overview"] = str(image_path)
        
        # 2. Comparison Diagram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        self._create_comparison_diagram(ax1, ax2)
        image_path = output_dir / "02_comparison.png"
        plt.savefig(image_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        generated_images["comparison"] = str(image_path)
        
        # 3. Architecture Diagram
        fig, ax = plt.subplots(figsize=(14, 10))
        self._create_architecture_diagram(ax)
        image_path = output_dir / "03_architecture.png"
        plt.savefig(image_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        generated_images["architecture"] = str(image_path)
        
        # 4. Step-by-Step Process
        fig, ax = plt.subplots(figsize=(16, 6))
        self._create_step_by_step_process(ax)
        image_path = output_dir / "04_step_by_step.png"
        plt.savefig(image_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        generated_images["step_by_step"] = str(image_path)
        
        # 5. Decision Tree
        fig, ax = plt.subplots(figsize=(12, 10))
        self._create_decision_tree(ax)
        image_path = output_dir / "05_decision_tree.png"
        plt.savefig(image_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        generated_images["decision_tree"] = str(image_path)
        
        # 6. Real Example
        fig, ax = plt.subplots(figsize=(12, 8))
        self._create_real_example(ax)
        image_path = output_dir / "06_real_example.png"
        plt.savefig(image_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        generated_images["real_example"] = str(image_path)
        
        # 7. Common Mistakes
        fig, ax = plt.subplots(figsize=(14, 8))
        self._create_common_mistakes(ax)
        image_path = output_dir / "07_common_mistakes.png"
        plt.savefig(image_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        generated_images["common_mistakes"] = str(image_path)
        
        # 8. Learning Path
        fig, ax = plt.subplots(figsize=(16, 8))
        self._create_learning_path(ax)
        image_path = output_dir / "08_learning_path.png"
        plt.savefig(image_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        generated_images["learning_path"] = str(image_path)
        
        print(f"✅ Generated {len(generated_images)} visual aids for videos")
        return generated_images
    
    def create_english_video_scripts(self, images: Dict[str, str]) -> Dict[str, str]:
        """Create English video scripts with visual references"""
        
        scripts = {
            "video_1_concepts": f"""# Video 1: {self.topic} - Core Concepts (10 minutes)

**Visual Assets Needed:**
- {images.get('concept_overview', 'concept_overview.png')}
- {images.get('comparison', 'comparison.png')}

## Opening Hook (0:00-0:15)
[VISUAL: Show {images.get('concept_overview', 'concept_overview.png')} - 3 seconds full screen]

"If you've been confused about {self.topic}, this video will clear it up in 10 minutes. I'll show you the one concept that makes everything click."

[VISUAL: Transition to talking head with concept image in corner]

## Section 1: The Big Picture (0:15-2:00)
[VISUAL: {images.get('concept_overview', 'concept_overview.png')} - highlight different sections as explained]

"Let's start with the fundamentals. {self.topic} is..."

[CUSTOMIZE: Add topic-specific big picture explanation]

## Section 2: Key Differences (2:00-5:00)  
[VISUAL: {images.get('comparison', 'comparison.png')} - split screen comparison]

"The key difference between traditional approaches and {self.topic} is..."

[CUSTOMIZE: Add comparison explanation with visual callouts]

## Section 3: Why This Matters (5:00-8:00)
[VISUAL: {images.get('real_example', 'real_example.png')} - real-world application]

"Here's why this matters in practice..."

[CUSTOMIZE: Add practical applications]

## Section 4: Key Insights (8:00-9:30)
[VISUAL: Return to concept overview with highlighted key points]

"The three things you need to remember..."

## Closing & Next Video Teaser (9:30-10:00)
[VISUAL: Preview thumbnail of next video]

"Next video: I'll show you actual code examples you can run right now..."

**Call to Action:**
- Like if this helped clarify the concepts
- Subscribe for the rest of the series
- Comment with questions

---

## Production Notes:
- All visuals should be 1920x1080 minimum
- Use consistent color scheme across videos
- Include captions for accessibility
- Highlight cursor for any on-screen elements
""",

            "video_2_code": f"""# Video 2: {self.topic} - Code Examples (15 minutes)

**Visual Assets Needed:**
- {images.get('step_by_step', 'step_by_step.png')}
- Screen recordings of code execution
- Terminal/IDE setup demonstrations

## Opening (0:00-0:30)
[VISUAL: Split screen - code editor and terminal]

"Let's get hands-on with {self.topic}. I'll show you two examples you can follow along with."

## Section 1: Setup & Environment (0:30-2:00)
[VISUAL: Screen recording of installation process]

"First, let's make sure your environment is ready..."

### Installation Commands:
```bash
# [SHOW ON SCREEN]
pip install numpy pandas matplotlib scikit-learn

# Verification
python -c "import pandas; print('Setup successful!')"
```

[VISUAL: Show successful installation output]

## Section 2: Simple Example (2:00-7:00)
[VISUAL: {images.get('step_by_step', 'step_by_step.png')} as background reference]

"Let's start with a simple example that demonstrates the core concept..."

```python
# [TYPE LIVE ON SCREEN]
{self._generate_simple_code_example()}
```

[VISUAL: Show code execution and output]

## Section 3: Real-World Example (7:00-12:00)
[VISUAL: Switch between code and architecture diagram]

"Now let's see how this works with real data..."

```python
# [TYPE LIVE ON SCREEN] 
{self._generate_realistic_code_example()}
```

[VISUAL: Show results and explain output]

## Section 4: Troubleshooting Common Issues (12:00-14:30)
[VISUAL: {images.get('common_mistakes', 'common_mistakes.png')}]

"Here are the most common issues beginners face..."

[CUSTOMIZE: Add common errors and solutions]

## Closing (14:30-15:00)
"Next video: How to decide when to use this approach..."

---

## Production Notes:
- Use large font sizes (16pt minimum) for code
- Include error examples and how to fix them
- Provide GitHub repository link in description
- Show both successful and failed runs
""",

            "video_3_decisions": f"""# Video 3: {self.topic} - Decision Framework (10 minutes)

**Visual Assets Needed:**
- {images.get('decision_tree', 'decision_tree.png')}
- Business scenario illustrations

## Opening (0:00-0:30)
[VISUAL: {images.get('decision_tree', 'decision_tree.png')} - preview then minimize]

"You've seen the concepts and code. Now: when should you actually use {self.topic}?"

## Section 1: Decision Framework (0:30-2:30)
[VISUAL: {images.get('decision_tree', 'decision_tree.png')} - highlight each decision point]

"Here's my 4-question framework..."

1. What type of data do you have?
2. How much data do you have?
3. Do you need to explain decisions?
4. What's your budget?

[VISUAL: Animate through decision tree as questions are asked]

## Section 2: Real Business Scenario 1 (2:30-5:00)
[VISUAL: Business scenario illustration]

"Let me show you how this works with a real example..."

**Scenario:** [CUSTOMIZE: Industry-specific example]
**Analysis:** [Walk through decision tree]
**Decision:** [Show recommended approach]
**Outcome:** [Real results]

## Section 3: Real Business Scenario 2 (5:00-7:30)
[VISUAL: Different business scenario]

"Here's another example where the decision is different..."

[CUSTOMIZE: Contrasting scenario]

## Section 4: Common Decision Mistakes (7:30-9:30)
[VISUAL: {images.get('common_mistakes', 'common_mistakes.png')}]

"Here are the mistakes I see most often..."

[CUSTOMIZE: Add decision-specific mistakes]

## Closing (9:30-10:00)
"Final video: Your realistic learning path and next steps..."
""",

            "video_4_learning": f"""# Video 4: {self.topic} - Learning Path & Next Steps (10 minutes)

**Visual Assets Needed:**
- {images.get('learning_path', 'learning_path.png')}
- Timeline animations

## Opening Hook (0:00-0:30)
[VISUAL: {images.get('learning_path', 'learning_path.png')} - dramatic reveal]

"Let me give you some hard truths about learning {self.topic} that will save you months of frustration."

## Section 1: Reality Check (0:30-2:00)
[VISUAL: Timeline comparison - marketing claims vs reality]

**Marketing Claims vs Reality:**
- "Master in 30 days" → Actually: 6-12 months
- "Build production apps immediately" → Actually: Start with fundamentals
- "Skip the boring basics" → Actually: Basics are crucial

## Section 2: Realistic Learning Phases (2:00-6:00)
[VISUAL: {images.get('learning_path', 'learning_path.png')} - animate through phases]

**Phase 1 (Months 1-2):** Foundation
- Python fundamentals
- Data handling basics
- Environment setup mastery

**Phase 2 (Months 3-4):** Core Concepts
- {self.topic} fundamentals
- Simple implementations
- Understanding limitations

**Phase 3 (Months 5-6):** Real-World Application
- Working with messy data
- Production considerations
- Performance optimization

**Phase 4 (Months 7-12):** Specialization
- Advanced techniques
- Domain expertise
- Building portfolio projects

## Section 3: Avoiding Common Learning Traps (6:00-8:30)
[VISUAL: {images.get('common_mistakes', 'common_mistakes.png')}]

**Trap 1:** Tutorial Hell
**Trap 2:** Perfectionism Paralysis  
**Trap 3:** Shiny Object Syndrome
**Trap 4:** Learning in Isolation

## Section 4: Your Action Plan (8:30-9:30)
[VISUAL: Checklist animation]

**This Week:**
1. Assess your current Python skills
2. Set up your development environment
3. Choose one practice project
4. Join a learning community

## Closing (9:30-10:00)
[VISUAL: Series overview and Day 2 preview]

"Tomorrow: Day {self.day_number + 1} - [Next Topic Preview]"

**Series Progress:** Day {self.day_number} of 40 complete!
"""
        }
        
        return scripts
    
    def create_chinese_video_scripts(self, images: Dict[str, str]) -> Dict[str, str]:
        """Create Chinese video scripts with visual references"""
        
        scripts = {
            "video_1_concepts": f"""# 视频 1：{self.topic} - 核心概念（10分钟）

**所需视觉素材：**
- {images.get('concept_overview', 'concept_overview.png')}
- {images.get('comparison', 'comparison.png')}

## 开场吸引（0:00-0:15）
[视觉效果：显示 {images.get('concept_overview', 'concept_overview.png')} - 全屏3秒]

"如果你对{self.topic}感到困惑，这个视频将在10分钟内为你解答。我将展示让一切豁然开朗的关键概念。"

[视觉效果：转到讲解画面，概念图显示在角落]

## 第一部分：全局概览（0:15-2:00）
[视觉效果：{images.get('concept_overview', 'concept_overview.png')} - 随着解释高亮不同部分]

"让我们从基础开始。{self.topic}是..."

[自定义：添加特定主题的全局解释]

## 第二部分：关键差异（2:00-5:00）
[视觉效果：{images.get('comparison', 'comparison.png')} - 分屏对比]

"传统方法和{self.topic}之间的关键区别是..."

[自定义：添加对比解释和视觉标注]

## 第三部分：为什么重要（5:00-8:00）
[视觉效果：{images.get('real_example', 'real_example.png')} - 实际应用案例]

"这在实践中的重要性体现在..."

[自定义：添加实际应用]

## 第四部分：关键洞察（8:00-9:30）
[视觉效果：回到概念总览，高亮关键要点]

"你需要记住的三个要点是..."

## 结尾与下期预告（9:30-10:00）
[视觉效果：下期视频缩略图预览]

"下期视频：我将展示你可以立即运行的实际代码示例..."

**行动号召：**
- 如果这帮助澄清了概念，请点赞
- 订阅观看完整系列
- 在评论中提出问题

---

## 制作注意事项：
- 所有视觉效果最少1920x1080分辨率
- 在所有视频中使用一致的色彩方案
- 包含字幕以提高可访问性
- 突出显示屏幕上的任何元素的光标
""",

            "video_2_code": f"""# 视频 2：{self.topic} - 代码示例（15分钟）

**所需视觉素材：**
- {images.get('step_by_step', 'step_by_step.png')}
- 代码执行的屏幕录制
- 终端/IDE设置演示

## 开场（0:00-0:30）
[视觉效果：分屏 - 代码编辑器和终端]

"让我们动手实践{self.topic}。我将展示两个你可以跟着做的示例。"

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
[视觉效果：{images.get('step_by_step', 'step_by_step.png')} 作为背景参考]

"让我们从一个演示核心概念的简单示例开始..."

```python
# [在屏幕上实时输入]
{self._generate_simple_code_example()}
```

[视觉效果：显示代码执行和输出]

## 第三部分：实际案例（7:00-12:00）
[视觉效果：在代码和架构图之间切换]

"现在让我们看看这如何与真实数据一起工作..."

```python
# [在屏幕上实时输入]
{self._generate_realistic_code_example()}
```

[视觉效果：显示结果并解释输出]

## 第四部分：常见问题排查（12:00-14:30）
[视觉效果：{images.get('common_mistakes', 'common_mistakes.png')}]

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
""",

            "video_3_decisions": f"""# 视频 3：{self.topic} - 决策框架（10分钟）

**所需视觉素材：**
- {images.get('decision_tree', 'decision_tree.png')}
- 商业场景插图

## 开场（0:00-0:30）
[视觉效果：{images.get('decision_tree', 'decision_tree.png')} - 预览后最小化]

"你已经了解了概念和代码。现在：你应该在什么时候实际使用{self.topic}？"

## 第一部分：决策框架（0:30-2:30）
[视觉效果：{images.get('decision_tree', 'decision_tree.png')} - 高亮每个决策点]

"这是我的4个问题框架..."

1. 你有什么类型的数据？
2. 你有多少数据？
3. 你需要解释决策吗？
4. 你的预算是多少？

[视觉效果：在提出问题时动画展示决策树]

## 第二部分：真实商业场景1（2:30-5:00）
[视觉效果：商业场景插图]

"让我用一个真实例子展示这如何工作..."

**场景：** [自定义：特定行业示例]
**分析：** [遍历决策树]
**决策：** [显示推荐方法]
**结果：** [真实结果]

## 第三部分：真实商业场景2（5:00-7:30）
[视觉效果：不同的商业场景]

"这里是另一个决策不同的例子..."

[自定义：对比场景]

## 第四部分：常见决策错误（7:30-9:30）
[视觉效果：{images.get('common_mistakes', 'common_mistakes.png')}]

"这些是我最常看到的错误..."

[自定义：添加特定决策错误]

## 结尾（9:30-10:00）
"最后一期视频：你的现实学习路径和下一步..."
""",

            "video_4_learning": f"""# 视频 4：{self.topic} - 学习路径与下一步（10分钟）

**所需视觉素材：**
- {images.get('learning_path', 'learning_path.png')}
- 时间线动画

## 开场吸引（0:00-0:30）
[视觉效果：{images.get('learning_path', 'learning_path.png')} - 戏剧性展现]

"让我告诉你一些关于学习{self.topic}的严酷真相，这将为你节省数月的挫折。"

## 第一部分：现实检验（0:30-2:00）
[视觉效果：时间线对比 - 营销宣称vs现实]

**营销宣称vs现实：**
- "30天精通" → 实际：6-12个月
- "立即构建生产应用" → 实际：从基础开始
- "跳过枯燥基础" → 实际：基础至关重要

## 第二部分：现实学习阶段（2:00-6:00）
[视觉效果：{images.get('learning_path', 'learning_path.png')} - 逐阶段动画]

**第1阶段（1-2个月）：** 基础
- Python基础
- 数据处理基础
- 环境设置掌握

**第2阶段（3-4个月）：** 核心概念
- {self.topic}基础
- 简单实现
- 理解局限性

**第3阶段（5-6个月）：** 实际应用
- 处理混乱数据
- 生产考虑
- 性能优化

**第4阶段（7-12个月）：** 专业化
- 高级技术
- 领域专业知识
- 构建作品集项目

## 第三部分：避免常见学习陷阱（6:00-8:30）
[视觉效果：{images.get('common_mistakes', 'common_mistakes.png')}]

**陷阱1：** 教程地狱
**陷阱2：** 完美主义麻痹
**陷阱3：** 新奇事物综合症
**陷阱4：** 孤立学习

## 第四部分：你的行动计划（8:30-9:30）
[视觉效果：清单动画]

**本周：**
1. 评估你当前的Python技能
2. 设置开发环境
3. 选择一个练习项目
4. 加入学习社区

## 结尾（9:30-10:00）
[视觉效果：系列概览和第2天预览]

"明天：第{self.day_number + 1}天 - [下一主题预览]"

**系列进度：** 40天中的第{self.day_number}天完成！
"""
        }
        
        return scripts
    
    # Visual generation helper methods
    def _create_concept_overview(self, ax):
        """Create concept overview diagram"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_title(f'{self.topic} - Concept Overview', fontsize=16, fontweight='bold')
        
        # Main concept box
        main_box = patches.FancyBboxPatch((2, 3), 6, 2, 
                                         boxstyle="round,pad=0.1",
                                         facecolor='lightblue',
                                         edgecolor='darkblue',
                                         linewidth=2)
        ax.add_patch(main_box)
        ax.text(5, 4, f'{self.topic}', ha='center', va='center', 
                fontsize=14, fontweight='bold')
        
        # Add placeholder elements - customize per topic
        ax.text(5, 6.5, 'CUSTOMIZE: Add topic-specific elements', 
                ha='center', va='center', fontsize=10, style='italic', color='red')
        
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _create_comparison_diagram(self, ax1, ax2):
        """Create before/after or A vs B comparison"""
        ax1.set_title('Traditional Approach', fontsize=14, fontweight='bold')
        ax2.set_title(f'{self.topic} Approach', fontsize=14, fontweight='bold')
        
        # Add placeholder content - customize per topic
        ax1.text(0.5, 0.5, 'CUSTOMIZE:\nTraditional method\ncharacteristics', 
                ha='center', va='center', transform=ax1.transAxes,
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
        
        ax2.text(0.5, 0.5, f'CUSTOMIZE:\n{self.topic} method\ncharacteristics', 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
        
        ax1.axis('off')
        ax2.axis('off')
    
    def _create_architecture_diagram(self, ax):
        """Create system architecture or flow diagram"""
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.set_title(f'{self.topic} - System Architecture', fontsize=16, fontweight='bold')
        
        # Flow boxes
        boxes = [
            (1, 6, 'Input\nData'),
            (4, 6, 'Processing\nLayer'),
            (7, 6, f'{self.topic}\nCore'),
            (10, 6, 'Output\nResults')
        ]
        
        for x, y, text in boxes:
            box = patches.FancyBboxPatch((x-0.8, y-0.8), 1.6, 1.6,
                                        boxstyle="round,pad=0.1",
                                        facecolor='lightblue',
                                        edgecolor='darkblue')
            ax.add_patch(box)
            ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Arrows
        for i in range(len(boxes)-1):
            ax.arrow(boxes[i][0]+0.8, boxes[i][1], 1.4, 0, 
                    head_width=0.2, head_length=0.2, fc='darkblue', ec='darkblue')
        
        ax.text(6, 3, 'CUSTOMIZE: Add specific architecture details', 
                ha='center', va='center', fontsize=10, style='italic', color='red')
        
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _create_step_by_step_process(self, ax):
        """Create step-by-step process visualization"""
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 6)
        ax.set_title(f'{self.topic} - Step-by-Step Process', fontsize=16, fontweight='bold')
        
        steps = [
            (2, 3, '1', 'Setup'),
            (5, 3, '2', 'Prepare'),
            (8, 3, '3', 'Process'),
            (11, 3, '4', 'Analyze'),
            (14, 3, '5', 'Deploy')
        ]
        
        for x, y, num, text in steps:
            # Step circle
            circle = patches.Circle((x, y), 0.8, facecolor='lightgreen', edgecolor='darkgreen')
            ax.add_patch(circle)
            ax.text(x, y+0.1, num, ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(x, y-1.5, text, ha='center', va='center', fontsize=10)
        
        # Connecting arrows
        for i in range(len(steps)-1):
            ax.arrow(steps[i][0]+0.8, steps[i][1], 1.4, 0,
                    head_width=0.15, head_length=0.2, fc='darkgreen', ec='darkgreen')
        
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _create_decision_tree(self, ax):
        """Create decision framework flowchart"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.set_title(f'When to Use {self.topic} - Decision Framework', fontsize=14, fontweight='bold')
        
        # Decision nodes
        ax.text(5, 11, 'Start Here', ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'), fontsize=12, fontweight='bold')
        
        ax.text(5, 9, 'What type of data?', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'), fontsize=10)
        
        ax.text(2, 7, 'Structured', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'), fontsize=9)
        
        ax.text(8, 7, 'Unstructured', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'), fontsize=9)
        
        # Add more decision points - customize per topic
        ax.text(5, 3, 'CUSTOMIZE: Add topic-specific decision points', 
                ha='center', va='center', fontsize=10, style='italic', color='red')
        
        ax.axis('off')
    
    def _create_real_example(self, ax):
        """Create real-world application example"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_title(f'{self.topic} - Real-World Application', fontsize=16, fontweight='bold')
        
        # Example scenario
        ax.text(5, 6, 'CUSTOMIZE: Real Industry Example', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow'),
                fontsize=12, fontweight='bold')
        
        ax.text(5, 4, f'Show how {self.topic} solves\na specific business problem', 
                ha='center', va='center', fontsize=10)
        
        ax.text(5, 2, 'Include metrics, outcomes, and lessons learned', 
                ha='center', va='center', fontsize=9, style='italic', color='blue')
        
        ax.axis('off')
    
    def _create_common_mistakes(self, ax):
        """Create common mistakes illustration"""
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 8)
        ax.set_title(f'{self.topic} - Common Mistakes to Avoid', fontsize=16, fontweight='bold')
        
        mistakes = [
            (2, 6, '❌', 'Mistake 1'),
            (7, 6, '❌', 'Mistake 2'), 
            (12, 6, '❌', 'Mistake 3')
        ]
        
        for x, y, symbol, text in mistakes:
            ax.text(x, y, symbol, ha='center', va='center', fontsize=20)
            ax.text(x, y-1, text, ha='center', va='center', fontsize=10, fontweight='bold')
            ax.text(x, y-2, 'CUSTOMIZE:\nSpecific mistake\ndetails', ha='center', va='center', 
                   fontsize=8, style='italic', color='red')
        
        ax.axis('off')
    
    def _create_learning_path(self, ax):
        """Create learning progression timeline"""
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 8)
        ax.set_title(f'Learning {self.topic} - Realistic Timeline', fontsize=16, fontweight='bold')
        
        # Timeline
        ax.plot([1, 15], [4, 4], 'k-', linewidth=3)
        
        phases = [
            (3, 'Months 1-2\nFoundation'),
            (6, 'Months 3-4\nCore Concepts'),
            (9, 'Months 5-6\nReal-World'),
            (12, 'Months 7-12\nSpecialization')
        ]
        
        for x, text in phases:
            ax.plot(x, 4, 'o', markersize=10, color='blue')
            ax.text(x, 6, text, ha='center', va='center', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
        
        ax.text(8, 1.5, 'Reality: 6-12 months to competency with consistent practice', 
                ha='center', va='center', fontsize=11, fontweight='bold', color='red')
        
        ax.set_ylim(0, 8)
        ax.axis('off')
    
    def _generate_simple_code_example(self) -> str:
        """Generate simple code example - customize per topic"""
        return f"""# Simple {self.topic} example
# CUSTOMIZE: Add topic-specific simple example
import pandas as pd
import numpy as np

# Basic implementation
def simple_example():
    # Add your simple example here
    print(f"This is a placeholder - customize for {self.topic}")
    return True

# Run example
result = simple_example()
print(f"Result: {{result}}")"""
    
    def _generate_realistic_code_example(self) -> str:
        """Generate realistic code example - customize per topic"""
        return f"""# Realistic {self.topic} example with real data
# CUSTOMIZE: Add topic-specific realistic example
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def realistic_example():
    # CUSTOMIZE: Add realistic example for {self.topic}
    print(f"This is a placeholder - customize for {self.topic}")
    
    # Include proper error handling
    try:
        # Your realistic implementation here
        pass
    except Exception as e:
        print(f"Error: {{e}}")
        print("Common solutions: Check data format, verify imports")
    
    return None

# Run with error handling
result = realistic_example()"""
    
    def generate_complete_content_package(self) -> Dict[str, any]:
        """Generate complete content package with visuals and bilingual scripts"""
        
        print(f"🚀 Generating complete content package for Day {self.day_number}: {self.topic}")
        
        # Create folder structure
        self.create_folder_structure()
        
        # Generate visual aids
        print("🎨 Generating visual aids...")
        images = self.generate_visual_aids()
        
        # Generate English scripts
        print("📝 Creating English video scripts...")
        english_scripts = self.create_english_video_scripts(images)
        
        # Generate Chinese scripts
        print("📝 Creating Chinese video scripts...")
        chinese_scripts = self.create_chinese_video_scripts(images)
        
        # Save all content
        self._save_content_to_files(english_scripts, chinese_scripts, images)
        
        package = {
            "visual_aids": images,
            "english_scripts": english_scripts,
            "chinese_scripts": chinese_scripts,
            "folder_structure": str(self.base_path),
            "customization_notes": self._generate_customization_guide()
        }
        
        print(f"✅ Complete content package generated in: {self.base_path}")
        print(f"📊 Generated: {len(images)} visuals, {len(english_scripts)} English scripts, {len(chinese_scripts)} Chinese scripts")
        
        return package
    
    def _save_content_to_files(self, english_scripts: Dict, chinese_scripts: Dict, images: Dict):
        """Save all generated content to appropriate files"""
        
        # Save English scripts
        english_dir = self.base_path / "scripts" / "english"
        for script_name, content in english_scripts.items():
            (english_dir / f"{script_name}.md").write_text(content, encoding='utf-8')
        
        # Save Chinese scripts
        chinese_dir = self.base_path / "scripts" / "chinese"  
        for script_name, content in chinese_scripts.items():
            (chinese_dir / f"{script_name}.md").write_text(content, encoding='utf-8')
        
        # Save image reference list
        image_list = "# Generated Visual Assets\n\n"
        for name, path in images.items():
            image_list += f"- **{name}**: {path}\n"
        
        (self.base_path / "assets" / "VISUAL_ASSETS_LIST.md").write_text(image_list)
        
        # Save customization guide
        customization_guide = self._generate_customization_guide()
        (self.base_path / "CUSTOMIZATION_GUIDE.md").write_text(customization_guide)
    
    def _generate_customization_guide(self) -> str:
        """Generate customization guide for topic-specific content"""
        return f"""# Customization Guide for Day {self.day_number}: {self.topic}

## Visual Assets Customization
All visual assets are generated with placeholder content marked as "CUSTOMIZE".
Replace these placeholders with topic-specific information:

### Image Customizations Needed:
1. **01_concept_overview.png**: Add specific {self.topic} components
2. **02_comparison.png**: Compare {self.topic} with relevant alternatives
3. **03_architecture.png**: Show {self.topic} system architecture
4. **04_step_by_step.png**: {self.topic} implementation steps
5. **05_decision_tree.png**: When to use {self.topic} decision points
6. **06_real_example.png**: Industry-specific use case
7. **07_common_mistakes.png**: {self.topic}-specific pitfalls
8. **08_learning_path.png**: {self.topic} learning progression

## Script Customizations Needed:

### English Scripts:
- Replace all "[CUSTOMIZE: ...]" placeholders
- Add topic-specific code examples
- Include real business scenarios
- Update technical explanations

### Chinese Scripts:
- Ensure cultural appropriateness of examples
- Verify technical terminology translation
- Adapt business scenarios for Chinese market
- Check character encoding (UTF-8)

## Code Examples:
- Update `_generate_simple_code_example()` 
- Update `_generate_realistic_code_example()`
- Test all code before recording
- Include proper error handling

## Production Checklist:
- [ ] All visual placeholders customized
- [ ] Code examples tested and working
- [ ] Scripts reviewed for technical accuracy
- [ ] Chinese translations reviewed by native speaker
- [ ] Visual quality checked (1920x1080 minimum)
- [ ] Accessibility features added (captions, alt-text)

## Estimated Customization Time:
- Visual updates: 2-3 hours
- Script customization: 3-4 hours
- Code testing: 1-2 hours
- Review and quality check: 1 hour

**Total: 7-10 hours per topic**
"""

def main():
    """Enhanced content creator with visual and bilingual support"""
    if len(sys.argv) < 3:
        print("Usage: python automated_content_creator_with_visuals.py <day_number> <topic>")
        print("Example: python automated_content_creator_with_visuals.py 2 'Python Environment Setup'")
        sys.exit(1)
    
    day_number = int(sys.argv[1])
    topic = sys.argv[2]
    
    # Create enhanced content creator
    creator = EnhancedContentCreator(day_number, topic)
    
    # Generate complete package
    package = creator.generate_complete_content_package()
    
    print("\n🎉 Content Generation Complete!")
    print(f"📁 All files saved to: {package['folder_structure']}")
    print(f"🎨 Visual assets: {len(package['visual_aids'])} images")
    print(f"🇺🇸 English scripts: {len(package['english_scripts'])} videos")
    print(f"🇨🇳 Chinese scripts: {len(package['chinese_scripts'])} videos")
    print(f"📋 Next: Review CUSTOMIZATION_GUIDE.md for topic-specific updates")

if __name__ == "__main__":
    main()