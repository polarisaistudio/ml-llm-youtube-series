#!/usr/bin/env python3
"""
Generate Educational Visualizations for Day 1: ML Theory
Creates clean, professional diagrams for video presentation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style for clean, professional visuals
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
output_dir = Path("day-01-0827/assets/images/theory")
output_dir.mkdir(parents=True, exist_ok=True)

def create_traditional_vs_ml_diagram():
    """Create comparison diagram: Traditional Programming vs Machine Learning"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Traditional Programming
    ax1.set_title("Traditional Programming", fontsize=16, fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Boxes for traditional
    input_box = FancyBboxPatch((1, 7), 2.5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightblue', 
                               edgecolor='darkblue', linewidth=2)
    rules_box = FancyBboxPatch((1, 4), 2.5, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor='lightgreen',
                              edgecolor='darkgreen', linewidth=2)
    output_box = FancyBboxPatch((6, 5.5), 2.5, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor='lightyellow',
                               edgecolor='orange', linewidth=2)
    
    ax1.add_patch(input_box)
    ax1.add_patch(rules_box)
    ax1.add_patch(output_box)
    
    # Labels
    ax1.text(2.25, 7.75, 'INPUT\nData', ha='center', va='center', fontsize=11, fontweight='bold')
    ax1.text(2.25, 4.75, 'RULES\nif/then logic', ha='center', va='center', fontsize=11, fontweight='bold')
    ax1.text(7.25, 6.25, 'OUTPUT\nResult', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Arrows
    arrow1 = FancyArrowPatch((3.5, 7.5), (6, 6.5),
                            arrowstyle='->', lw=2, color='black',
                            connectionstyle="arc3,rad=0.3")
    arrow2 = FancyArrowPatch((3.5, 5), (6, 6),
                            arrowstyle='->', lw=2, color='black',
                            connectionstyle="arc3,rad=-0.3")
    ax1.add_patch(arrow1)
    ax1.add_patch(arrow2)
    
    # Plus sign
    ax1.text(4.5, 6, '+', fontsize=24, ha='center', va='center')
    
    # Example
    ax1.text(5, 2, 'Example:\nif temperature > 30°C:\n    return "hot"\nelse:\n    return "cold"',
            ha='center', va='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Machine Learning
    ax2.set_title("Machine Learning", fontsize=16, fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Boxes for ML
    input_ml = FancyBboxPatch((1, 7), 2.5, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor='lightblue',
                             edgecolor='darkblue', linewidth=2)
    output_ml = FancyBboxPatch((1, 4), 2.5, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor='lightyellow',
                              edgecolor='orange', linewidth=2)
    model_box = FancyBboxPatch((6, 5.5), 2.5, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor='lightcoral',
                              edgecolor='darkred', linewidth=2)
    
    ax2.add_patch(input_ml)
    ax2.add_patch(output_ml)
    ax2.add_patch(model_box)
    
    # Labels
    ax2.text(2.25, 7.75, 'INPUT\nExamples', ha='center', va='center', fontsize=11, fontweight='bold')
    ax2.text(2.25, 4.75, 'OUTPUT\nLabels', ha='center', va='center', fontsize=11, fontweight='bold')
    ax2.text(7.25, 6.25, 'MODEL\nLearned Rules', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Arrows
    arrow3 = FancyArrowPatch((3.5, 7.5), (6, 6.5),
                            arrowstyle='->', lw=2, color='black',
                            connectionstyle="arc3,rad=0.3")
    arrow4 = FancyArrowPatch((3.5, 5), (6, 6),
                            arrowstyle='->', lw=2, color='black',
                            connectionstyle="arc3,rad=-0.3")
    ax2.add_patch(arrow3)
    ax2.add_patch(arrow4)
    
    # Plus sign
    ax2.text(4.5, 6, '+', fontsize=24, ha='center', va='center')
    
    # Example
    ax2.text(5, 2, 'Example:\nShow many temperatures\nwith "hot"/"cold" labels\n→ Learn the pattern',
            ha='center', va='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle("Programming Paradigms Comparison", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "01_traditional_vs_ml.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: Traditional vs ML comparison diagram")

def create_ml_algorithms_visualization():
    """Create visualization of common ML algorithms"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Linear Regression
    ax = axes[0, 0]
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + np.random.normal(0, 2, 50)
    ax.scatter(x, y, alpha=0.5, color='blue', s=30)
    ax.plot(x, 2*x + 1, 'r-', lw=2, label='Linear Regression')
    ax.set_title("Linear Regression", fontsize=14, fontweight='bold')
    ax.set_xlabel("Feature (e.g., House Size)")
    ax.set_ylabel("Target (e.g., Price)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Decision Tree
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title("Decision Tree", fontsize=14, fontweight='bold')
    
    # Draw tree structure
    def draw_node(ax, x, y, text, color='lightblue'):
        circle = Circle((x, y), 0.5, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Root
    draw_node(ax, 5, 8, 'Age>30?')
    # Level 2
    draw_node(ax, 3, 5, 'Income?', 'lightgreen')
    draw_node(ax, 7, 5, 'Credit?', 'lightgreen')
    # Level 3
    draw_node(ax, 2, 2, 'Approve', 'lightcoral')
    draw_node(ax, 4, 2, 'Deny', 'lightyellow')
    draw_node(ax, 6, 2, 'Approve', 'lightcoral')
    draw_node(ax, 8, 2, 'Review', 'lightyellow')
    
    # Connect nodes
    connections = [(5, 8, 3, 5), (5, 8, 7, 5),
                  (3, 5, 2, 2), (3, 5, 4, 2),
                  (7, 5, 6, 2), (7, 5, 8, 2)]
    for x1, y1, x2, y2 in connections:
        ax.plot([x1, x2], [y1, y2], 'k-', lw=1.5)
    
    ax.text(3.5, 6.5, 'Yes', fontsize=9)
    ax.text(6.5, 6.5, 'No', fontsize=9)
    
    # SVM
    ax = axes[1, 0]
    np.random.seed(42)
    # Generate two classes
    class1_x = np.random.normal(2, 1, 50)
    class1_y = np.random.normal(2, 1, 50)
    class2_x = np.random.normal(5, 1, 50)
    class2_y = np.random.normal(5, 1, 50)
    
    ax.scatter(class1_x, class1_y, c='blue', label='Class A', alpha=0.6)
    ax.scatter(class2_x, class2_y, c='red', label='Class B', alpha=0.6)
    
    # Draw decision boundary
    x_line = np.linspace(0, 7, 100)
    y_line = -x_line + 7
    ax.plot(x_line, y_line, 'g-', lw=2, label='Decision Boundary')
    ax.fill_between(x_line, y_line-0.5, y_line+0.5, alpha=0.2, color='green')
    
    ax.set_title("Support Vector Machine", fontsize=14, fontweight='bold')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 8)
    ax.grid(True, alpha=0.3)
    
    # K-Means Clustering
    ax = axes[1, 1]
    np.random.seed(42)
    # Generate clusters
    cluster1_x = np.random.normal(2, 0.5, 30)
    cluster1_y = np.random.normal(6, 0.5, 30)
    cluster2_x = np.random.normal(6, 0.5, 30)
    cluster2_y = np.random.normal(6, 0.5, 30)
    cluster3_x = np.random.normal(4, 0.5, 30)
    cluster3_y = np.random.normal(2, 0.5, 30)
    
    ax.scatter(cluster1_x, cluster1_y, c='purple', label='Cluster 1', alpha=0.6, s=50)
    ax.scatter(cluster2_x, cluster2_y, c='orange', label='Cluster 2', alpha=0.6, s=50)
    ax.scatter(cluster3_x, cluster3_y, c='green', label='Cluster 3', alpha=0.6, s=50)
    
    # Mark centroids
    ax.scatter([2, 6, 4], [6, 6, 2], c='black', marker='X', s=200, 
              edgecolor='white', linewidth=2, label='Centroids')
    
    ax.set_title("K-Means Clustering", fontsize=14, fontweight='bold')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Traditional ML Algorithms", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "02_ml_algorithms.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: ML algorithms visualization")

def create_neural_network_diagram():
    """Create a simple neural network architecture diagram"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Simple Neural Network
    ax1.set_title("Neural Network Architecture", fontsize=16, fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Define layer positions
    layers = [
        (2, [3, 5, 7]),  # Input layer (3 neurons)
        (4, [2, 3.5, 5, 6.5, 8]),  # Hidden layer 1 (5 neurons)
        (6, [3, 5, 7]),  # Hidden layer 2 (3 neurons)
        (8, [5])  # Output layer (1 neuron)
    ]
    
    # Draw neurons
    for layer_x, neurons_y in layers:
        for neuron_y in neurons_y:
            circle = Circle((layer_x, neuron_y), 0.3, 
                          facecolor='lightblue', edgecolor='darkblue', linewidth=2)
            ax1.add_patch(circle)
    
    # Draw connections
    for i in range(len(layers) - 1):
        layer1_x, layer1_neurons = layers[i]
        layer2_x, layer2_neurons = layers[i + 1]
        for n1_y in layer1_neurons:
            for n2_y in layer2_neurons:
                ax1.plot([layer1_x, layer2_x], [n1_y, n2_y], 
                        'gray', alpha=0.3, lw=0.5)
    
    # Labels
    ax1.text(2, 1, 'Input Layer', ha='center', fontsize=12, fontweight='bold')
    ax1.text(4, 1, 'Hidden Layer 1', ha='center', fontsize=12, fontweight='bold')
    ax1.text(6, 1, 'Hidden Layer 2', ha='center', fontsize=12, fontweight='bold')
    ax1.text(8, 1, 'Output Layer', ha='center', fontsize=12, fontweight='bold')
    
    # Input/Output labels
    ax1.text(1, 5, 'Features\n(Input)', ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat'))
    ax1.text(9, 5, 'Prediction\n(Output)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Deep Learning Comparison
    ax2.set_title("Traditional ML vs Deep Learning", fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    # Create comparison text
    comparison_text = """
    Traditional ML:
    • Hand-crafted features
    • Shallow models (1-2 layers)
    • Fast training
    • Good for structured data
    • Interpretable
    
    Deep Learning:
    • Automatic feature learning
    • Deep models (many layers)
    • Slow training (needs GPU)
    • Excels at unstructured data
    • Black box
    """
    
    ax2.text(0.5, 0.5, comparison_text, ha='center', va='center',
            fontsize=12, transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "03_neural_network.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: Neural network diagram")

def create_feature_engineering_illustration():
    """Create feature engineering visualization"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Raw Data to Features
    ax1.set_title("Feature Engineering Process", fontsize=16, fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Raw data box
    raw_box = FancyBboxPatch((0.5, 3), 2, 4,
                             boxstyle="round,pad=0.1",
                             facecolor='lightgray',
                             edgecolor='black', linewidth=2)
    ax1.add_patch(raw_box)
    ax1.text(1.5, 5, 'Raw Data\n\nHouse:\n- Address\n- Description\n- Images\n- History',
            ha='center', va='center', fontsize=9)
    
    # Feature extraction arrow
    arrow = FancyArrowPatch((2.5, 5), (4.5, 5),
                           arrowstyle='->', lw=3, color='green')
    ax1.add_patch(arrow)
    ax1.text(3.5, 6, 'Feature\nEngineering', ha='center', fontsize=10, fontweight='bold')
    
    # Engineered features box
    features_box = FancyBboxPatch((4.5, 3), 2.5, 4,
                                 boxstyle="round,pad=0.1",
                                 facecolor='lightgreen',
                                 edgecolor='darkgreen', linewidth=2)
    ax1.add_patch(features_box)
    ax1.text(5.75, 5, 'Features\n\n• Size (sqft)\n• Bedrooms\n• Location score\n• Age (years)\n• Price/sqft nearby',
            ha='center', va='center', fontsize=9)
    
    # ML model arrow
    arrow2 = FancyArrowPatch((7, 5), (8.5, 5),
                            arrowstyle='->', lw=3, color='blue')
    ax1.add_patch(arrow2)
    
    # Model box
    model_box = FancyBboxPatch((8.5, 4.5), 1.2, 1,
                              boxstyle="round,pad=0.1",
                              facecolor='lightblue',
                              edgecolor='darkblue', linewidth=2)
    ax1.add_patch(model_box)
    ax1.text(9.1, 5, 'ML\nModel', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Automatic Feature Learning
    ax2.set_title("Deep Learning: Automatic Feature Learning", fontsize=16, fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Raw data
    raw_box2 = FancyBboxPatch((0.5, 3), 2, 4,
                              boxstyle="round,pad=0.1",
                              facecolor='lightgray',
                              edgecolor='black', linewidth=2)
    ax2.add_patch(raw_box2)
    ax2.text(1.5, 5, 'Raw Data\n\nImages\nText\nAudio\nVideo',
            ha='center', va='center', fontsize=9)
    
    # Direct arrow to deep network
    arrow3 = FancyArrowPatch((2.5, 5), (4, 5),
                            arrowstyle='->', lw=3, color='purple')
    ax2.add_patch(arrow3)
    
    # Deep network representation
    for x in [4, 5, 6, 7, 8]:
        for y in [3, 4, 5, 6, 7]:
            circle = Circle((x, y), 0.15, facecolor='mediumpurple', 
                          edgecolor='purple', alpha=0.7)
            ax2.add_patch(circle)
        if x < 8:
            for y1 in [3, 4, 5, 6, 7]:
                for y2 in [3, 4, 5, 6, 7]:
                    ax2.plot([x, x+1], [y1, y2], 'purple', alpha=0.1, lw=0.5)
    
    ax2.text(6, 8, 'Deep Neural Network', ha='center', fontsize=12, fontweight='bold')
    ax2.text(6, 2, 'Learns features automatically!', ha='center', fontsize=11,
            style='italic', color='purple')
    
    # Output
    output_box = FancyBboxPatch((8.5, 4.5), 1.2, 1,
                               boxstyle="round,pad=0.1",
                               facecolor='lightyellow',
                               edgecolor='orange', linewidth=2)
    ax2.add_patch(output_box)
    ax2.text(9.1, 5, 'Output', ha='center', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "04_feature_engineering.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: Feature engineering illustration")

def create_ml_timeline():
    """Create ML evolution timeline"""
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Timeline data
    events = [
        (1950, "Turing Test", "Alan Turing proposes test for machine intelligence"),
        (1957, "Perceptron", "First neural network implementation"),
        (1980, "Expert Systems", "Rule-based AI dominates"),
        (1990, "SVM & Random Forests", "Traditional ML flourishes"),
        (2006, "Deep Learning Revival", "Hinton's deep belief networks"),
        (2012, "AlexNet", "Deep learning wins ImageNet"),
        (2014, "GANs", "Generative Adversarial Networks"),
        (2017, "Transformers", "Attention is all you need"),
        (2018, "BERT", "Bidirectional language understanding"),
        (2020, "GPT-3", "175B parameter language model"),
        (2022, "ChatGPT", "AI goes mainstream"),
        (2023, "GPT-4", "Multimodal AI"),
        (2024, "Claude 3", "Constitutional AI advances")
    ]
    
    # Extract years and labels
    years = [e[0] for e in events]
    labels = [e[1] for e in events]
    descriptions = [e[2] for e in events]
    
    # Create timeline
    ax.set_xlim(1945, 2025)
    ax.set_ylim(-2, 2)
    
    # Draw main timeline
    ax.axhline(y=0, color='black', linewidth=2)
    
    # Add events
    for i, (year, label, desc) in enumerate(events):
        # Alternate above and below timeline
        y_pos = 0.5 if i % 2 == 0 else -0.5
        y_text = 1.2 if i % 2 == 0 else -1.2
        
        # Draw marker
        ax.scatter(year, 0, s=100, c='red' if year >= 2012 else 'blue', 
                  zorder=5, edgecolor='black', linewidth=2)
        
        # Draw connection line
        ax.plot([year, year], [0, y_pos], 'gray', linestyle='--', alpha=0.5)
        
        # Add label
        ax.text(year, y_text, f"{label}\n({year})", 
               ha='center', va='center' if i % 2 == 0 else 'top',
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat' if year >= 2012 else 'lightblue',
                        alpha=0.8))
    
    # Era labels
    ax.text(1975, -1.8, "Traditional AI Era", ha='center', fontsize=12, 
           style='italic', color='blue')
    ax.text(2018, -1.8, "Deep Learning Era", ha='center', fontsize=12,
           style='italic', color='red')
    
    # Title and labels
    ax.set_title("Evolution of Machine Learning & AI", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "05_ml_timeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: ML timeline")

def create_decision_flowchart():
    """Create decision flowchart for choosing ML vs AI"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define flowchart elements
    def draw_box(x, y, w, h, text, color='lightblue'):
        box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                             boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    def draw_diamond(x, y, w, h, text, color='yellow'):
        points = [(x-w/2, y), (x, y+h/2), (x+w/2, y), (x, y-h/2)]
        diamond = mpatches.Polygon(points, closed=True, 
                                  facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    def draw_arrow(x1, y1, x2, y2, label=''):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', lw=2, color='black')
        ax.add_patch(arrow)
        if label:
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            ax.text(mid_x, mid_y, label, fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Start
    draw_box(5, 9, 2, 0.8, "Start", 'lightgreen')
    
    # First decision
    draw_arrow(5, 8.6, 5, 7.5)
    draw_diamond(5, 7, 2.5, 1, "Structured\nData?")
    
    # Structured data path (left)
    draw_arrow(4, 6.5, 3, 5.5, "Yes")
    draw_diamond(3, 5, 2, 0.8, "Need\nInterpretability?")
    
    draw_arrow(2, 4.6, 1.5, 3.5, "Yes")
    draw_box(1.5, 3, 1.8, 0.8, "Traditional ML\n(Linear, Tree)", 'lightblue')
    
    draw_arrow(4, 4.6, 4.5, 3.5, "No")
    draw_diamond(4.5, 3, 1.8, 0.8, "Large\nDataset?")
    
    draw_arrow(3.5, 2.6, 2.5, 1.5, "No")
    draw_box(2.5, 1, 1.8, 0.8, "Traditional ML\n(SVM, RF)", 'lightblue')
    
    draw_arrow(5.5, 2.6, 6.5, 1.5, "Yes")
    draw_box(6.5, 1, 1.8, 0.8, "Deep Learning\n(Neural Net)", 'lightcoral')
    
    # Unstructured data path (right)
    draw_arrow(6, 6.5, 7, 5.5, "No")
    draw_diamond(7, 5, 2, 0.8, "Generation\nTask?")
    
    draw_arrow(6, 4.6, 5.5, 3.5, "No")
    draw_box(5.5, 3, 1.8, 0.8, "Deep Learning\n(CNN, RNN)", 'lightcoral')
    
    draw_arrow(8, 4.6, 8.5, 3.5, "Yes")
    draw_box(8.5, 3, 1.8, 0.8, "Modern AI\n(GPT, DALL-E)", 'mediumpurple')
    
    # Title
    ax.text(5, 9.8, "ML vs AI Decision Guide", fontsize=16, fontweight='bold', ha='center')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='lightblue', label='Traditional ML'),
        mpatches.Patch(color='lightcoral', label='Deep Learning'),
        mpatches.Patch(color='mediumpurple', label='Modern AI/LLMs')
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=3, frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "06_decision_flowchart.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: Decision flowchart")

def create_data_types_comparison():
    """Create visualization comparing structured vs unstructured data"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # Structured Data
    ax1.set_title("Structured Data", fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Create table representation
    data = [
        ["ID", "Size", "Rooms", "Age", "Price"],
        ["001", "1500", "3", "5", "$300K"],
        ["002", "2100", "4", "2", "$450K"],
        ["003", "1200", "2", "10", "$250K"],
        ["004", "1800", "3", "3", "$380K"],
        ["005", "2500", "5", "1", "$550K"]
    ]
    
    # Draw table
    table = ax1.table(cellText=data, loc='center', cellLoc='center',
                     colWidths=[0.15]*5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Add characteristics
    characteristics = """
    ✓ Organized in rows and columns
    ✓ Fixed schema/structure
    ✓ Easy to analyze with SQL
    ✓ Perfect for traditional ML
    ✓ Examples: Databases, Spreadsheets, CSV files
    """
    ax1.text(0.5, 0.15, characteristics, ha='center', va='top',
            transform=ax1.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # Unstructured Data
    ax2.set_title("Unstructured Data", fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    # Add visual representations
    # Text example
    ax2.text(0.5, 0.85, "📝 Text", ha='center', fontweight='bold', 
            transform=ax2.transAxes, fontsize=12)
    ax2.text(0.5, 0.78, '"The house has a beautiful garden..."', 
            ha='center', style='italic', transform=ax2.transAxes, fontsize=9)
    
    # Image example
    ax2.text(0.2, 0.6, "🖼️ Images", ha='center', fontweight='bold',
            transform=ax2.transAxes, fontsize=12)
    
    # Audio example
    ax2.text(0.5, 0.6, "🎵 Audio", ha='center', fontweight='bold',
            transform=ax2.transAxes, fontsize=12)
    
    # Video example
    ax2.text(0.8, 0.6, "🎬 Video", ha='center', fontweight='bold',
            transform=ax2.transAxes, fontsize=12)
    
    # Draw waveform for audio
    x_wave = np.linspace(0.35, 0.65, 100)
    y_wave = 0.5 + 0.05 * np.sin(10 * np.pi * x_wave)
    ax2.plot(x_wave, y_wave, 'blue', alpha=0.7, transform=ax2.transAxes)
    
    # Add characteristics
    characteristics2 = """
    ✓ No predefined structure
    ✓ Various formats and types
    ✓ Requires preprocessing
    ✓ Perfect for deep learning
    ✓ 80% of world's data
    """
    ax2.text(0.5, 0.25, characteristics2, ha='center', va='top',
            transform=ax2.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    plt.suptitle("Structured vs Unstructured Data", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "07_data_types.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: Data types comparison")

def create_ml_pipeline():
    """Create ML pipeline visualization"""
    
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Pipeline stages
    stages = [
        (1, "Data\nCollection", 'lightblue'),
        (3, "Data\nCleaning", 'lightgreen'),
        (5, "Feature\nEngineering", 'yellow'),
        (7, "Model\nTraining", 'lightcoral'),
        (9, "Model\nEvaluation", 'orange'),
        (11, "Deployment", 'mediumpurple')
    ]
    
    # Draw pipeline boxes and connections
    for i, (x, text, color) in enumerate(stages):
        # Box
        box = FancyBboxPatch((x-0.6, 2), 1.2, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, 2.75, text, ha='center', va='center', 
               fontsize=10, fontweight='bold')
        
        # Arrow to next stage
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((x+0.6, 2.75), (x+1.4, 2.75),
                                   arrowstyle='->', lw=2, color='black')
            ax.add_patch(arrow)
    
    # Add details below each stage
    details = [
        "• Gather data\n• APIs/Databases\n• Web scraping",
        "• Handle missing\n• Remove outliers\n• Fix errors",
        "• Create features\n• Normalize\n• Encode categorical",
        "• Split data\n• Choose algorithm\n• Train model",
        "• Test accuracy\n• Cross-validate\n• Tune parameters",
        "• Production\n• Monitor\n• Update"
    ]
    
    for (x, _, _), detail in zip(stages, details):
        ax.text(x, 1, detail, ha='center', va='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Title
    ax.text(6, 5, "Complete Machine Learning Pipeline", 
           fontsize=18, fontweight='bold', ha='center')
    
    # Iterative arrow
    curved_arrow = FancyArrowPatch((10.5, 1.5), (1.5, 1.5),
                                  arrowstyle='->', lw=1.5, color='gray',
                                  connectionstyle="arc3,rad=-0.5", 
                                  linestyle='--', alpha=0.5)
    ax.add_patch(curved_arrow)
    ax.text(6, 0.5, "Iterative Process", ha='center', fontsize=9, 
           style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(output_dir / "08_ml_pipeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: ML pipeline diagram")

def main():
    """Generate all visualizations"""
    print("\n🎨 Generating Theory Visualizations for Day 1\n" + "="*50)
    
    # Create all visualizations
    create_traditional_vs_ml_diagram()
    create_ml_algorithms_visualization()
    create_neural_network_diagram()
    create_feature_engineering_illustration()
    create_ml_timeline()
    create_decision_flowchart()
    create_data_types_comparison()
    create_ml_pipeline()
    
    print("\n" + "="*50)
    print(f"✨ All visualizations created successfully!")
    print(f"📁 Location: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"  • {file.name}")

if __name__ == "__main__":
    main()