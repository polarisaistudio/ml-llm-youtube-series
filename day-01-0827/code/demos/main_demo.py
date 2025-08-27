#!/usr/bin/env python3
"""
Day 1: Machine Learning Introduction - Demo Code
Traditional ML vs Modern AI Examples
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def print_section(title):
    """Helper function to print section headers"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

# ============================================================================
# PART 1: TRADITIONAL ML - LINEAR REGRESSION FOR HOUSE PRICES
# ============================================================================

def traditional_ml_demo():
    """Demonstrate traditional ML with Linear Regression"""
    
    print_section("TRADITIONAL ML: House Price Prediction")
    
    # Generate more realistic house data
    np.random.seed(42)
    
    # Features: [size_sqft, bedrooms, bathrooms, age_years]
    print("\n📊 Creating dataset with multiple features:")
    print("   - House size (square feet)")
    print("   - Number of bedrooms")
    print("   - Number of bathrooms")
    print("   - Age of house (years)")
    
    # Create 100 sample houses
    n_samples = 100
    
    # Generate features
    sizes = np.random.normal(1500, 500, n_samples)  # Avg 1500 sqft, std 500
    bedrooms = np.random.choice([2, 3, 4, 5], n_samples, p=[0.2, 0.4, 0.3, 0.1])
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    age = np.random.uniform(0, 50, n_samples)
    
    # Combine features
    X = np.column_stack([sizes, bedrooms, bathrooms, age])
    
    # Generate prices with realistic formula
    # Base price + size effect + bedroom bonus + bathroom bonus - age depreciation + noise
    base_price = 50000
    price_per_sqft = 150
    bedroom_bonus = 10000
    bathroom_bonus = 15000
    depreciation_per_year = 1000
    
    prices = (base_price + 
              sizes * price_per_sqft + 
              bedrooms * bedroom_bonus + 
              bathrooms * bathroom_bonus - 
              age * depreciation_per_year +
              np.random.normal(0, 20000, n_samples))  # Add some noise
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, prices, test_size=0.2, random_state=42
    )
    
    print(f"\n📈 Dataset created:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Train the Linear Regression model
    print("\n🤖 Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n✅ Model trained successfully!")
    print(f"   Mean Squared Error: ${mse:,.2f}")
    print(f"   R² Score: {r2:.3f} (closer to 1 is better)")
    
    # Feature importance (coefficients)
    print("\n📊 Feature Importance (How much each feature affects price):")
    feature_names = ['Size (sqft)', 'Bedrooms', 'Bathrooms', 'Age (years)']
    for name, coef in zip(feature_names, model.coef_):
        sign = "+" if coef > 0 else "-"
        print(f"   {name}: {sign}${abs(coef):,.2f}")
    
    # Make a prediction for a new house
    print("\n🏠 Let's predict a price for a new house:")
    new_house = np.array([[1650, 3, 2, 10]])  # 1650 sqft, 3 bed, 2 bath, 10 years old
    predicted_price = model.predict(new_house)[0]
    
    print(f"   House specs: 1650 sqft, 3 bedrooms, 2 bathrooms, 10 years old")
    print(f"   Predicted price: ${predicted_price:,.2f}")
    
    # Visualize predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Traditional ML: House Price Predictions vs Actual')
    plt.grid(True, alpha=0.3)
    
    # Format axis labels
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    plt.tight_layout()
    plt.savefig('day-01-0827/assets/images/traditional_ml_results.png', dpi=150)
    print("\n📊 Visualization saved to: assets/images/traditional_ml_results.png")
    plt.show()
    
    return model

# ============================================================================
# PART 2: COMPARISON WITH SIMPLE NEURAL NETWORK
# ============================================================================

def simple_neural_network_comparison():
    """Show how a simple neural network would approach the same problem"""
    
    print_section("MODERN AI APPROACH: Simple Neural Network Comparison")
    
    print("\n🧠 How would a neural network handle this?")
    print("\nKey Differences:")
    print("1. ✅ Automatic feature learning - no need to manually select features")
    print("2. ✅ Can capture non-linear relationships")
    print("3. ✅ Scales to millions of features")
    print("4. ⚠️  Requires more data to train effectively")
    print("5. ⚠️  Less interpretable ('black box')")
    print("6. ⚠️  Computationally more expensive")
    
    print("\n💡 Example Neural Network Architecture for House Prices:")
    print("""
    Input Layer (4 neurons) → [House features]
           ↓
    Hidden Layer 1 (64 neurons) → [Learn patterns]
           ↓
    Hidden Layer 2 (32 neurons) → [Combine patterns]
           ↓
    Output Layer (1 neuron) → [Predicted price]
    """)
    
    # Create a simple visualization of accuracy vs data size
    data_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    traditional_ml_accuracy = [0.3, 0.5, 0.65, 0.72, 0.75, 0.76, 0.77]
    neural_net_accuracy = [0.2, 0.35, 0.55, 0.70, 0.78, 0.85, 0.90]
    
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, traditional_ml_accuracy, 'b-o', label='Traditional ML', linewidth=2)
    plt.plot(data_sizes, neural_net_accuracy, 'r-s', label='Neural Network', linewidth=2)
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Model Accuracy')
    plt.title('Traditional ML vs Neural Networks: Data Efficiency')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axvline(x=100, color='gray', linestyle='--', alpha=0.5)
    plt.text(100, 0.8, 'Our dataset size', rotation=90, verticalalignment='bottom')
    plt.tight_layout()
    plt.savefig('day-01-0827/assets/images/ml_vs_nn_comparison.png', dpi=150)
    print("\n📊 Comparison chart saved to: assets/images/ml_vs_nn_comparison.png")
    plt.show()

# ============================================================================
# PART 3: MODERN AI - LLM EXAMPLE (SIMULATED)
# ============================================================================

def modern_ai_llm_demo():
    """Demonstrate modern AI capabilities with LLM example"""
    
    print_section("MODERN AI: Large Language Model Capabilities")
    
    print("\n🤖 Modern AI (LLMs) can understand and generate natural language:")
    
    # Simulated examples (in real implementation, would use OpenAI API)
    examples = [
        {
            "task": "Explain ML to a 5-year-old",
            "response": "Machine learning is like teaching a computer to be smart by showing it lots of examples, just like how you learn to recognize animals by looking at many pictures!"
        },
        {
            "task": "Generate Python code for data analysis",
            "response": "```python\nimport pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.describe())\n```"
        },
        {
            "task": "Translate technical concepts",
            "response": "Neural Network → 神经网络 (A system inspired by the human brain)"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n📝 Example {i}: {example['task']}")
        print(f"   Response: {example['response']}")
    
    print("\n🔑 Key Capabilities of Modern AI/LLMs:")
    print("   • Natural language understanding")
    print("   • Code generation and debugging")
    print("   • Translation and summarization")
    print("   • Creative content generation")
    print("   • Multi-modal understanding (text + images)")
    
    # API usage example (commented out to avoid requiring API key)
    print("\n💻 To use OpenAI's GPT in your code:")
    print("""
    ```python
    import openai
    
    client = openai.Client(api_key="your-key-here")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Your prompt here"}]
    )
    print(response.choices[0].message.content)
    ```
    """)

# ============================================================================
# PART 4: WHEN TO USE WHAT - DECISION GUIDE
# ============================================================================

def decision_guide():
    """Create a decision guide for ML vs AI selection"""
    
    print_section("DECISION GUIDE: Traditional ML vs Modern AI")
    
    # Create decision matrix
    scenarios = [
        ["Predicting sales numbers", "✅ Traditional ML", "❌ Overkill"],
        ["Chatbot development", "❌ Too limited", "✅ Modern AI"],
        ["Fraud detection", "✅ Traditional ML", "🔄 Both work"],
        ["Image generation", "❌ Can't do it", "✅ Modern AI"],
        ["Customer churn prediction", "✅ Traditional ML", "🔄 Both work"],
        ["Language translation", "⚠️  Basic only", "✅ Modern AI"],
        ["Stock price prediction", "✅ Traditional ML", "🔄 Both work"],
        ["Code generation", "❌ Can't do it", "✅ Modern AI"],
    ]
    
    print("\n" + "-"*60)
    print(f"{'Scenario':<30} {'Traditional ML':<20} {'Modern AI':<20}")
    print("-"*60)
    for scenario in scenarios:
        print(f"{scenario[0]:<30} {scenario[1]:<20} {scenario[2]:<20}")
    print("-"*60)
    
    print("\n📚 Rule of Thumb:")
    print("   • Structured data + clear patterns → Traditional ML")
    print("   • Unstructured data + complex patterns → Modern AI")
    print("   • Need interpretability → Traditional ML")
    print("   • Need creativity/generation → Modern AI")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all demonstrations"""
    
    print("\n")
    print("🚀 " + "="*56 + " 🚀")
    print("   DAY 1: MACHINE LEARNING - TRADITIONAL ML VS MODERN AI")
    print("🚀 " + "="*56 + " 🚀")
    
    # Run demonstrations
    try:
        # 1. Traditional ML Demo
        model = traditional_ml_demo()
        
        # 2. Neural Network Comparison
        simple_neural_network_comparison()
        
        # 3. Modern AI/LLM Demo
        modern_ai_llm_demo()
        
        # 4. Decision Guide
        decision_guide()
        
        # Summary
        print_section("SUMMARY & NEXT STEPS")
        print("\n✅ What we learned today:")
        print("   1. Machine Learning learns patterns from data")
        print("   2. Traditional ML is great for structured data")
        print("   3. Modern AI excels at unstructured data and generation")
        print("   4. Both have their place in the ML ecosystem")
        
        print("\n🎯 Your homework:")
        print("   1. Run this code and experiment with the house price data")
        print("   2. Try changing the features and see how predictions change")
        print("   3. Research one real-world application of each type")
        
        print("\n📅 Tomorrow (Day 2):")
        print("   We'll set up your Python ML environment and master essential libraries!")
        
        print("\n" + "="*60)
        print(" 🎉 Day 1 Complete! See you tomorrow! 🎉")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("Make sure you have the required packages installed:")
        print("pip install numpy matplotlib scikit-learn")

if __name__ == "__main__":
    main()