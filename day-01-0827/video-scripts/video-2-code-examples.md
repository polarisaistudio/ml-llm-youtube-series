# Video 2: Traditional ML vs Modern AI - Code Examples
**Target Length:** 15 minutes  
**Format:** Live coding with clear explanations  
**Audience:** Beginners with basic Python knowledge

## Opening (0:00-0:30)
"In the last video, we covered the concepts. Now let's see the actual code. I'm going to show you two working examples - one Traditional ML, one Modern AI - and you can follow along.

By the end of this video, you'll have run your first machine learning model and understand why the setup for Modern AI is more complex."

**[Visual: Split screen showing code editor and terminal]**

## Section 1: Traditional ML Setup (0:30-2:30)
"Let's start with Traditional ML because it requires zero setup - everything we need comes with Python.

**[Screen recording: Opening VS Code/Python environment]**

I'm using Python 3.9, but any recent version works. The beauty of Traditional ML is these libraries are usually pre-installed:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
```

If you get an error, just run:
```bash
pip install scikit-learn
```

**[Live demonstration of imports working]**

Now, the Traditional ML approach always follows the same pattern:
1. Get data
2. Split data  
3. Train model
4. Test model

Let's see this with a real example."

## Section 2: Traditional ML Example (2:30-6:00)
"We're going to build a flower classifier - it learns to identify flower species from petal and sepal measurements.

**[Live coding with explanations]**

```python
# Step 1: Get data (150 flower samples, 4 measurements each)
data, labels = load_iris(return_X_y=True)
print(f"Data shape: {data.shape}")  # (150, 4)
print(f"Unique species: {len(set(labels))}")  # 3 species

# Step 2: Split data - crucial for testing!
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.3, random_state=42
)
print(f"Training samples: {len(X_train)}")  # 105
print(f"Testing samples: {len(X_test)}")    # 45
```

**[Pause to explain train/test split concept]**

"This split is crucial - we train on 70% of data, test on the other 30%. The model never sees the test data during training, so it shows real-world performance.

```python
# Step 3: Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model trained!")

# Step 4: Test performance  
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Training accuracy: {train_accuracy:.1%}")  # Usually 100%
print(f"Testing accuracy: {test_accuracy:.1%}")    # Usually ~97%
```

**[Run the code live, show actual output]**

Notice something important - training accuracy is perfect (100%), but test accuracy is slightly lower (97%). This is normal and healthy. If they were both 100%, I'd be suspicious of overfitting."

## Section 3: Understanding the Traditional ML Results (6:00-7:30)
"Let's understand what just happened:

**[Visual: Show confusion matrix or predictions]**

```python
# Let's see some actual predictions
sample_data = X_test[:5]  # First 5 test samples
predictions = model.predict(sample_data)
actual = y_test[:5]

for i in range(5):
    print(f"Predicted: {predictions[i]}, Actual: {actual[i]}")
```

**[Show output]**

The model learned patterns like:
- If petal length > 4.5 AND petal width > 1.5 → Species 2
- If sepal length < 5.0 → Species 0

It's making decisions based on mathematical rules it derived from the training data.

**Traditional ML strengths shown here:**
✅ Fast training (instant)
✅ Predictable performance  
✅ Can explain decisions (feature importance)
✅ Works with small datasets (150 samples)"

## Section 4: Modern AI Setup Challenge (7:30-9:30)
"Now let's try Modern AI. This is where things get more complex.

**[Screen recording showing setup process]**

For Modern AI, we need:
1. An API key (costs money)
2. Internet connection
3. External service dependencies

```bash
# Install required libraries
pip install openai python-dotenv

# Set up environment variable
export OPENAI_API_KEY="your-key-here"
```

**[Show actual API key setup process - with key blurred]**

This already highlights a key difference - Traditional ML runs locally and free. Modern AI often requires cloud services.

Let's build a sentiment analyzer that understands if text is positive, negative, or neutral:

```python
from openai import OpenAI
import os

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_sentiment(text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "Analyze sentiment. Reply only with 'positive', 'negative', or 'neutral'"
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
        return f"Error: {e}"
```

**[Live coding with explanations of each parameter]**"

## Section 5: Modern AI Example (9:30-12:00)
"Let's test our sentiment analyzer:

**[Live demonstration]**

```python
# Test various examples
examples = [
    "I love this product! It exceeded my expectations!",
    "This is the worst purchase I've ever made.",
    "The item arrived on time. It works as expected.",
    "OMG this is absolutely amazing!!! 🔥🔥🔥",
    "Meh, it's okay I guess."
]

for text in examples:
    sentiment = analyze_sentiment(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")
    print("-" * 50)
```

**[Show actual API responses]**

Notice what's happening - the AI understands:
- Context and nuance
- Emotional language  
- Even emojis and slang
- Subtle differences between neutral and negative

This would be extremely difficult with Traditional ML. You'd need to manually engineer features for:
- Positive/negative word counts
- Punctuation patterns
- Capitalization
- Emoji meanings
- Contextual relationships

Modern AI learned all this automatically from massive text datasets."

## Section 6: Comparison and Costs (12:00-13:30)
"Let's compare what we just saw:

**[Visual: Side-by-side comparison table]**

**Traditional ML (Flower Classifier):**
- Setup time: 30 seconds
- Training time: Instant  
- Cost per prediction: $0
- Data needed: 150 samples worked fine
- Explainability: High (can show decision rules)

**Modern AI (Sentiment Analysis):**
- Setup time: 10 minutes (API key, etc.)
- Training time: Already trained (months of pre-training)
- Cost per prediction: ~$0.001-0.01
- Data needed: Pre-trained on billions of text samples
- Explainability: Low (black box)

**[Live demonstration of checking OpenAI API costs]**

```python
# Each API call costs money
print("Cost per request: ~$0.001")
print("If you process 1000 reviews: ~$1")
print("Traditional ML: $0 after initial setup")
```

This cost difference matters at scale."

## Section 7: When Each Approach Fails (13:30-14:30)
"Let's see what happens when we use the wrong approach:

**[Live demonstration]**

```python
# Can Traditional ML handle creative text?
# Let's try to use our flower classifier on text...
# (This will obviously fail)

text_example = "I love this product"
# We can't even pass text to our flower model!
# model.predict(text_example)  # This would crash

print("Traditional ML can't handle unstructured text without massive preprocessing")
```

And Modern AI limitations:

```python
# Can Modern AI predict house prices efficiently?
house_data = "3 bedrooms, 1500 sqft, built in 1990"
# We could use GPT, but it would be:
# - Expensive for every prediction
# - Inconsistent results  
# - No guarantee of accuracy
# - Overkill for structured data

print("Modern AI is overkill and expensive for simple structured predictions")
```

**The key insight: Use the right tool for the job.**"

## Closing & Next Video (14:30-15:00)
"You've now seen both approaches in action. The Traditional ML example you can run right now for free. The Modern AI example requires setup but shows incredible language understanding.

Next video: I'll show you exactly how to decide which approach to use for any problem you encounter. We'll cover real business scenarios and build a decision framework.

Try running this code yourself - links are in the description. If you get stuck, drop a comment and I'll help you debug.

Subscribe for the rest of this series - tomorrow we're diving into practical decision-making."

**[End screen with code repository link and subscribe button]**

---

## Production Notes:

**Code Repository Structure:**
```
/day-01-video-2/
  ├── traditional_ml_example.py
  ├── modern_ai_example.py  
  ├── requirements.txt
  └── README.md
```

**Screen Recording Setup:**
- Large font size for mobile viewing
- Dark theme for better contrast
- Terminal and code editor side-by-side
- Cursor highlighting for following along

**Common Beginner Issues to Address:**
- Python version compatibility
- Library installation problems
- API key setup confusion
- Environment variable setup on different OS

**Engagement Elements:**
- Pause for questions at 7:00
- "Try this yourself" moments
- Code download links
- Troubleshooting help offer