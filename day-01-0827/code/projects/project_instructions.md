# Day 1 Project: Build Your First ML Classifier
## Hands-on Project: Email Spam Detector

### 🎯 Project Goal
Build a simple spam email classifier using traditional ML to understand the complete ML workflow:
1. Data preparation
2. Feature extraction
3. Model training
4. Evaluation
5. Making predictions

### 📋 Prerequisites
- Python 3.8+
- Basic Python knowledge (variables, functions, loops)
- Packages listed in requirements.txt

### 🚀 Getting Started

#### Step 1: Install Requirements
```bash
pip install -r requirements.txt
```

#### Step 2: Run the Setup Script
```bash
python setup_project.py
```

### 📝 Project Tasks

#### Task 1: Data Exploration (10 minutes)
- Load the email dataset
- Explore the structure
- Check class distribution
- View sample emails

#### Task 2: Text Preprocessing (15 minutes)
- Convert text to lowercase
- Remove special characters
- Remove stop words
- Tokenize emails

#### Task 3: Feature Engineering (15 minutes)
- Convert text to numerical features using TF-IDF
- Understand what TF-IDF does
- Visualize feature importance

#### Task 4: Model Training (10 minutes)
- Split data into train/test sets
- Train a Naive Bayes classifier
- Train a Logistic Regression classifier
- Compare performance

#### Task 5: Model Evaluation (10 minutes)
- Calculate accuracy
- Generate confusion matrix
- Understand precision vs recall
- Test on new emails

#### Task 6: Build a Simple App (15 minutes)
- Create a function to classify new emails
- Build a simple command-line interface
- Test with your own examples

### 📂 Project Structure
```
day-01-project/
├── data/
│   ├── spam_emails.csv     # Dataset
│   └── sample_emails.txt    # Test emails
├── src/
│   ├── data_loader.py      # Load and explore data
│   ├── preprocessor.py     # Text preprocessing
│   ├── trainer.py          # Model training
│   └── classifier.py       # Final classifier
├── notebooks/
│   └── exploration.ipynb   # Jupyter notebook for exploration
├── models/
│   └── spam_classifier.pkl # Saved model
├── requirements.txt        # Dependencies
├── setup_project.py       # Setup script
└── main.py               # Main application
```

### 💻 Code Walkthrough

#### Step-by-Step Implementation

**1. Data Loading (data_loader.py)**
```python
import pandas as pd
import numpy as np

def load_email_data():
    """Load the spam email dataset"""
    # We'll use a simple CSV with 'text' and 'label' columns
    df = pd.read_csv('data/spam_emails.csv')
    print(f"Loaded {len(df)} emails")
    print(f"Spam emails: {sum(df['label'] == 'spam')}")
    print(f"Ham emails: {sum(df['label'] == 'ham')}")
    return df
```

**2. Text Preprocessing (preprocessor.py)**
```python
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    """Clean and preprocess email text"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def extract_features(emails):
    """Convert emails to TF-IDF features"""
    vectorizer = TfidfVectorizer(max_features=1000)
    features = vectorizer.fit_transform(emails)
    return features, vectorizer
```

**3. Model Training (trainer.py)**
```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

def train_classifier(X, y):
    """Train a spam classifier"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model Accuracy: {accuracy:.2%}")
    return model
```

**4. Main Application (main.py)**
```python
import pickle
from data_loader import load_email_data
from preprocessor import clean_text, extract_features
from trainer import train_classifier

def classify_email(email_text, model, vectorizer):
    """Classify a single email"""
    # Preprocess
    cleaned = clean_text(email_text)
    # Transform to features
    features = vectorizer.transform([cleaned])
    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return prediction, max(probability)

def main():
    print("🚀 Training Spam Classifier...")
    
    # Load data
    df = load_email_data()
    
    # Preprocess
    df['cleaned'] = df['text'].apply(clean_text)
    
    # Extract features
    X, vectorizer = extract_features(df['cleaned'])
    y = df['label']
    
    # Train model
    model = train_classifier(X, y)
    
    # Save model
    with open('models/spam_classifier.pkl', 'wb') as f:
        pickle.dump((model, vectorizer), f)
    
    print("\n✅ Model trained and saved!")
    
    # Interactive testing
    print("\n📧 Test the classifier (type 'quit' to exit):")
    while True:
        email = input("\nEnter an email: ")
        if email.lower() == 'quit':
            break
        
        label, confidence = classify_email(email, model, vectorizer)
        print(f"Classification: {label.upper()} (confidence: {confidence:.2%})")

if __name__ == "__main__":
    main()
```

### 🎯 Challenge Tasks

1. **Easy**: Modify the cleaning function to keep important punctuation
2. **Medium**: Add a new feature: email length
3. **Hard**: Implement cross-validation to better evaluate the model
4. **Expert**: Build a web interface using Flask

### 📊 Expected Results

After completing this project, you should:
- Achieve ~95% accuracy on the test set
- Understand the complete ML pipeline
- Be able to classify your own emails
- Have a working spam detector!

### 🐛 Common Issues & Solutions

**Issue**: ImportError for sklearn
**Solution**: Run `pip install scikit-learn`

**Issue**: Low accuracy
**Solution**: Check if text preprocessing is working correctly

**Issue**: Memory error with large datasets
**Solution**: Reduce max_features in TfidfVectorizer

### 📚 Further Learning

- Try different algorithms (SVM, Random Forest)
- Experiment with different text representations (Count Vectorizer, Word2Vec)
- Add more features (sender information, email metadata)
- Deploy your model as a web service

### ✅ Submission Checklist

- [ ] All code files are complete
- [ ] Model achieves >90% accuracy
- [ ] Can classify new emails
- [ ] Code is well-commented
- [ ] README is updated with your results

---

**Need help?** Check the solution branch or ask in the Discord community!

**Next Step**: Tomorrow we'll dive deeper into Python for ML and explore essential libraries!