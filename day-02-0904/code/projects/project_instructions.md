# Day 2 Project: Build Your First ML Data Pipeline

## 🎯 Project Goal
Create a complete data processing pipeline that takes raw data and transforms it into ML-ready format.

## 📋 Requirements

### Level 1: Beginner (Complete At Least This)
1. Load a CSV file using Pandas
2. Display basic statistics about the data
3. Handle missing values
4. Create at least 2 new features
5. Save the processed data

### Level 2: Intermediate
All of Level 1, plus:
1. Implement proper error handling
2. Add data validation checks
3. Create visualizations of the data
4. Implement feature scaling
5. Split data into train/test sets

### Level 3: Advanced
All of Level 2, plus:
1. Create a reusable Pipeline class
2. Add logging functionality
3. Implement multiple preprocessing strategies
4. Add configuration file support
5. Create unit tests

## 📁 Project Structure
```
your_project/
├── data/
│   ├── raw/           # Original data files
│   └── processed/      # Processed data files
├── src/
│   ├── data_loader.py # Data loading functions
│   ├── preprocessor.py # Preprocessing functions
│   └── pipeline.py     # Main pipeline
├── tests/              # Unit tests (Level 3)
├── config.yaml         # Configuration (Level 3)
└── main.py            # Entry point
```

## 🚀 Starter Code

```python
# main.py - Your starting point
import pandas as pd
import numpy as np
from pathlib import Path

class DataPipeline:
    """Your ML data pipeline"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.data = None
        self.processed_data = None
        
    def load_data(self):
        """Load data from CSV"""
        # TODO: Implement data loading
        # Hint: Use pd.read_csv()
        pass
        
    def explore_data(self):
        """Explore and understand the data"""
        # TODO: Print shape, dtypes, missing values
        # Hint: Use .shape, .dtypes, .isnull().sum()
        pass
        
    def clean_data(self):
        """Clean the data"""
        # TODO: Handle missing values and duplicates
        # Hint: Use .dropna() or .fillna()
        pass
        
    def create_features(self):
        """Create new features"""
        # TODO: Engineer at least 2 new features
        # Example: ratios, combinations, bins
        pass
        
    def preprocess(self):
        """Complete preprocessing pipeline"""
        # TODO: Combine all steps
        self.load_data()
        self.explore_data()
        self.clean_data()
        self.create_features()
        return self.processed_data
        
    def save_processed_data(self, output_path):
        """Save processed data"""
        # TODO: Save to CSV
        # Hint: Use .to_csv()
        pass

# Example usage
if __name__ == "__main__":
    # Create pipeline
    pipeline = DataPipeline("data/raw/your_data.csv")
    
    # Process data
    processed = pipeline.preprocess()
    
    # Save results
    pipeline.save_processed_data("data/processed/clean_data.csv")
    
    print("Pipeline complete!")
```

## 📊 Sample Dataset

If you don't have your own data, create this sample dataset:

```python
# create_sample_data.py
import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

# Create sample e-commerce data
data = pd.DataFrame({
    'customer_age': np.random.randint(18, 70, n_samples),
    'account_age_days': np.random.randint(1, 1000, n_samples),
    'total_spent': np.random.exponential(100, n_samples) * 10,
    'n_purchases': np.random.poisson(5, n_samples),
    'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_samples),
    'has_premium': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'satisfaction_score': np.random.uniform(1, 5, n_samples)
})

# Add some missing values
missing_idx = np.random.choice(n_samples, 50, replace=False)
data.loc[missing_idx, 'satisfaction_score'] = np.nan

# Save
data.to_csv('sample_ecommerce_data.csv', index=False)
print(f"Created sample dataset with {len(data)} records")
```

## ✅ Submission Checklist

### Level 1 Requirements
- [ ] Data loads successfully
- [ ] Statistics are displayed clearly
- [ ] Missing values are handled
- [ ] At least 2 new features created
- [ ] Processed data is saved

### Level 2 Requirements  
- [ ] Error handling implemented
- [ ] Data validation checks pass
- [ ] Visualizations created
- [ ] Features are scaled
- [ ] Train/test split implemented

### Level 3 Requirements
- [ ] Pipeline class is reusable
- [ ] Logging provides useful information
- [ ] Multiple strategies available
- [ ] Configuration file works
- [ ] Tests pass

## 💡 Hints & Tips

1. **Start Simple**: Get Level 1 working before moving on
2. **Test Often**: Run your code after each function
3. **Use Print Statements**: Debug by printing intermediate results
4. **Check Data Types**: Many errors come from wrong types
5. **Google is Your Friend**: Look up error messages

## 🎯 Success Criteria

Your pipeline is successful if it can:
1. Take any CSV file as input
2. Handle common data issues automatically
3. Output clean, ML-ready data
4. Be reused for different datasets

## 📚 Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy User Guide](https://numpy.org/doc/stable/user/)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

## 🏆 Bonus Challenges

1. Add support for multiple file formats (Excel, JSON)
2. Implement automatic feature type detection
3. Create a summary report in HTML/PDF
4. Add data quality scoring
5. Implement parallel processing for large files

Good luck! Remember: the goal is to learn, not to be perfect. Focus on understanding each step.