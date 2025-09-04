#!/usr/bin/env python3
"""
Day 2: Python Fundamentals for ML - Main Demo
Complete demonstration of essential Python concepts for machine learning
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import matplotlib.pyplot as plt

def numpy_fundamentals():
    """Demonstrate NumPy basics for ML"""
    print("=" * 50)
    print("NUMPY FUNDAMENTALS")
    print("=" * 50)
    
    # Creating arrays
    arr_1d = np.array([1, 2, 3, 4, 5])
    arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    print(f"1D Array: {arr_1d}")
    print(f"2D Array:\n{arr_2d}")
    
    # Basic operations
    print(f"\nMean: {np.mean(arr_1d)}")
    print(f"Std Dev: {np.std(arr_1d)}")
    print(f"Min/Max: {np.min(arr_1d)}/{np.max(arr_1d)}")
    
    # Reshaping - crucial for ML
    reshaped = arr_1d.reshape(-1, 1)  # Common for sklearn
    print(f"\nReshaped for ML (column vector):\n{reshaped}")
    
    # Broadcasting - powerful feature
    scaled = arr_1d * 2 + 1
    print(f"\nBroadcasting (x*2 + 1): {scaled}")
    
    # Random numbers - for ML experiments
    np.random.seed(42)
    random_data = np.random.randn(5)
    print(f"\nRandom normal data: {random_data}")
    
    return arr_2d

def pandas_fundamentals():
    """Demonstrate Pandas basics for ML"""
    print("\n" + "=" * 50)
    print("PANDAS FUNDAMENTALS")
    print("=" * 50)
    
    # Creating DataFrames
    data = {
        'age': [25, 30, 35, 28, 42],
        'salary': [50000, 60000, 75000, 55000, 90000],
        'department': ['IT', 'HR', 'IT', 'Sales', 'IT'],
        'years_exp': [2, 5, 8, 4, 15]
    }
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    
    # Basic statistics
    print("\nStatistical Summary:")
    print(df.describe())
    
    # Data selection
    print("\nSelecting numerical columns:")
    numerical_cols = df.select_dtypes(include=[np.number])
    print(numerical_cols.columns.tolist())
    
    # Grouping - common in feature engineering
    print("\nAverage salary by department:")
    print(df.groupby('department')['salary'].mean())
    
    # Creating new features
    df['salary_per_year_exp'] = df['salary'] / df['years_exp']
    df['is_senior'] = df['years_exp'] > 5
    
    print("\nDataFrame with new features:")
    print(df)
    
    return df

def data_preprocessing_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Demonstrate a complete preprocessing pipeline"""
    print("\n" + "=" * 50)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 50)
    
    # Make a copy to avoid modifying original
    df_processed = df.copy()
    
    # 1. Handle categorical variables
    df_processed = pd.get_dummies(df_processed, columns=['department'], prefix='dept')
    print("After one-hot encoding:")
    print(df_processed.head())
    
    # 2. Feature scaling (normalization)
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    numerical_features = ['age', 'salary', 'years_exp', 'salary_per_year_exp']
    df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])
    
    print("\nAfter scaling:")
    print(df_processed.head())
    
    # 3. Feature statistics
    stats = {
        'n_features': len(df_processed.columns),
        'n_numerical': len(numerical_features),
        'n_categorical': len([col for col in df_processed.columns if 'dept_' in col]),
        'n_boolean': len([col for col in df_processed.columns if df_processed[col].dtype == bool])
    }
    
    print("\nPipeline Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return df_processed, stats

def ml_ready_data_loader(filepath: Optional[str] = None) -> pd.DataFrame:
    """Template function for loading and preparing ML data"""
    print("\n" + "=" * 50)
    print("ML-READY DATA LOADER TEMPLATE")
    print("=" * 50)
    
    # For demo, create synthetic data
    if filepath is None:
        print("Creating synthetic data for demonstration...")
        np.random.seed(42)
        n_samples = 100
        
        synthetic_data = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples) * 2 + 1,
            'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.choice([0, 1], n_samples)
        })
        data = synthetic_data
    else:
        # Load real data
        try:
            data = pd.read_csv(filepath)
            print(f"Loaded {len(data)} records from {filepath}")
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    
    # Data quality checks
    print("\nData Quality Report:")
    print(f"  Shape: {data.shape}")
    print(f"  Missing values: {data.isnull().sum().sum()}")
    print(f"  Duplicates: {data.duplicated().sum()}")
    print(f"  Memory usage: {data.memory_usage().sum() / 1024:.2f} KB")
    
    # Basic preprocessing
    data = data.dropna()
    data = data.drop_duplicates()
    
    print(f"\nAfter cleaning: {data.shape}")
    
    return data

def visualize_data_distribution(df: pd.DataFrame):
    """Create visualizations for data understanding"""
    print("\n" + "=" * 50)
    print("DATA VISUALIZATION")
    print("=" * 50)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns[:4]
    
    if len(numerical_cols) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.ravel()
        
        for idx, col in enumerate(numerical_cols[:4]):
            axes[idx].hist(df[col], bins=20, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('Data Distribution Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save the figure
        output_path = 'day-02-0904/assets/images/data_distributions.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        plt.close()

def main():
    """Run all demonstrations"""
    print("\n🚀 DAY 2: PYTHON FUNDAMENTALS FOR ML - COMPLETE DEMO")
    print("=" * 60)
    
    # 1. NumPy fundamentals
    numpy_array = numpy_fundamentals()
    
    # 2. Pandas fundamentals
    pandas_df = pandas_fundamentals()
    
    # 3. Data preprocessing pipeline
    processed_df, pipeline_stats = data_preprocessing_pipeline(pandas_df)
    
    # 4. ML-ready data loader
    ml_data = ml_ready_data_loader()
    
    # 5. Visualization
    if ml_data is not None:
        visualize_data_distribution(ml_data)
    
    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE!")
    print("\nKey Takeaways:")
    print("1. NumPy provides efficient numerical operations")
    print("2. Pandas makes data manipulation intuitive")
    print("3. Always preprocess data before ML")
    print("4. Visualize to understand your data")
    print("5. Use pipelines for reproducible workflows")
    
    return {
        'numpy_data': numpy_array,
        'pandas_data': pandas_df,
        'processed_data': processed_df,
        'ml_data': ml_data,
        'stats': pipeline_stats
    }

if __name__ == "__main__":
    results = main()
    print(f"\n📊 Processed {len(results['ml_data'])} samples successfully!")