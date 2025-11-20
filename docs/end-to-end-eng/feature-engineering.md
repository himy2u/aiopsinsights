# Feature Engineering for Machine Learning

*Published: November 2025 | 25 min read*

## The Art and Science of Feature Engineering

Feature engineering is the process of transforming raw data into meaningful features that better represent the underlying problem to predictive models, resulting in improved model accuracy on unseen data.

### Key Concepts

1. **Feature Types**
   - Numerical (continuous/discrete)
   - Categorical (nominal/ordinal)
   - Text/Unstructured
   - Time-series/Temporal
   - Geospatial

2. **Feature Transformation**
   - Normalization/Scaling
   - Encoding categorical variables
   - Handling missing values
   - Binning/Discretization

3. **Feature Creation**
   - Domain-specific features
   - Interaction terms
   - Polynomial features
   - Time-based aggregations

## Practical Implementation

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, 
    KBinsDiscretizer, FunctionTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from feature_engine import (
    datetime as dt_engine,
    imputation as imp,
    encoding as enc
)

# Sample data
data = {
    'transaction_date': pd.date_range('2023-01-01', periods=100, freq='D'),
    'amount': np.random.normal(100, 20, 100).round(2),
    'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
    'customer_age': np.random.randint(18, 80, 100),
    'is_fraud': np.random.choice([0, 1], 100, p=[0.95, 0.05])
}
df = pd.DataFrame(data)

# Create time-based features
df['day_of_week'] = df['transaction_date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['month'] = df['transaction_date'].dt.month

# Create interaction features
df['amount_per_age'] = df['amount'] / df['customer_age']

# Define preprocessing steps
numeric_features = ['amount', 'customer_age', 'amount_per_age']
categorical_features = ['category', 'day_of_week', 'month']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a feature engineering pipeline
feature_engineering_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    # Add more feature engineering steps as needed
])

# Apply transformations
X = df.drop(['is_fraud', 'transaction_date'], axis=1)
y = df['is_fraud']

X_transformed = feature_engineering_pipeline.fit_transform(X)

# Get feature names after transformation
numeric_features_transformed = feature_engineering_pipeline.named_steps['preprocessor']\
    .named_transformers_['num'].get_feature_names_out(numeric_features)

categorical_features_transformed = feature_engineering_pipeline.named_steps['preprocessor']\
    .named_transformers_['cat'].get_feature_names_out(categorical_features)

all_features = np.concatenate([
    numeric_features_transformed,
    categorical_features_transformed
])

print(f"Total features after transformation: {len(all_features)}")
```

## Advanced Feature Engineering Techniques

### 1. Target Encoding
```python
from category_encoders import TargetEncoder

# Initialize target encoder
target_enc = TargetEncoder(cols=['category'])

# Fit and transform
df['category_encoded'] = target_enc.fit_transform(
    df['category'], 
    df['is_fraud']
)
```

### 2. Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select top 10 features
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_transformed, y)

# Get selected feature indices
selected_indices = selector.get_support(indices=True)
selected_features = [all_features[i] for i in selected_indices]
print(f"Selected features: {selected_features}")
```

### 3. Time-Series Features
```python
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series

# Create time-series features
df_ts = roll_time_series(
    df, 
    column_id="customer_id",
    column_sort="transaction_date",
    max_timeshift=30,
    min_timeshift=5
)

# Extract time-series features
features_ts = extract_features(
    df_ts.drop("is_fraud", axis=1),
    column_id="id", 
    column_sort="transaction_date"
)
```

## Feature Engineering Best Practices

1. **Start Simple**
   - Begin with basic features
   - Add complexity gradually
   - Validate each addition

2. **Domain Knowledge**
   - Incorporate business insights
   - Understand the data generation process
   - Consult with domain experts

3. **Automation**
   - Use feature stores
   - Implement feature versioning
   - Automate feature validation

4. **Monitoring**
   - Track feature distributions
   - Monitor feature importance
   - Set up data quality checks

## Feature Stores

Modern feature stores help manage the feature engineering lifecycle:

1. **Feast** - Open source feature store
2. **Tecton** - Enterprise feature platform
3. **Hopsworks** - Open-source feature store
4. **AWS Feature Store** - Managed service

## Common Pitfalls

1. **Data Leakage**
   - Using future information
   - Improper cross-validation
   - Target leakage

2. **Over-Engineering**
   - Creating too many features
   - Complex transformations without justification
   - Ignoring model interpretability

3. **Scalability Issues**
   - High-dimensional feature spaces
   - Inefficient transformations
   - Lack of incremental updates

## Next Steps

1. Implement automated feature validation
2. Set up feature monitoring
3. Explore automated feature engineering
4. Consider feature store implementation
