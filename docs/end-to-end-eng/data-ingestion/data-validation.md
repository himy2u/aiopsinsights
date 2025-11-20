# Data Validation in ML Pipelines

*Published: November 2025 | 20 min read*

## The Importance of Data Validation

Data validation is a critical step in any ML pipeline, ensuring that the data meets quality standards before it's used for training or inference. Proper validation helps prevent model failures, biases, and unexpected behavior in production.

### Key Validation Categories

1. **Schema Validation**
   - Data types
   - Required fields
   - Value ranges
   - Allowed values

2. **Statistical Validation**
   - Distribution shifts
   - Data drift
   - Outlier detection
   - Missing values

3. **Business Rule Validation**
   - Domain-specific rules
   - Data relationships
   - Temporal consistency

## Implementation with Great Expectations

```python
import great_expectations as ge
import pandas as pd
from datetime import datetime

# Sample data
data = {
    'transaction_id': [1001, 1002, 1003, 1004, 1005],
    'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
    'amount': [125.50, 89.99, 1500.00, 45.75, 200.00],
    'currency': ['USD', 'USD', 'USD', 'USD', 'USD'],
    'timestamp': [
        '2023-11-19T10:00:00',
        '2023-11-19T10:05:23',
        '2023-11-19T10:12:45',
        '2023-11-19T10:15:30',
        '2023-11-19T10:20:15'
    ]
}

df = pd.DataFrame(data)

# Create a Great Expectations dataset
df_ge = ge.from_pandas(df)

# Define expectations
results = df_ge.expect_table_columns_to_match_ordered_list([
    'transaction_id', 'customer_id', 'amount', 'currency', 'timestamp'
])

results = df_ge.expect_column_values_to_be_in_set(
    'currency', ['USD', 'EUR', 'GBP']
)

results = df_ge.expect_column_values_to_be_between(
    'amount', min_value=0, max_value=10000
)

results = df_ge.expect_column_values_to_not_be_null('customer_id')

# Validate timestamps
results = df_ge.expect_column_values_to_match_strftime_format(
    'timestamp', 
    '%Y-%m-%dT%H:%M:%S'
)

# Check for duplicates
results = df_ge.expect_compound_columns_to_be_unique([
    'transaction_id', 'customer_id'
])

# Generate validation report
validation_results = df_ge.validate()

# Save validation results
validation_results.save_as_html('validation_report.html')

# Example of programmatic response
if not validation_results['success']:
    handle_validation_failure(validation_results)
```

## Best Practices

1. **Automated Validation**
   - Integrate validation into CI/CD pipelines
   - Set up pre-commit hooks
   - Implement data contracts

2. **Monitoring**
   - Track validation metrics over time
   - Set up alerts for critical failures
   - Monitor data drift

3. **Testing**
   - Unit tests for validation rules
   - Integration tests for data pipelines
   - Performance testing for large datasets

## Common Validation Rules

| Category | Rule | Example |
|----------|------|---------|
| Completeness | Required fields | Customer ID must not be null |
| Validity | Data type validation | Amount must be numeric |
| Accuracy | Value ranges | Age must be between 18-120 |
| Consistency | Cross-field validation | End date ≥ Start date |
| Timeliness | Freshness | Data should be updated within 24h |
| Uniqueness | Duplicate detection | Transaction IDs must be unique |

## Tools Comparison

| Tool | Description | Key Features |
|------|-------------|--------------|
| Great Expectations | Open-source data validation | Rich validation rules, Data documentation |
| Pandera | Statistical data validation | Integration with pandas, Schema enforcement |
| Pydantic | Data validation and parsing | Type hints, Schema definition |
| Deequ | Library for data quality | Built for Spark, Constraint suggestion |
| TensorFlow Data Validation | ML-focused validation | Data skew detection, Schema inference |

## Implementing a Validation Pipeline

1. **Define Validation Rules**
   - Document data contracts
   - Set up baseline statistics
   - Define thresholds for alerts

2. **Automate Validation**
   - Create reusable validation components
   - Implement validation at each pipeline stage
   - Set up automated testing

3. **Monitor and Iterate**
   - Track validation metrics
   - Update rules as requirements change
   - Continuously improve data quality

## Next Steps

1. Set up automated data quality monitoring
2. Create data quality dashboards
3. Implement data lineage tracking
4. Establish data quality SLAs
