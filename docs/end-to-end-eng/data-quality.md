# Ensuring Data Quality in ML Systems

*Published: November 2025 | 20 min read*

## The Pillars of Data Quality

Data quality is the foundation of reliable machine learning systems. Poor data quality can lead to inaccurate models, biased predictions, and ultimately, poor business decisions.

### Key Dimensions of Data Quality

1. **Completeness**
   - Missing values
   - Incomplete records
   - Coverage across dimensions

2. **Accuracy**
   - Correctness of values
   - Precision of measurements
   - Free from errors

3. **Consistency**
   - Uniform format
   - Cross-system alignment
   - Temporal consistency

4. **Timeliness**
   - Data freshness
   - Update frequency
   - Processing latency

5. **Validity**
   - Adherence to schemas
   - Business rules compliance
   - Data type consistency

## Implementing Data Quality Checks

```python
import pandas as pd
import numpy as np
from datetime import datetime
from great_expectations.dataset import PandasDataset

class DataQualityChecker:
    """Comprehensive data quality validation framework."""
    
    def __init__(self, data):
        self.data = PandasDataset(data)
        self.results = {
            'checks': [],
            'passed': 0,
            'failed': 0,
            'warnings': 0
        }
    
    def check_completeness(self, threshold=0.95):
        """Check for missing values in required columns."""
        for col in self.data.columns:
            null_count = self.data[col].isnull().sum()
            completeness = 1 - (null_count / len(self.data))
            
            result = {
                'check': f'completeness_{col}',
                'status': 'PASS' if completeness >= threshold else 'FAIL',
                'metric': f"{completeness:.1%}",
                'threshold': f">{threshold:.0%}",
                'details': f"{null_count} null values found"
            }
            self._record_result(result)
    
    def check_uniqueness(self, columns):
        """Verify uniqueness constraints on specified columns."""
        for col in columns:
            total = len(self.data)
            unique = self.data[col].nunique()
            is_unique = total == unique
            
            result = {
                'check': f'uniqueness_{col}',
                'status': 'PASS' if is_unique else 'WARNING',
                'metric': f"{unique}/{total} unique",
                'threshold': '100% unique',
                'details': 'All values should be unique' if is_unique \
                          else f"{total - unique} duplicate(s) found"
            }
            self._record_result(result)
    
    def check_value_ranges(self, column_ranges):
        """Validate that values fall within expected ranges."""
        for col, (min_val, max_val) in column_ranges.items():
            out_of_range = ((self.data[col] < min_val) | 
                           (self.data[col] > max_val)).sum()
            
            result = {
                'check': f'range_{col}',
                'status': 'PASS' if out_of_range == 0 else 'FAIL',
                'metric': f"{out_of_range} out of range",
                'threshold': f"{min_val} ≤ x ≤ {max_val}",
                'details': f"Found {out_of_range} values outside range"
            }
            self._record_result(result)
    
    def check_data_freshness(self, date_column, max_days_old=1):
        """Ensure data is up-to-date."""
        latest_date = pd.to_datetime(self.data[date_column]).max()
        days_old = (pd.Timestamp.now() - latest_date).days
        
        result = {
            'check': 'data_freshness',
            'status': 'PASS' if days_old <= max_days_old else 'WARNING',
            'metric': f"{days_old} days old",
            'threshold': f"≤ {max_days_old} days",
            'details': f"Latest data point: {latest_date.date()}"
        }
        self._record_result(result)
    
    def _record_result(self, result):
        """Record the result of a check."""
        self.results['checks'].append(result)
        if result['status'] == 'PASS':
            self.results['passed'] += 1
        elif result['status'] == 'FAIL':
            self.results['failed'] += 1
        else:
            self.results['warnings'] += 1
    
    def get_report(self):
        """Generate a summary report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_checks': len(self.results['checks']),
            'checks_passed': self.results['passed'],
            'checks_failed': self.results['failed'],
            'warnings': self.results['warnings'],
            'score': self.results['passed'] / len(self.results['checks']) * 100,
            'details': self.results['checks']
        }
        return pd.DataFrame(report['details'])

# Example usage
if __name__ == "__main__":
    # Sample data
    data = pd.DataFrame({
        'id': range(1000),
        'value': np.random.normal(100, 10, 1000),
        'category': np.random.choice(['A', 'B', 'C', None], 1000, p=[0.33, 0.33, 0.33, 0.01]),
        'timestamp': pd.date_range('2023-11-01', periods=1000, freq='H')
    })
    
    # Initialize checker
    checker = DataQualityChecker(data)
    
    # Run checks
    checker.check_completeness()
    checker.check_uniqueness(['id'])
    checker.check_value_ranges({'value': (0, 200)})
    checker.check_data_freshness('timestamp')
    
    # Get report
    report = checker.get_report()
    print("\nData Quality Report:")
    print("=" * 50)
    print(report)
    
    # Calculate overall score
    score = (report['status'] == 'PASS').mean() * 100
    print(f"\nOverall Data Quality Score: {score:.1f}%")
```

## Data Quality Monitoring Framework

### 1. Automated Testing
- Unit tests for data transformations
- Integration tests for data pipelines
- Statistical tests for data distributions

### 2. Continuous Monitoring
- Real-time quality metrics
- Automated alerts for anomalies
- Trend analysis over time

### 3. Data Profiling
```python
from ydata_profiling import ProfileReport

# Generate profile report
profile = ProfileReport(
    data,
    title="Data Quality Report",
    explorative=True
)

# Save to HTML
profile.to_file("data_quality_report.html")
```

## Data Quality Tools Comparison

| Tool | Type | Key Features | Best For |
|------|------|--------------|----------|
| Great Expectations | Validation | Data testing, Documentation | Data validation |
| Deequ | Library | Unit testing, Metrics | Large-scale data |
| Monte Carlo | Platform | Data observability | Enterprise monitoring |
| Soda Core | Open source | Data testing | Data quality checks |
| Datafold | Platform | Data diffing | Data quality in CI/CD |

## Implementing Data Quality in CI/CD

1. **Pre-commit Hooks**
   - Schema validation
   - Basic data quality checks
   - Format validation

2. **CI Pipeline**
   - Automated testing
   - Data quality gates
   - Performance benchmarks

3. **CD Pipeline**
   - Data validation in staging
   - A/B testing for data changes
   - Rollback mechanisms

## Data Quality Metrics

1. **Completeness Score**
   ```
   Completeness = (1 - (Missing Values / Total Values)) * 100
   ```

2. **Accuracy Score**
   ```
   Accuracy = (Correct Values / Total Values) * 100
   ```

3. **Consistency Score**
   ```
   Consistency = (Consistent Records / Total Records) * 100
   ```

## Next Steps

1. Set up automated data quality monitoring
2. Define data quality SLAs
3. Implement data quality dashboards
4. Establish data governance policies
