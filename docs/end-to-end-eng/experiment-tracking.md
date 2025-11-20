# Experiment Tracking in Machine Learning

*Published: November 2025 | 22 min read*

## The Importance of Experiment Tracking

Experiment tracking is the practice of systematically recording and managing machine learning experiments to enable reproducibility, collaboration, and insight generation. It's a critical component of the MLOps lifecycle.

### Key Benefits

1. **Reproducibility**
   - Track exact code, data, and hyperparameters
   - Recreate any model version
   - Audit model development history

2. **Collaboration**
   - Share experiments across teams
   - Compare results
   - Knowledge transfer

3. **Insight Generation**
   - Identify best performing models
   - Understand impact of changes
   - Optimize hyperparameters

## Implementing Experiment Tracking with MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Enable auto-logging
mlflow.sklearn.autolog()

# Sample data
data = pd.DataFrame({
    'feature1': np.random.rand(1000),
    'feature2': np.random.rand(1000),
    'target': np.random.randint(0, 2, 1000)
})

X_train, X_test, y_train, y_test = train_test_split(
    data[['feature1', 'feature2']], 
    data['target'], 
    test_size=0.2,
    random_state=42
)

def train_model(params):
    """Train a model with the given parameters and log the experiment."""
    with mlflow.start_run(run_name="RandomForest_Experiment"):
        # Log parameters
        mlflow.log_params(params)
        
        # Initialize and train model
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            random_state=42
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="RandomForestClassifier"
        )
        
        # Log artifacts (e.g., feature importance plot)
        import matplotlib.pyplot as plt
        
        feature_importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        feature_importance.plot(kind='bar')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Log the plot
        mlflow.log_figure(plt.gcf(), "feature_importance.png")
        plt.close()
        
        return metrics

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

# Set tracking URI (e.g., local file store or remote server)
mlflow.set_tracking_uri("file:./mlruns")

# Create experiment
mlflow.set_experiment("RandomForest_Classification")

# Run experiments
for n_estimators in param_grid['n_estimators']:
    for max_depth in param_grid['max_depth']:
        for min_samples_split in param_grid['min_samples_split']:
            params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split
            }
            print(f"Training with params: {params}")
            train_model(params)

# Compare experiments
df = mlflow.search_runs(
    experiment_ids=[mlflow.get_experiment_by_name("RandomForest_Classification").experiment_id],
    order_by=["metrics.f1 DESC"]
)

print("\nTop 5 models by F1 score:")
print(df[['params.n_estimators', 'params.max_depth', 'params.min_samples_split', 
          'metrics.f1', 'metrics.accuracy']].head())
```

## Advanced Experiment Tracking Features

### 1. Nested Runs
```python
with mlflow.start_run(run_name="parent_run") as parent_run:
    mlflow.log_param("parent_param", "value")
    
    # Child run 1
    with mlflow.start_run(run_name="child_1", nested=True) as child_run:
        mlflow.log_param("child_param", "value1")
        # Training and logging...
    
    # Child run 2
    with mlflow.start_run(run_name="child_2", nested=True) as child_run:
        mlflow.log_param("child_param", "value2")
        # Training and logging...
```

### 2. Model Registry
```python
# Register model
model_uri = f"runs:/{run_id}/model"
model_details = mlflow.register_model(
    model_uri=model_uri,
    name="RandomForestClassifier"
)

# Transition model stage
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="RandomForestClassifier",
    version=model_details.version,
    stage="Production"
)
```

### 3. Hyperparameter Tuning Integration
```python
import optuna
from optuna.integration.mlflow import MLflowCallback

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
    }
    
    metrics = train_model(params)
    return metrics['f1']  # Optimize for F1 score

# Initialize MLflow callback
mlflc = MLflowCallback(
    tracking_uri=mlflow.get_tracking_uri(),
    metric_name="f1"
)

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(
    objective,
    n_trials=20,
    callbacks=[mlflc],
    n_jobs=-1
)
```

## Experiment Tracking Best Practices

1. **Consistent Naming Conventions**
   - Use descriptive run names
   - Tag experiments consistently
   - Document experiment purpose

2. **Comprehensive Logging**
   - Log all hyperparameters
   - Track data versions
   - Record environment details

3. **Version Control Integration**
   - Link experiments to git commits
   - Track code changes
   - Enable reproducibility

4. **Artifact Management**
   - Save model checkpoints
   - Log visualizations
   - Store evaluation reports

## Tools Comparison

| Tool | Type | Key Features | Best For |
|------|------|--------------|----------|
| MLflow | Open source | Experiment tracking, Model registry | End-to-end ML lifecycle |
| Weights & Biases | Cloud/Self-hosted | Experiment tracking, Visualization | Research, Deep Learning |
| TensorBoard | Open source | Visualization, Debugging | TensorFlow/PyTorch |
| Comet.ml | Cloud/Self-hosted | Experiment tracking, Model management | Enterprise ML |
| Neptune.ai | Cloud | Experiment tracking, Model registry | Team collaboration |

## Implementing a Robust Experiment Tracking System

1. **Infrastructure Setup**
   - Choose tracking server (MLflow, etc.)
   - Configure storage backend
   - Set up access controls

2. **Integration**
   - Connect to version control
   - Set up CI/CD pipelines
   - Integrate with model serving

3. **Governance**
   - Define retention policies
   - Implement audit trails
   - Set up monitoring

## Next Steps

1. Set up a centralized experiment tracking server
2. Implement experiment templates
3. Create automated reporting
4. Establish model governance processes
