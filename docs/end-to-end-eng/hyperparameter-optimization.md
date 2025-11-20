# Hyperparameter Optimization: Advanced Techniques and Best Practices

*Published: November 2025 | 28 min read*

## Understanding Hyperparameter Optimization

Hyperparameter optimization (HPO) is the process of finding the optimal set of hyperparameters for a machine learning model that results in the best performance on a given dataset. It's a critical step in the model development process that can significantly impact model performance.

### Types of Hyperparameters

1. **Model Hyperparameters**
   - Learning rate
   - Network architecture (layers, units)
   - Activation functions
   - Regularization parameters

2. **Training Hyperparameters**
   - Batch size
   - Number of epochs
   - Optimization algorithm
   - Learning rate schedule

3. **Data Hyperparameters**
   - Data augmentation
   - Feature selection
   - Class weights
   - Cross-validation strategy

## Hyperparameter Optimization Techniques

### 1. Grid Search
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='f1_macro'
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

### 2. Random Search
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [None] + list(range(5, 50, 5)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42,
    scoring='f1_macro'
)

random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.4f}")
```

### 3. Bayesian Optimization with Optuna
```python
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score

def objective(trial):
    # Define hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 30, step=1),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'random_state': 42
    }
    
    # Train and evaluate model
    model = RandomForestClassifier(**params)
    scores = cross_val_score(
        model, X_train, y_train, 
        cv=5, scoring='f1_macro', n_jobs=-1
    )
    
    return scores.mean()

# Create study
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42)
)

# Optimize
study.optimize(
    objective, 
    n_trials=100,
    n_jobs=-1,
    show_progress_bar=True
)

# Get best parameters
print(f"Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value:.4f}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Visualize optimization history
optuna.visualization.plot_optimization_history(study).show()

# Visualize parameter importance
optuna.visualization.plot_param_importances(study).show()
```

## Advanced HPO Techniques

### 1. Population Based Training (PBT)
```python
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

# Define trainable function
def train_mnist(config):
    # Model setup
    model = ConvNet()
    model = torch.nn.DataParallel(model)
    model.to(device)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"]
    )
    
    # Training loop
    for epoch in range(10):
        train_epoch(model, optimizer, train_loader)
        acc = test(model, test_loader)
        
        # Report metrics to Tune
        tune.report(mean_accuracy=acc)

# Define PBT scheduler
pbt_scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=5,
    hyperparam_mutations={
        "lr": tune.loguniform(1e-4, 1e-1),
        "momentum": tune.uniform(0.1, 0.9),
    })

# Run the trial
analysis = tune.run(
    train_mnist,
    name="pbt_mnist",
    scheduler=pbt_scheduler,
    metric="mean_accuracy",
    mode="max",
    stop={"training_iteration": 100},
    num_samples=10,
    config={
        "lr": 0.001,
        "momentum": 0.8,
    }
)
```

### 2. Multi-fidelity Optimization with Hyperband
```python
from ray.tune.schedulers import HyperBandScheduler

# Define hyperband scheduler
hyperband = HyperBandScheduler(
    time_attr="training_iteration",
    metric="mean_accuracy",
    mode="max",
    max_t=100,
    reduction_factor=3
)

# Run the trial
analysis = tune.run(
    train_mnist,
    name="hyperband_mnist",
    scheduler=hyperband,
    resources_per_trial={"cpu": 2, "gpu": 0.5},
    num_samples=20,
    stop={"training_iteration": 100},
    config={
        "lr": tune.loguniform(1e-4, 1e-1),
        "momentum": tune.uniform(0.1, 0.9),
    }
)
```

## HPO Best Practices

1. **Search Space Design**
   - Use appropriate scales (log vs linear)
   - Define meaningful ranges
   - Consider conditional parameters

2. **Efficient Search**
   - Start with broad search, then refine
   - Use early stopping
   - Leverage parallelization

3. **Evaluation Strategy**
   - Use proper cross-validation
   - Consider time-based splits for time series
   - Maintain a hold-out test set

4. **Resource Management**
   - Balance exploration vs. exploitation
   - Use multi-fidelity optimization
   - Consider cost of evaluation

## HPO Tools Comparison

| Tool | Type | Key Features | Best For |
|------|------|--------------|----------|
| Optuna | Framework | Distributed, Pruning | General HPO |
| Ray Tune | Framework | Scalable, Multi-fidelity | Large-scale |
| Hyperopt | Library | Bayesian optimization | Medium-scale |
| Weights & Biases | Platform | Visualization, Tracking | Experiment tracking |
| Katib | Kubernetes-native | AutoML, Hyperparameter tuning | Kubernetes environments |

## Implementing HPO in Production

1. **Infrastructure**
   - Distributed computing
   - GPU/TPU support
   - Resource management

2. **Monitoring**
   - Track experiments
   - Visualize progress
   - Compare trials

3. **Automation**
   - Continuous training
   - Automated model selection
   - Pipeline integration

## Next Steps

1. Implement automated HPO pipelines
2. Set up distributed training
3. Monitor model performance drift
4. Explore AutoML solutions
