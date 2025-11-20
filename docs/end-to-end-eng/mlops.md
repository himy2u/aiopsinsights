# MLOps Best Practices: From Experimentation to Production

*Published: November 2025 | 35 min read*

## The MLOps Lifecycle

MLOps (Machine Learning Operations) is the practice of unifying ML system development (Dev) and ML system operations (Ops) to standardize and streamline the continuous delivery of high-performing models in production.

### Core Principles

1. **Versioning**
   - Code versioning (Git)
   - Data versioning (DVC, Dolt)
   - Model versioning (MLflow, DVC)
   - Environment versioning (Docker, Conda)

2. **Reproducibility**
   - Deterministic training
   - Pinned dependencies
   - Immutable artifacts
   - Complete audit trails

3. **Automation**
   - CI/CD for ML
   - Automated testing
   - Automated deployment
   - Automated monitoring

4. **Collaboration**
   - Cross-functional teams
   - Shared tooling
   - Documentation
   - Knowledge sharing

## MLOps Maturity Model

### Level 1: Manual Process
- Ad-hoc, manual workflows
- No CI/CD
- Manual deployment
- No monitoring

### Level 2: ML Pipeline Automation
- Automated training pipeline
- Experiment tracking
- Basic CI/CD
- Manual model validation

### Level 3: CI/CD Pipeline Automation
- Automated model testing
- Automated deployment
- Model versioning
- Basic monitoring

### Level 4: Full MLOps Automation
- Continuous training
- Automated model validation
- Canary deployments
- Comprehensive monitoring
- Automated rollback

## Implementing MLOps with GitHub Actions

```yaml
# .github/workflows/ml-train-eval-deploy.yml
name: ML Train, Evaluate, and Deploy

on:
  push:
    branches: [ main ]
    paths:
      - 'models/**'
      - 'data/**'
      - '.github/workflows/ml-train-eval-deploy.yml'
  workflow_dispatch:
    inputs:
      retrain:
        description: 'Force retrain model'
        required: false
        default: 'false'

env:
  MODEL_NAME: sentiment-classifier
  VERSION: 1.0.0
  DOCKER_IMAGE: ghcr.io/your-org/${{ env.MODEL_NAME }}:${{ github.sha }}

jobs:
  train:
    name: Train Model
    runs-on: ubuntu-latest
    
    services:
      minio:
        image: minio/minio
        env:
          MINIO_ROOT_USER: minio
          MINIO_ROOT_PASSWORD: minio123
        ports:
          - 9000:9000
        options: >-
          --health-cmd "curl -f http://localhost:9000/minio/health/live"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install dvc dvc-s3
    
    - name: Configure DVC
      run: |
        dvc remote add --default minio s3://models
        dvc remote modify minio endpointurl http://localhost:9000
        dvc remote modify minio --local access_key_id minio
        dvc remote modify minio --local secret_access_key minio123
    
    - name: Pull data and models
      run: |
        dvc pull -r minio
    
    - name: Run training
      run: |
        python train.py \
          --data-path data/processed \
          --model-path models \
          --experiment-name ${{ github.run_id }}
    
    - name: Evaluate model
      run: |
        python evaluate.py \
          --model-path models/${{ env.MODEL_NAME }} \
          --test-data data/processed/test.parquet \
          --output-path metrics/
    
    - name: Compare with production
      id: compare
      run: |
        # Compare with production metrics
        # If new model is better, set should_deploy=true
        echo "should_deploy=true" >> $GITHUB_OUTPUT
    
    - name: Push model and metrics
      if: steps.compare.outputs.should_deploy == 'true'
      run: |
        dvc add models/${{ env.MODEL_NAME}}
        dvc push -r minio
        
        # Log metrics to MLflow
        python log_metrics.py \
          --metrics metrics/evaluation.json \
          --run-id ${{ github.run_id }} \
          --model-path models/${{ env.MODEL_NAME}}
    
    - name: Package model
      if: steps.compare.outputs.should_deploy == 'true'
      run: |
        # Package model with MLflow
        mlflow models build-docker \
          --model-uri models/${{ env.MODEL_NAME}} \
          --name ${{ env.MODEL_NAME}} \
          --env-manager local
    
    - name: Login to GitHub Container Registry
      if: steps.compare.outputs.should_deploy == 'true'
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Tag and push Docker image
      if: steps.compare.outputs.should_deploy == 'true'
      run: |
        docker tag ${{ env.MODEL_NAME }} ${{ env.DOCKER_IMAGE }}
        docker push ${{ env.DOCKER_IMAGE }}

  deploy-staging:
    name: Deploy to Staging
    needs: train
    if: needs.train.outputs.should_deploy == 'true'
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout
      uses: actions/checkout@v3
    
    - name: Deploy to Kubernetes (Staging)
      uses: azure/k8s-deploy@v4
      with:
        namespace: staging
        manifests: k8s/staging/*.yaml
        images: ${{ env.DOCKER_IMAGE }}
        imagepullsecrets: |
          registry-credentials
    
    - name: Run integration tests
      run: |
        # Run integration tests against staging
        pytest tests/integration/ -v
    
    - name: Approve Production Deployment
      if: success()
      uses: actions/github-script@v6
      with:
        script: |
          const { data } = await github.rest.actions.createWorkflowDispatch({
            owner: context.repo.owner,
            repo: context.repo.repo,
            workflow_id: 'deploy-production.yml',
            ref: 'main',
            inputs: {
              image: '${{ env.DOCKER_IMAGE }}',
              version: '${{ env.VERSION }}',
              run_id: '${{ github.run_id }}'
            }
          });
          return data;

  deploy-production:
    name: Deploy to Production
    needs: deploy-staging
    if: needs.deploy-staging.result == 'success'
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout
      uses: actions/checkout@v3
    
    - name: Deploy to Kubernetes (Production)
      uses: azure/k8s-deploy@v4
      with:
        namespace: production
        strategy: canary
        traffic-split-method: smi
        baseline-and-canary-replicas: replicas=3
        max-surge: 25%
        max-unavailable: 0
        manifests: k8s/production/*.yaml
        images: ${{ env.DOCKER_IMAGE }}
        imagepullsecrets: |
          registry-credentials
    
    - name: Verify deployment
      run: |
        # Run smoke tests against production
        pytest tests/smoke/ -v
    
    - name: Complete Canary Deployment
      if: success()
      run: |
        # Complete canary deployment
        kubectl set traffic deployment/${{ env.MODEL_NAME }} \
          --namespace=production \
          --source=TrafficSplit/${{ env.MODEL_NAME }} \
          --traffic=canary=100% \
          --type=trafficsplits
```

## Model Testing Strategy

### 1. Unit Tests
```python
# tests/unit/test_preprocessing.py
def test_tokenizer():
    from src.preprocessing import Tokenizer
    
    tokenizer = Tokenizer()
    text = "This is a test"
    expected = ["this", "is", "a", "test"]
    
    assert tokenizer.tokenize(text) == expected

def test_feature_extractor():
    from src.features import FeatureExtractor
    
    extractor = FeatureExtractor()
    data = [{"text": "Great product!"}, {"text": "Not good."}]
    
    features = extractor.transform(data)
    
    assert features.shape[0] == 2
    assert "sentiment_score" in features.columns
```

### 2. Integration Tests
```python
# tests/integration/test_training.py
def test_training_pipeline():
    from src.pipeline import TrainingPipeline
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    
    # Initialize and run pipeline
    pipeline = TrainingPipeline()
    model, metrics = pipeline.run(X, y)
    
    # Assert model was trained
    assert hasattr(model, 'predict')
    
    # Assert metrics meet expectations
    assert metrics['accuracy'] > 0.8
    assert metrics['f1_score'] > 0.8
```

### 3. Model Validation Tests
```python
# tests/validation/test_model_validation.py
def test_model_performance():
    from src.validation import ModelValidator
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(random_state=42)
    model.fit(X[:800], y[:800])
    
    # Validate model
    validator = ModelValidator(
        min_accuracy=0.8,
        min_precision=0.75,
        min_recall=0.7
    )
    
    is_valid, report = validator.validate(
        model=model,
        X_test=X[800:],
        y_test=y[800:]
    )
    
    assert is_valid
    assert report['accuracy'] >= 0.8
```

## Model Monitoring with Prometheus and Grafana

### 1. Data Drift Detection
```python
from alibi_detect import KSDrift
import numpy as np

class DriftDetector:
    def __init__(self, reference_data, p_val=0.05):
        self.drift_detector = KSDrift(
            reference_data,
            p_val=p_val,
            input_shape=reference_data.shape[1:],
        )
    
    def detect_drift(self, new_data):
        """Detect drift in new data compared to reference data."""
        preds = self.drift_detector.predict(
            new_data,
            drift_type='feature',
            return_p_val=True,
            return_distance=True
        )
        
        return {
            'is_drift': preds['data']['is_drift'][0],
            'p_value': float(preds['data']['p_val'][0]),
            'distance': float(preds['data']['distance'][0]),
            'threshold': float(preds['data']['threshold'][0])
        }

# Example usage
if __name__ == "__main__":
    # Generate reference data (e.g., from training set)
    reference_data = np.random.normal(0, 1, (1000, 10))
    
    # Initialize drift detector
    detector = DriftDetector(reference_data)
    
    # Simulate new data (with some drift)
    new_data = np.random.normal(0.5, 1, (100, 10))
    
    # Detect drift
    result = detector.detect_drift(new_data)
    print(f"Drift detected: {result['is_drift']}")
    print(f"P-value: {result['p_value']:.4f}")
```

### 2. Model Performance Monitoring
```python
from prometheus_client import Gauge, start_http_server
import time
import random

class ModelMonitor:
    def __init__(self, model_name):
        self.model_name = model_name
        
        # Define metrics
        self.prediction_latency = Gauge(
            f'model_{model_name}_prediction_latency_seconds',
            'Prediction latency in seconds',
            ['model_version']
        )
        
        self.prediction_counter = Gauge(
            f'model_{model_name}_predictions_total',
            'Total number of predictions',
            ['model_version', 'status']
        )
        
        self.feature_drift = Gauge(
            f'model_{model_name}_feature_drift',
            'Feature drift score',
            ['feature_name', 'model_version']
        )
        
        # Start metrics server
        start_http_server(8000)
    
    def record_prediction(self, version, latency_ms, status='success'):
        """Record prediction metrics."""
        self.prediction_latency.labels(
            model_version=version
        ).set(latency_ms / 1000)
        
        self.prediction_counter.labels(
            model_version=version,
            status=status
        ).inc()
    
    def record_feature_drift(self, feature_name, version, drift_score):
        """Record feature drift metrics."""
        self.feature_drift.labels(
            feature_name=feature_name,
            model_version=version
        ).set(drift_score)

# Example usage
if __name__ == "__main__":
    monitor = ModelMonitor("sentiment-classifier")
    
    # Simulate recording metrics
    while True:
        # Simulate prediction
        latency = random.uniform(10, 100)  # ms
        monitor.record_prediction(
            version="1.0.0",
            latency_ms=latency,
            status="success" if random.random() > 0.1 else "error"
        )
        
        # Simulate feature drift
        for feature in ["text_length", "sentiment_score", "word_count"]:
            drift = random.uniform(0, 0.2)  # Some small drift
            monitor.record_feature_drift(feature, "1.0.0", drift)
        
        time.sleep(5)
```

## Model Governance

### 1. Model Registry
```python
from mlflow.tracking import MlflowClient
from datetime import datetime

class ModelRegistry:
    def __init__(self, tracking_uri):
        self.client = MlflowClient(tracking_uri)
    
    def register_model(self, run_id, model_name, description=None):
        """Register a new model version."""
        # Create model if it doesn't exist
        try:
            self.client.create_registered_model(model_name)
        except:
            pass  # Model already exists
        
        # Create model version
        model_uri = f"runs:/{run_id}/model"
        mv = self.client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
            description=description
        )
        
        return mv.version
    
    def transition_stage(self, model_name, version, stage):
        """Transition model to a new stage (Staging, Production, Archived)."""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True
        )
        
        # Add description
        self.client.update_model_version(
            name=model_name,
            version=version,
            description=f"Transitioned to {stage} on {datetime.utcnow().isoformat()}"
        )
    
    def get_production_model(self, model_name):
        """Get the current production model."""
        try:
            return self.client.get_latest_versions(
                model_name, 
                stages=["Production"]
            )[0]
        except:
            return None
    
    def compare_models(self, model_name, version_a, version_b):
        """Compare two model versions."""
        a = self.client.get_model_version(model_name, version_a)
        b = self.client.get_model_version(model_name, version_b)
        
        # Compare metrics, parameters, tags
        run_a = self.client.get_run(a.run_id)
        run_b = self.client.get_run(b.run_id)
        
        return {
            'metrics': {
                'a': run_a.data.metrics,
                'b': run_b.data.metrics
            },
            'params': {
                'a': run_a.data.params,
                'b': run_b.data.params
            },
            'tags': {
                'a': run_a.data.tags,
                'b': run_b.data.tags
            }
        }
```

## MLOps Tools Ecosystem

| Category | Tools |
|----------|-------|
| **Version Control** | Git, DVC, Git LFS |
| **Experiment Tracking** | MLflow, Weights & Biases, Comet |
| **Workflow Orchestration** | Airflow, Kubeflow, Argo Workflows |
| **Model Serving** | TorchServe, TensorFlow Serving, KServe |
| **Monitoring** | Prometheus, Grafana, Evidently |
| **Feature Store** | Feast, Tecton, Hopsworks |
| **Model Registry** | MLflow, Neptune, Seldon Core |
| **Infrastructure** | Kubernetes, Docker, Terraform |

## Next Steps

1. **Start Small**
   - Implement basic CI/CD for ML
   - Add model versioning
   - Set up basic monitoring

2. **Scale Up**
   - Implement feature store
   - Add automated retraining
   - Set up advanced monitoring

3. **Optimize**
   - Implement canary deployments
   - Add A/B testing
   - Optimize resource usage

4. **Govern**
   - Implement model governance
   - Set up access controls
   - Document everything
