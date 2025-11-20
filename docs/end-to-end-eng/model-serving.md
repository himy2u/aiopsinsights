# Model Serving: Production Deployment Strategies

*Published: November 2025 | 30 min read*

## Introduction to Model Serving

Model serving is the process of deploying machine learning models to production environments where they can make predictions on new data. A robust serving infrastructure is crucial for delivering model predictions reliably, scalably, and with low latency.

### Key Requirements for Production Serving

1. **Scalability**
   - Handle varying load
   - Scale to zero when not in use
   - Support for batch and real-time inference

2. **Reliability**
   - High availability
   - Fault tolerance
   - Graceful degradation

3. **Performance**
   - Low latency
   - High throughput
   - Efficient resource utilization

4. **Operational**
   - Monitoring
   - Logging
   - Versioning
   - Rollback capabilities

## Model Serving Patterns

### 1. Real-time Inference
```python
from fastapi import FastAPI, HTTPException
import torch
from pydantic import BaseModel
import numpy as np
import logging
from typing import Dict, List

# Initialize FastAPI app
app = FastAPI(title="ML Model Serving API")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model (simplified example)
class ModelWrapper:
    def __init__(self):
        self.model = self._load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def _load_model(self):
        # In practice, load your trained model here
        # model = load_your_model()
        # return model
        return torch.nn.Linear(10, 2)  # Dummy model
    
    def preprocess(self, input_data: Dict) -> torch.Tensor:
        # Convert input to model-expected format
        features = np.array(input_data["features"]).astype(np.float32)
        return torch.from_numpy(features).to(self.device)
    
    def predict(self, input_data: Dict) -> Dict:
        try:
            with torch.no_grad():
                inputs = self.preprocess(input_data)
                outputs = self.model(inputs)
                predictions = torch.softmax(outputs, dim=1).cpu().numpy()
            return {"predictions": predictions.tolist()}
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize model wrapper
model_wrapper = ModelWrapper()

# Request/Response models
class PredictionRequest(BaseModel):
    features: List[List[float]]

class PredictionResponse(BaseModel):
    predictions: List[List[float]]
    model_version: str = "1.0.0"

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Add request logging
        logger.info(f"Received prediction request with {len(request.features)} samples")
        
        # Get predictions
        result = model_wrapper.predict({"features": request.features})
        
        # Log prediction metrics
        logger.info(f"Successfully processed prediction request")
        
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Model metadata endpoint
@app.get("/model/metadata")
async def model_metadata():
    return {
        "model_name": "sentiment-classifier",
        "version": "1.0.0",
        "input_schema": {
            "features": "List[List[float]] - 10 dimensional features"
        },
        "output_schema": {
            "predictions": "List[List[float]] - Class probabilities"
        }
    }

# Run with: uvicorn model_serving:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Batch Inference
```python
import pandas as pd
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class BatchInference:
    def __init__(self, model_wrapper, batch_size: int = 32, max_workers: int = 4):
        self.model_wrapper = model_wrapper
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a single batch of records."""
        try:
            # Convert batch to model input format
            features = [record["features"] for record in batch]
            
            # Get predictions
            result = self.model_wrapper.predict({"features": features})
            
            # Add predictions to records
            for i, pred in enumerate(result["predictions"]):
                batch[i]["prediction"] = pred
                batch[i]["prediction_label"] = np.argmax(pred)
            
            return batch
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            # Return records with error flag
            for record in batch:
                record["error"] = str(e)
            return batch
    
    def process_dataset(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process entire dataset in parallel batches."""
        results = []
        
        # Create batches
        batches = [
            data[i:i + self.batch_size] 
            for i in range(0, len(data), self.batch_size)
        ]
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            batch_results = list(executor.map(self.process_batch, batches))
        
        # Flatten results
        for batch in batch_results:
            results.extend(batch)
        
        return results

# Example usage
if __name__ == "__main__":
    # Sample data
    data = [{"id": i, "features": np.random.rand(10).tolist()} for i in range(1000)]
    
    # Initialize batch processor
    batch_processor = BatchInference(model_wrapper, batch_size=64, max_workers=4)
    
    # Process data
    results = batch_processor.process_dataset(data)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_parquet("predictions.parquet", index=False)
```

## Model Serving Infrastructure

### 1. Containerization with Docker
```dockerfile
# Dockerfile for model serving
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application code
COPY model.pth /app/model.pth
COPY app /app/app

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000
ENV WORKERS=4
ENV TIMEOUT=120

# Expose port
EXPOSE ${PORT}

# Start the application
CMD exec gunicorn --bind :${PORT} --workers ${WORKERS} --timeout ${TIMEOUT} \
    --worker-class uvicorn.workers.UvicornWorker app.main:app
```

### 2. Kubernetes Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving
  labels:
    app: model-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-serving
  template:
    metadata:
      labels:
        app: model-serving
    spec:
      containers:
      - name: model-serving
        image: your-registry/model-serving:1.0.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        env:
        - name: PORT
          value: "8000"
        - name: WORKERS
          value: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: model-serving
spec:
  selector:
    app: model-serving
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Advanced Serving Features

### 1. Model Versioning and A/B Testing
```python
from fastapi import Request

class ModelRouter:
    def __init__(self):
        self.models = {
            "v1": ModelWrapper("models/v1"),
            "v2": ModelWrapper("models/v2"),
        }
        self.default_version = "v2"
    
    def get_model(self, version: str = None):
        return self.models.get(version, self.models[self.default_version])

# Add versioned endpoint
@app.post("/v{version}/predict")
async def versioned_predict(
    version: str,
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    model = model_router.get_model(version)
    
    # Log prediction for analysis
    background_tasks.add_task(
        log_prediction,
        version=version,
        input_data=request.dict(),
        timestamp=datetime.utcnow()
    )
    
    return model.predict(request)
```

### 2. Request Batching
```python
from fastapi import BackgroundTasks
import asyncio
from typing import List

class RequestBatcher:
    def __init__(self, max_batch_size=32, max_wait_time=0.1):
        self.batch = []
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.lock = asyncio.Lock()
        self.condition = asyncio.Condition()
    
    async def add_request(self, request):
        async with self.lock:
            self.batch.append(request)
            
            # If batch is full, process immediately
            if len(self.batch) >= self.max_batch_size:
                await self.condition.acquire()
                self.condition.notify_all()
                self.condition.release()
                return await self.process_batch()
            
            # Otherwise, wait for batch to fill or timeout
            try:
                await asyncio.wait_for(
                    self._wait_for_batch(),
                    timeout=self.max_wait_time
                )
                return await self.process_batch()
            except asyncio.TimeoutError:
                return await self.process_batch()
    
    async def _wait_for_batch(self):
        async with self.condition:
            await self.condition.wait()
    
    async def process_batch(self):
        async with self.lock:
            if not self.batch:
                return []
                
            # Get current batch and clear
            current_batch = self.batch.copy()
            self.batch = []
            
            # Process batch (in practice, this would call your model)
            features = [item["features"] for item in current_batch]
            predictions = model_wrapper.predict({"features": features})
            
            # Map predictions back to requests
            for i, item in enumerate(current_batch):
                item["prediction"] = predictions["predictions"][i]
            
            return current_batch

# Initialize batcher
batcher = RequestBatcher()

# Batched prediction endpoint
@app.post("/predict/batch")
async def batch_predict(request: PredictionRequest):
    results = await batcher.add_request({"features": request.features})
    return {"predictions": [item["prediction"] for item in results]}
```

## Model Serving Best Practices

1. **Performance Optimization**
   - Model quantization
   - ONNX/TensorRT conversion
   - Request batching
   - Caching frequent predictions

2. **Reliability**
   - Circuit breakers
   - Retry mechanisms
   - Fallback strategies
   - Rate limiting

3. **Observability**
   - Metrics collection
   - Distributed tracing
   - Log aggregation
   - Alerting

4. **Security**
   - Authentication/Authorization
   - Input validation
   - Model extraction protection
   - Data encryption

## Model Serving Tools

| Tool | Type | Key Features | Best For |
|------|------|--------------|----------|
| TorchServe | Framework | Multi-model, Versioning | PyTorch models |
| TensorFlow Serving | Framework | High performance, Batching | TensorFlow models |
| KServe | Platform | Kubernetes-native, Autoscaling | Enterprise serving |
| BentoML | Framework | Model packaging, Deployment | MLOps pipelines |
| Seldon Core | Platform | Advanced routing, A/B testing | Complex deployments |

## Next Steps

1. Implement canary deployments
2. Set up model monitoring
3. Add feature store integration
4. Implement shadow mode testing
