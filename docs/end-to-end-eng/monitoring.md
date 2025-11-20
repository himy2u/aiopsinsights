# Monitoring & Observability for ML Systems

*Published: November 2025 | 28 min read*

## The Three Pillars of ML Observability

Effective monitoring of machine learning systems requires going beyond traditional software metrics to include specialized ML-specific telemetry. A comprehensive monitoring strategy should cover:

### 1. System Metrics
- Resource utilization (CPU, GPU, memory)
- Request rates and latencies
- Error rates and types
- Container/pod health

### 2. Data Quality
- Feature distributions
- Data drift detection
- Missing values
- Outlier detection

### 3. Model Performance
- Prediction accuracy
- Business metrics
- Concept drift
- Feature importance shifts

## Implementing ML Monitoring with Prometheus and Grafana

```python
# monitoring/metrics.py
from prometheus_client import start_http_server, Gauge, Histogram, Counter
import time
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class ModelMetrics:
    """A class to track and expose model metrics for Prometheus."""
    
    def __init__(self, model_name: str, label_names: Optional[List[str]] = None):
        self.model_name = model_name
        self.label_names = label_names or []
        
        # Common metrics
        self.request_counter = Counter(
            'model_requests_total',
            'Total number of prediction requests',
            ['model_name', 'endpoint']
        )
        
        self.prediction_latency = Histogram(
            'model_prediction_latency_seconds',
            'Prediction latency in seconds',
            ['model_name', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        )
        
        self.prediction_errors = Counter(
            'model_prediction_errors_total',
            'Total number of prediction errors',
            ['model_name', 'endpoint', 'error_type']
        )
        
        # Data quality metrics
        self.feature_drift = Gauge(
            'model_feature_drift',
            'Feature drift score',
            ['model_name', 'feature_name']
        )
        
        # Model performance metrics
        self.prediction_distribution = Histogram(
            'model_prediction_distribution',
            'Distribution of prediction values',
            ['model_name'],
            buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # Business metrics
        self.business_metric = Gauge(
            'model_business_metric',
            'Business metric (e.g., revenue, conversion rate)',
            ['model_name', 'metric_name']
        )
    
    def record_prediction(
        self,
        endpoint: str,
        features: Dict[str, Any],
        prediction: float,
        label: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a prediction and its metadata."""
        try:
            # Increment request counter
            self.request_counter.labels(
                model_name=self.model_name,
                endpoint=endpoint
            ).inc()
            
            # Record prediction distribution
            self.prediction_distribution.labels(
                model_name=self.model_name
            ).observe(prediction)
            
            # Record feature drift (simplified example)
            for feature_name, value in features.items():
                # In practice, you'd calculate drift using a reference distribution
                self.feature_drift.labels(
                    model_name=self.model_name,
                    feature_name=feature_name
                ).set(value)
            
            # Record business metrics if available
            if metadata and 'business_metric' in metadata:
                for metric_name, metric_value in metadata['business_metric'].items():
                    self.business_metric.labels(
                        model_name=self.model_name,
                        metric_name=metric_name
                    ).set(metric_value)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record metrics: {str(e)}")
            return False
    
    def record_error(
        self,
        endpoint: str,
        error_type: str,
        error_message: str = "",
        exception: Optional[Exception] = None
    ) -> None:
        """Record an error that occurred during prediction."""
        self.prediction_errors.labels(
            model_name=self.model_name,
            endpoint=endpoint,
            error_type=error_type
        ).inc()
        
        logger.error(
            f"Prediction error in {endpoint}: {error_message}",
            exc_info=exception
        )

# Example usage
if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(8000)
    
    # Initialize metrics
    metrics = ModelMetrics(model_name="fraud_detection")
    
    # Simulate recording predictions
    while True:
        features = {
            'amount': np.random.normal(100, 50),
            'transaction_hour': np.random.randint(0, 24),
            'user_risk_score': np.random.uniform(0, 1)
        }
        
        # Record prediction
        metrics.record_prediction(
            endpoint="/predict",
            features=features,
            prediction=np.random.uniform(0, 1),
            metadata={
                'business_metric': {
                    'revenue': np.random.uniform(10, 1000),
                    'conversion_rate': np.random.uniform(0, 1)
                }
            }
        )
        
        # Simulate occasional errors
        if np.random.random() < 0.1:
            try:
                raise ValueError("Invalid input features")
            except Exception as e:
                metrics.record_error(
                    endpoint="/predict",
                    error_type="validation_error",
                    error_message=str(e),
                    exception=e
                )
        
        time.sleep(1)
```

## Grafana Dashboard for ML Monitoring

```json
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": "-- Dashboard --",
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": 1,
  "links": [],
  "panels": [
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              },
              {
                "color": "red",
                "value": 0.5
              }
            ]
          },
          "unit": "s"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "id": 2,
      "options": {
        "legend": {
          "calcs": [
            "mean",
            "max",
            "lastNotNull"
          ],
          "displayMode": "table",
          "placement": "bottom",
          "showLegend": true
        },
        "tooltip": {
          "mode": "single",
          "sort": "none"
        }
      },
      "targets": [
        {
          "expr": "rate(model_prediction_latency_seconds_sum{model_name=\"fraud_detection"}[5m]) / rate(model_prediction_latency_seconds_count{model_name=\"fraud_detection"}[5m])",
          "legendFormat": "{{endpoint}} - p99",
          "refId": "A"
        }
      ],
      "title": "Prediction Latency (p99)",
      "type": "timeseries"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "thresholds"
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              },
              {
                "color": "red",
                "value": 0.5
              }
            ]
          },
          "unit": "percentunit"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 0
      },
      "id": 3,
      "options": {
        "orientation": "auto",
        "reduceOptions": {
          "calcs": [
            "lastNotNull"
          ],
          "fields": "",
          "values": false
        },
        "showThresholdLabels": false,
        "showThresholdMarkers": true
      },
      "pluginVersion": "9.0.0",
      "targets": [
        {
          "expr": "sum(rate(model_prediction_errors_total{model_name=\"fraud_detection"}[5m])) / sum(rate(model_requests_total{model_name=\"fraud_detection"}[5m]))",
          "legendFormat": "Error Rate",
          "refId": "A"
        }
      ],
      "title": "Error Rate",
      "type": "gauge"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              }
            ]
          },
          "unit": "short"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 8
      },
      "id": 4,
      "options": {
        "legend": {
          "calcs": [],
          "displayMode": "list",
          "placement": "bottom",
          "showLegend": true
        },
        "tooltip": {
          "mode": "single",
          "sort": "none"
        }
      },
      "targets": [
        {
          "expr": "sum(rate(model_requests_total{model_name=\"fraud_detection"}[5m])) by (endpoint)",
          "legendFormat": "{{endpoint}}",
          "refId": "A"
        }
      ],
      "title": "Request Rate",
      "type": "timeseries"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              },
              {
                "color": "red",
                "value": 0.8
              }
            ]
          },
          "unit": "percentunit"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 8
      },
      "id": 5,
      "options": {
        "legend": {
          "calcs": [
            "mean",
            "max"
          ],
          "displayMode": "table",
          "placement": "bottom",
          "showLegend": true
        },
        "tooltip": {
          "mode": "single",
          "sort": "none"
        }
      },
      "targets": [
        {
          "expr": "model_feature_drift{model_name=\"fraud_detection"}",
          "legendFormat": "{{feature_name}}",
          "refId": "A"
        }
      ],
      "title": "Feature Drift",
      "type": "timeseries"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              }
            ]
          },
          "unit": "short"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 16
      },
      "id": 6,
      "options": {
        "legend": {
          "calcs": [],
          "displayMode": "list",
          "placement": "bottom",
          "showLegend": true
        },
        "tooltip": {
          "mode": "single",
          "sort": "none"
        }
      },
      "targets": [
        {
          "expr": "model_business_metric{model_name=\"fraud_detection"}",
          "legendFormat": "{{metric_name}}",
          "refId": "A"
        }
      ],
      "title": "Business Metrics",
      "type": "timeseries"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              }
            ]
          },
          "unit": "short"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 16
      },
      "id": 7,
      "options": {
        "legend": {
          "calcs": [],
          "displayMode": "list",
          "placement": "bottom",
          "showLegend": true
        },
        "tooltip": {
          "mode": "single",
          "sort": "none"
        }
      },
      "targets": [
        {
          "expr": "sum by (error_type) (rate(model_prediction_errors_total{model_name=\"fraud_detection"}[5m]))",
          "legendFormat": "{{error_type}}",
          "refId": "A"
        }
      ],
      "title": "Error Types",
      "type": "timeseries"
    }
  ],
  "refresh": "5s",
  "schemaVersion": 36,
  "style": "dark",
  "tags": ["ml", "monitoring"],
  "templating": {
    "list": [
      {
        "current": {
          "selected": false,
          "text": "fraud_detection",
          "value": "fraud_detection"
        },
        "hide": 0,
        "includeAll": false,
        "label": "Model",
        "multi": false,
        "name": "model",
        "options": [
          {
            "selected": true,
            "text": "fraud_detection",
            "value": "fraud_detection"
          }
        ],
        "query": "fraud_detection",
        "queryValue": "",
        "skipUrlSync": false,
        "type": "custom"
      }
    ]
  },
  "time": {
    "from": "now-6h",
    "to": "now"
  },
  "timepicker": {
    "refresh_intervals": [
      "5s",
      "10s",
      "30s",
      "1m",
      "5m",
      "15m",
      "30m",
      "1h",
      "2h",
      "1d"
    ]
  },
  "timezone": "browser",
  "title": "ML Model Monitoring Dashboard",
  "version": 1,
  "weekStart": ""
}
