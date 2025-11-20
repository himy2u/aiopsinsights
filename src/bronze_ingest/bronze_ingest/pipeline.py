"""Bronze to Silver data pipeline for healthcare triage system."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.data_context import BaseDataContext
from great_expectations.data_context.types.base import DataContextConfig

from .models import BronzeRecord, SilverRecord, Gender, InputChannel
from .pii import pii_detector


class BronzeToSilverPipeline:
    """Pipeline to process raw bronze records into normalized silver records."""

    def __init__(
        self,
        base_path: Union[str, Path],
        ml_pii_detection: bool = False,
        validate: bool = True,
    ):
        """Initialize the pipeline.

        Args:
            base_path: Base directory for data storage (bronze/silver)
            ml_pii_detection: Whether to use ML-based PII detection (slower but more accurate)
            validate: Whether to validate data against expectations
        """
        self.base_path = Path(base_path)
        self.ml_pii_detection = ml_pii_detection
        self.validate = validate
        self.context = self._init_ge_context()

    def _init_ge_context(self) -> BaseDataContext:
        """Initialize Great Expectations context."""
        context = BaseDataContext(
            project_config=DataContextConfig(
                config_version=1.0,
                plugins_directory=None,
                config_variables_file_path=None,
                datasources={
                    "bronze_datasource": {
                        "class_name": "PandasDatasource",
                        "module_name": "great_expectations.datasource",
                    },
                    "silver_datasource": {
                        "class_name": "PandasDatasource",
                        "module_name": "great_expectations.datasource",
                    },
                },
                stores={
                    "expectations_store": {
                        "class_name": "ExpectationsStore",
                        "store_backend": {
                            "class_name": "TupleFilesystemStoreBackend",
                            "base_directory": str(
                                self.base_path / "expectations"
                            ),
                        },
                    },
                    "validations_store": {
                        "class_name": "ValidationsStore",
                        "store_backend": {
                            "class_name": "TupleFilesystemStoreBackend",
                            "base_directory": str(
                                self.base_path / "validations"
                            ),
                        },
                    },
                },
                expectations_store_name="expectations_store",
                validations_store_name="validations_store",
                evaluation_parameter_store_name="evaluation_parameter_store",
                data_docs_sites={
                    "local_site": {
                        "class_name": "SiteBuilder",
                        "store_backend": {
                            "class_name": "TupleFilesystemStoreBackend",
                            "base_directory": str(
                                self.base_path / "data_docs"
                            ),
                        },
                    }
                },
            )
        )
        return context

    def _save_bronze(self, records: List[BronzeRecord]) -> str:
        """Save records to bronze layer.

        Args:
            records: List of bronze records to save

        Returns:
            Path to the saved bronze file
        """
        bronze_path = self.base_path / "bronze"
        bronze_path.mkdir(parents=True, exist_ok=True)

        # Create timestamp-based partition
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_file = bronze_path / f"ingest_{timestamp}.parquet"

        # Convert to DataFrame and save as Parquet
        df = pd.DataFrame([record.dict() for record in records])
        df.to_parquet(output_file, index=False)

        return str(output_file)

    def _process_to_silver(self, bronze_file: str) -> str:
        """Process bronze file to silver layer.

        Args:
            bronze_file: Path to bronze file

        Returns:
            Path to the saved silver file
        """
        # Read bronze data
        df = pd.read_parquet(bronze_file)

        # Apply transformations
        silver_records = []
        for _, row in df.iterrows():
            try:
                bronze = BronzeRecord(**row.to_dict())
                
                # Redact PII from patient text
                redacted_text = pii_detector.redact_text(bronze.patient_text)
                
                # Create silver record
                silver = SilverRecord(
                    id=bronze.id,
                    timestamp=bronze.timestamp,
                    source=bronze.channel,
                    symptom_text=redacted_text,
                    demographics={
                        "age": bronze.age,
                        "gender": bronze.gender.value,
                        "zip": bronze.zip_code,
                    },
                    pii_redacted=True,
                )
                silver_records.append(silver.dict())
            except Exception as e:
                print(f"Error processing record {row.get('id', 'unknown')}: {e}")

        # Save to silver
        silver_path = self.base_path / "silver"
        silver_path.mkdir(parents=True, exist_ok=True)
        
        silver_file = silver_path / f"silver_{Path(bronze_file).name}"
        pd.DataFrame(silver_records).to_parquet(silver_file, index=False)
        
        return str(silver_file)

    def _validate_data(self, df: pd.DataFrame, layer: str) -> bool:
        """Validate data against expectations.

        Args:
            df: DataFrame to validate
            layer: Data layer ('bronze' or 'silver')

        Returns:
            True if validation passes, False otherwise
        """
        if not self.validate:
            return True

        try:
            batch_request = RuntimeBatchRequest(
                datasource_name=f"{layer}_datasource",
                data_connector_name="default_runtime_data_connector",
                data_asset_name=f"{layer}_data",
                runtime_parameters={"batch_data": df},
                batch_identifiers={"pipeline_stage": "ingestion"},
            )

            # Get or create checkpoint
            checkpoint_name = f"{layer}_checkpoint"
            if checkpoint_name not in self.context.list_checkpoints():
                self.context.add_checkpoint(
                    name=checkpoint_name,
                    config_version=1.0,
                    class_name="SimpleCheckpoint",
                    validations=[
                        {
                            "batch_request": batch_request,
                            "expectation_suite_name": f"{layer}_suite",
                        }
                    ],
                )

            # Run validation
            result = self.context.run_checkpoint(
                checkpoint_name=checkpoint_name,
                batch_request=batch_request,
            )

            if not result["success"]:
                print(f"Validation failed for {layer} data:")
                for res in result["run_results"].values():
                    print(res["validation_result"].to_json_dict())
                return False

            return True

        except Exception as e:
            print(f"Error during {layer} validation: {e}")
            return False

    def process(self, records: List[Dict[str, Any]]) -> Dict[str, str]:
        """Process records through the pipeline.

        Args:
            records: List of raw record dictionaries

        Returns:
            Dictionary with paths to saved files
        """
        # Convert to Pydantic models
        bronze_records = [BronzeRecord(**record) for record in records]
        
        # Save to bronze
        bronze_file = self._save_bronze(bronze_records)
        
        # Convert to DataFrame for validation
        bronze_df = pd.DataFrame([r.dict() for r in bronze_records])
        
        # Validate bronze data
        if self.validate and not self._validate_data(bronze_df, "bronze"):
            raise ValueError("Bronze data validation failed")
        
        # Process to silver
        silver_file = self._process_to_silver(bronze_file)
        
        # Validate silver data
        silver_df = pd.read_parquet(silver_file)
        if self.validate and not self._validate_data(silver_df, "silver"):
            raise ValueError("Silver data validation failed")
        
        return {
            "bronze": bronze_file,
            "silver": silver_file,
        }
