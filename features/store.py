# -*- coding: utf-8 -*-
"""
Feature Store
-------------
Store and manage extracted features.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime

from core.interfaces import FeatureSet
from core.exceptions import FeatureExtractionError


class FeatureStore:
    """
    Store and manage feature sets.
    Supports caching and persistence.
    """

    def __init__(self, storage_path: Path = None):
        self._store: Dict[str, FeatureSet] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self.storage_path = storage_path or Path("data/features")

    def save(
        self,
        name: str,
        feature_set: FeatureSet,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Save feature set to store.

        Args:
            name: Unique identifier
            feature_set: FeatureSet to store
            metadata: Additional metadata
        """
        self._store[name] = feature_set
        self._metadata[name] = {
            'created_at': datetime.now().isoformat(),
            'num_samples': len(feature_set.data),
            'num_features': len(feature_set.feature_names),
            'feature_names': feature_set.feature_names,
            **(metadata or {}),
            **(feature_set.metadata or {})
        }

    def load(self, name: str) -> FeatureSet:
        """
        Load feature set from store.

        Args:
            name: Feature set identifier

        Returns:
            FeatureSet
        """
        if name not in self._store:
            raise FeatureExtractionError(f"Feature set '{name}' not found")
        return self._store[name]

    def get_data(self, name: str) -> pd.DataFrame:
        """Get DataFrame from stored feature set."""
        return self.load(name).data.copy()

    def list(self) -> List[str]:
        """List all stored feature sets."""
        return list(self._store.keys())

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for feature set."""
        return self._metadata.get(name, {})

    def delete(self, name: str) -> None:
        """Delete feature set from store."""
        if name in self._store:
            del self._store[name]
            del self._metadata[name]

    def export_csv(self, name: str, path: str) -> None:
        """Export feature set to CSV."""
        df = self.get_data(name)
        df.to_csv(path)

    def export_excel(self, name: str, path: str) -> None:
        """Export feature set to Excel."""
        df = self.get_data(name)
        df.to_excel(path)

    def persist(self, name: str) -> Path:
        """
        Persist feature set to disk.

        Returns:
            Path to saved file
        """
        self.storage_path.mkdir(parents=True, exist_ok=True)

        feature_set = self.load(name)

        # Save data
        data_path = self.storage_path / f"{name}.parquet"
        feature_set.data.to_parquet(data_path)

        # Save metadata
        meta_path = self.storage_path / f"{name}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(self._metadata[name], f, indent=2, default=str)

        return data_path

    def load_from_disk(self, name: str) -> FeatureSet:
        """
        Load feature set from disk.

        Args:
            name: Feature set name

        Returns:
            FeatureSet
        """
        data_path = self.storage_path / f"{name}.parquet"
        meta_path = self.storage_path / f"{name}_meta.json"

        if not data_path.exists():
            raise FeatureExtractionError(
                f"Feature set '{name}' not found on disk"
            )

        # Load data
        df = pd.read_parquet(data_path)

        # Load metadata
        metadata = {}
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

        feature_names = metadata.get('feature_names', list(df.columns))

        feature_set = FeatureSet(
            data=df,
            feature_names=feature_names,
            index_column=metadata.get('index_column', df.index.name),
            metadata=metadata
        )

        # Also store in memory
        self._store[name] = feature_set
        self._metadata[name] = metadata

        return feature_set
