"""
Configuration management utilities with validation
"""

import json
from pathlib import Path
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class StudyAreaConfig:
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    time_range_start: str
    time_range_end: str
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'StudyAreaConfig':
        return cls(
            min_lon=d['bounds']['min_lon'],
            max_lon=d['bounds']['max_lon'],
            min_lat=d['bounds']['min_lat'],
            max_lat=d['bounds']['max_lat'],
            time_range_start=d['time_range']['start'],
            time_range_end=d['time_range']['end']
        )
    
    def validate(self) -> None:
        """Validate study area configuration"""
        if not (-180 <= self.min_lon <= 180 and -180 <= self.max_lon <= 180):
            raise ValueError("Longitude values must be between -180 and 180")
        if not (-90 <= self.min_lat <= 90 and -90 <= self.max_lat <= 90):
            raise ValueError("Latitude values must be between -90 and 90")
        if self.min_lon >= self.max_lon:
            raise ValueError("min_lon must be less than max_lon")
        if self.min_lat >= self.max_lat:
            raise ValueError("min_lat must be less than max_lat")
        
        # Validate dates
        try:
            start = datetime.strptime(self.time_range_start, '%Y-%m-%d')
            end = datetime.strptime(self.time_range_end, '%Y-%m-%d')
            if end <= start:
                raise ValueError("End date must be after start date")
        except ValueError as e:
            raise ValueError(f"Invalid date format: {str(e)}")

class ConfigManager:
    def __init__(self, config_path: str):
        self.logger = logging.getLogger('xingu_pipeline.config')
        self.config_path = Path(config_path)
        self.config = self._load_and_validate_config()
    
    def _load_and_validate_config(self) -> Dict[str, Any]:
        """Load and validate configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Validate study area
            study_area = StudyAreaConfig.from_dict(config['study_area'])
            study_area.validate()
            
            # Validate satellite data configuration
            self._validate_satellite_config(config['satellite_data'])
            
            # Validate model configuration
            self._validate_model_config(config['model'])
            
            self.logger.info("Configuration validation successful")
            return config
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            raise
    
    def _validate_satellite_config(self, config: Dict[str, Any]) -> None:
        """Validate satellite data configuration"""
        required_keys = ['sources', 'bands', 'cloud_cover_threshold']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in satellite config: {key}")
        
        if not 0 <= config['cloud_cover_threshold'] <= 100:
            raise ValueError("Cloud cover threshold must be between 0 and 100")
        
        for source in config['sources']:
            if source not in ['sentinel2', 'landsat8']:
                raise ValueError(f"Unsupported satellite data source: {source}")
    
    def _validate_model_config(self, config: Dict[str, Any]) -> None:
        """Validate model configuration"""
        required_keys = ['architecture', 'training']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in model config: {key}")
        
        if 'in_channels' not in config['architecture']:
            raise ValueError("Model architecture must specify in_channels")
        if 'num_classes' not in config['architecture']:
            raise ValueError("Model architecture must specify num_classes")
    
    def get_study_area(self) -> Dict[str, Any]:
        """Get study area configuration"""
        return self.config['study_area']
    
    def get_satellite_config(self) -> Dict[str, Any]:
        """Get satellite data configuration"""
        return self.config['satellite_data']
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model architecture and training configuration"""
        return self.config['model']
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get site validation configuration"""
        return self.config['site_validation']
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration"""
        return self.config['visualization']
    
    def get_paths(self) -> Dict[str, str]:
        """Get data and model paths"""
        return self.config['paths']
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = deep_update(d[k], v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, updates)
        self._save_config()
    
    def _save_config(self) -> None:
        """Save configuration to JSON file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def create_paths(self) -> None:
        """Create all necessary directories defined in the configuration"""
        for category, path in self.get_paths().items():
            if isinstance(path, dict):
                for subcategory, subpath in path.items():
                    Path(subpath).mkdir(parents=True, exist_ok=True)
            else:
                Path(path).mkdir(parents=True, exist_ok=True)
