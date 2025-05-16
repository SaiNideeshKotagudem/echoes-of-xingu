"""
Main pipeline orchestrator for archaeological site discovery
"""

import os
import json
from pathlib import Path
import logging
import logging.handlers
from typing import Dict, List, Any
import torch
import geopandas as gpd

from .config import ConfigManager
from .satellite_processing import SatelliteDataProcessor
from .text_processing import HistoricalTextProcessor
from .models import get_model
from .geospatial import GeospatialIntegrator
from .visualization import ArchaeologicalVisualizer
from .lidar_processing import LidarProcessor
from .data_validation import DataValidator, ValidationResult

def setup_logging(log_dir: str = 'logs') -> logging.Logger:
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('xingu_pipeline')
    logger.setLevel(logging.INFO)
    
    # File handler with rotation
    log_file = Path(log_dir) / 'pipeline.log'
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=1024*1024, backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatters and add it to the handlers
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_str)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class Pipeline:
    def __init__(self, config_path: str):
        # Set up logging
        self.logger = setup_logging()
        self.logger.info(f"Initializing pipeline with config from {config_path}")
        
        # Initialize components
        try:
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.config
            
            # Initialize processors
            self.satellite_processor = SatelliteDataProcessor(
                api_key=os.getenv('PLANETARY_COMPUTER_API_KEY')
            )
            self.text_processor = HistoricalTextProcessor()
            self.lidar_processor = LidarProcessor(self.config.get('lidar', {}))
            self.geospatial_integrator = GeospatialIntegrator()
            self.visualizer = ArchaeologicalVisualizer()
            
            # Initialize validator
            self.validator = DataValidator()
            
            # Create necessary directories
            self.config_manager.create_paths()
            self.logger.info("Pipeline initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {str(e)}")
            raise
    
    def run(self):
        """Execute the complete pipeline"""
        try:
            # 1. Data Collection
            self.logger.info("Starting data collection...")
            satellite_data = self._collect_satellite_data()
            historical_data = self._process_historical_data()
            
            # 2. Preprocessing
            self.logger.info("Preprocessing data...")
            processed_data = self._preprocess_data(satellite_data, historical_data)
            
            # 3. Model Prediction
            self.logger.info("Running model predictions...")
            site_predictions = self._run_model_predictions(processed_data)
            
            # 4. Validation
            self.logger.info("Validating predictions...")
            validated_sites = self._validate_predictions(site_predictions)
            
            # 5. Visualization
            self.logger.info("Creating visualizations...")
            visualization_results = self._create_visualizations(
                validated_sites,
                processed_data,
                historical_data
            )
            
            # 6. Save Results
            self.logger.info("Saving results...")
            self._save_results(validated_sites, visualization_results)
            
            self.logger.info("Pipeline completed successfully!")
            return validated_sites, visualization_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise
    
    def _collect_satellite_data(self) -> Dict[str, Any]:
        """Collect satellite imagery and LIDAR data"""
        study_area = self.config_manager.get_study_area()
        satellite_config = self.config_manager.get_satellite_config()
        
        data = {
            'satellite': {},
            'lidar': {}
        }
        
        # Collect satellite data
        for source in satellite_config['sources']:
            try:
                if source == 'sentinel2':
                    data['satellite']['sentinel2'] = self.satellite_processor.fetch_sentinel2_imagery(
                        study_area['bounds'],
                        study_area['time_range']
                    )
                elif source == 'landsat8':
                    data['satellite']['landsat8'] = self.satellite_processor.fetch_landsat8_imagery(
                        study_area['bounds'],
                        study_area['time_range']
                    )
                
                # Validate satellite data
                validation_result = self.validator.validate_satellite_data(
                    data['satellite'][source],
                    source
                )
                
                if not validation_result.is_valid:
                    self.logger.warning(
                        f"Validation issues with {source} data: {validation_result.messages}"
                    )
                    
            except Exception as e:
                self.logger.error(f"Failed to collect {source} data: {str(e)}")
                continue
        
        # Collect LIDAR data if available
        lidar_paths = list(Path(study_area.get('lidar_path', '')).glob('*.laz'))
        if lidar_paths:
            try:
                data['lidar'] = self.lidar_processor.process_lidar_data(
                    lidar_paths,
                    study_area['bounds']
                )
                
                # Validate LIDAR data
                validation_result = self.validator.validate_lidar_data(data['lidar'])
                if not validation_result.is_valid:
                    self.logger.warning(
                        f"Validation issues with LIDAR data: {validation_result.messages}"
                    )
                    
            except Exception as e:
                self.logger.error(f"Failed to process LIDAR data: {str(e)}")
        
        return data
    
    def _process_historical_data(self) -> Dict[str, Any]:
        """Process historical texts and maps"""
        paths = self.config_manager.get_paths()
        
        historical_data = {
            'texts': [],
            'maps': [],
            'linguistics': []
        }
        
        # Process each data type
        for data_type in historical_data.keys():
            data_path = Path(paths['data']['raw']) / data_type
            if data_path.exists():
                for file in data_path.glob('*.*'):
                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if data_type == 'texts':
                            processed = self.text_processor.process_indigenous_narrative(content)
                        elif data_type == 'maps':
                            processed = self.text_processor.extract_spatial_markers(content)
                        historical_data[data_type].append(processed)
        
        return historical_data
    
    def _preprocess_data(self, 
                        satellite_data: Dict[str, Any],
                        historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess all collected data"""
        processed_data = {
            'satellite': {},
            'lidar': {},
            'historical': historical_data
        }
        
        # Preprocess satellite imagery
        for source, data in satellite_data.get('satellite', {}).items():
            self.logger.info(f"Preprocessing {source} data...")
            processed, metadata = self.satellite_processor.preprocess_imagery(data)
            processed_data['satellite'][source] = {
                'data': processed,
                'metadata': metadata
            }
        
        # Process LIDAR data if available
        if 'lidar' in satellite_data and satellite_data['lidar']:
            self.logger.info("Processing LIDAR derivatives...")
            processed_data['lidar'] = satellite_data['lidar']
            
            # Add DEM to processed data for visualization
            if 'dem' in processed_data['lidar']:
                processed_data['dem'] = processed_data['lidar']['dem']
        
        # Validate extracted locations from historical data
        if historical_data.get('texts'):
            validation_result = self.validator.validate_extracted_locations(
                [loc for text in historical_data['texts'] for loc in text]
            )
            if not validation_result.is_valid:
                self.logger.warning(
                    f"Validation issues with extracted locations: {validation_result.messages}"
                )
        
        return processed_data

    def train_model(self) -> None:
        """Train the archaeological site detection model"""
        from .training import Trainer
        from .data_preparation import DatasetPreparator, TileDimensions
        
        self.logger.info("Starting model training pipeline...")
        
        try:
            # Initialize data preparator
            data_preparator = DatasetPreparator(
                data_root=Path(self.config_manager.get_paths()['data']['processed']),
                config_path=str(self.config_manager.config_path)
            )
            
            # Prepare dataset
            self.logger.info("Preparing training dataset...")
            tile_dims = TileDimensions(
                width=224,  # Match ViT input size
                height=224,
                overlap=32
            )
            data_preparator.prepare_dataset(tile_dims)
            
            # Initialize trainer
            trainer = Trainer(
                config_path=str(self.config_manager.config_path),
                data_root=Path(self.config_manager.get_paths()['data']['processed'])
            )
            
            # Train model
            self.logger.info("Starting model training...")
            metrics_history = trainer.train()
            
            # Save training metrics
            metrics_path = Path(self.config_manager.get_paths()['results']) / 'training_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics_history, f)
            
            self.logger.info("Model training completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise
    
    def _run_model_predictions(self, processed_data: Dict[str, Any]) -> gpd.GeoDataFrame:
        """Run ML model to predict archaeological sites"""
        model_config = self.config_manager.get_model_config()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get model
        model, _ = get_model(model_config)
        model = model.to(device)
        
        # Load trained weights
        weights_path = Path(self.config_manager.get_paths()['models']) / 'model.pth'
        if not weights_path.exists():
            self.logger.warning("No trained model found. Training new model...")
            self.train_model()
        
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        
        # Prepare input data
        input_data = torch.from_numpy(processed_data['satellite']['sentinel2']['data']).float()
        input_data = input_data.unsqueeze(0).to(device)  # Add batch dimension
        
        # Run predictions
        with torch.no_grad():
            predictions = model(input_data)
            predictions = torch.sigmoid(predictions)  # Convert to probabilities
        
        # Convert predictions to GeoDataFrame
        site_predictions = self.geospatial_integrator.create_site_predictions(
            predictions.cpu().numpy()[0],  # Remove batch dimension
            self.config_manager.get_study_area()['bounds']
        )
        
        return site_predictions
    
    def _validate_predictions(self, site_predictions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Validate predicted sites"""
        validation_config = self.config_manager.get_validation_config()
        
        # Validate based on hydrology
        validated_sites = self.geospatial_integrator.integrate_hydrology(
            site_predictions,
            max_distance=validation_config['hydrology']['max_distance']
        )
        
        # Validate based on ring village patterns
        validated_sites = self.geospatial_integrator.validate_ring_village_pattern(
            validated_sites,
            radius_range=validation_config['ring_village']['radius_range'],
            min_points=validation_config['ring_village']['min_points']
        )
        
        return validated_sites
    
    def _create_visualizations(self,
                             validated_sites: gpd.GeoDataFrame,
                             processed_data: Dict[str, Any],
                             historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create all visualizations"""
        vis_config = self.config_manager.get_visualization_config()
        
        visualizations = {
            'kepler_map': self.visualizer.create_kepler_map({
                'sites': validated_sites,
                'historical': historical_data
            }),
            'temporal_vis': self.visualizer.create_temporal_visualization(
                validated_sites,
                historical_data.get('rivers', {}),
                historical_data.get('linguistics', None)
            )
        }
        
        # Add 3D visualization if DEM data is available
        if 'dem' in processed_data:
            visualizations['3d_terrain'] = self.visualizer.create_3d_terrain_view(
                processed_data['dem'],
                validated_sites,
                resolution=vis_config['3d_terrain']['resolution']
            )
        
        return visualizations
    
    def _save_results(self,
                     validated_sites: gpd.GeoDataFrame,
                     visualizations: Dict[str, Any]) -> None:
        """Save all results to disk"""
        paths = self.config_manager.get_paths()
        
        # Save validated sites
        validated_sites.to_file(
            Path(paths['data']['outputs']) / 'validated_sites.geojson',
            driver='GeoJSON'
        )
        
        # Save visualizations
        vis_path = Path(paths['data']['outputs']) / 'visualizations'
        vis_path.mkdir(exist_ok=True)
        
        for name, vis in visualizations.items():
            if name == 'kepler_map':
                vis.save_to_html(str(vis_path / 'kepler_map.html'))
            elif name == 'temporal_vis':
                vis.write_html(str(vis_path / 'temporal_visualization.html'))
            elif name == '3d_terrain':
                vis.write_html(str(vis_path / '3d_terrain.html'))
