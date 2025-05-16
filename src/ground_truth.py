"""
Ground truth data management and integration utilities
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box, Point, Polygon
import logging
from sklearn.model_selection import train_test_split

@dataclass
class GroundTruthSource:
    name: str
    type: str  # 'field_survey', 'archaeological_db', 'historical_map'
    confidence: float
    date: str
    attribution: str

@dataclass
class ValidationSite:
    geometry: Any  # Shapely geometry
    site_type: str
    dating_method: Optional[str] = None
    dating_confidence: Optional[float] = None
    artifacts: Optional[List[str]] = None
    source: Optional[GroundTruthSource] = None

class GroundTruthManager:
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger('xingu_pipeline.ground_truth')
        self.config = config
        
        # Initialize confidence thresholds
        self.confidence_thresholds = {
            'field_survey': 0.9,
            'archaeological_db': 0.8,
            'historical_map': 0.7
        }
    
    def load_ground_truth_data(self, data_path: Path) -> Dict[str, Any]:
        """Load and validate ground truth data from multiple sources"""
        ground_truth = {
            'sites': [],
            'sources': [],
            'validation_splits': {}
        }
        
        try:
            # Load field survey data
            field_surveys = self._load_field_surveys(data_path / 'field_surveys')
            ground_truth['sites'].extend(field_surveys)
            
            # Load archaeological database records
            db_records = self._load_archaeological_db(data_path / 'archaeological_db')
            ground_truth['sites'].extend(db_records)
            
            # Load digitized historical maps
            historical_sites = self._load_historical_maps(data_path / 'historical_maps')
            ground_truth['sites'].extend(historical_sites)
            
            # Create validation splits
            ground_truth['validation_splits'] = self._create_validation_splits(
                ground_truth['sites']
            )
            
            return ground_truth
            
        except Exception as e:
            self.logger.error(f"Failed to load ground truth data: {str(e)}")
            raise
    
    def create_training_masks(self,
                            sites: List[ValidationSite],
                            bounds: Dict[str, float],
                            resolution: float) -> np.ndarray:
        """Create rasterized training masks from ground truth sites"""
        # Calculate dimensions
        width = int((bounds['max_lon'] - bounds['min_lon']) / resolution)
        height = int((bounds['max_lat'] - bounds['min_lat']) / resolution)
        
        # Create transform
        transform = rasterio.transform.from_bounds(
            bounds['min_lon'], bounds['min_lat'],
            bounds['max_lon'], bounds['max_lat'],
            width, height
        )
        
        # Create shapes for rasterization
        shapes = [(site.geometry, 1) for site in sites]
        
        # Rasterize sites
        mask = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        
        return mask
    
    def _load_field_surveys(self, survey_path: Path) -> List[ValidationSite]:
        """Load field survey data"""
        sites = []
        
        if not survey_path.exists():
            return sites
        
        # Load survey GeoJSON files
        for file in survey_path.glob('*.geojson'):
            try:
                gdf = gpd.read_file(file)
                source = GroundTruthSource(
                    name=file.stem,
                    type='field_survey',
                    confidence=self.confidence_thresholds['field_survey'],
                    date=gdf.metadata.get('survey_date', 'unknown'),
                    attribution=gdf.metadata.get('surveyor', 'unknown')
                )
                
                for _, row in gdf.iterrows():
                    sites.append(ValidationSite(
                        geometry=row.geometry,
                        site_type=row['site_type'],
                        dating_method=row.get('dating_method'),
                        dating_confidence=row.get('dating_confidence'),
                        artifacts=row.get('artifacts', '').split(','),
                        source=source
                    ))
                    
            except Exception as e:
                self.logger.warning(f"Failed to load survey file {file}: {str(e)}")
                continue
        
        return sites
    
    def _load_archaeological_db(self, db_path: Path) -> List[ValidationSite]:
        """Load archaeological database records"""
        sites = []
        
        if not db_path.exists():
            return sites
        
        # Load database files (assuming CSV format)
        for file in db_path.glob('*.csv'):
            try:
                df = pd.read_csv(file)
                source = GroundTruthSource(
                    name=file.stem,
                    type='archaeological_db',
                    confidence=self.confidence_thresholds['archaeological_db'],
                    date=df.metadata.get('last_updated', 'unknown'),
                    attribution=df.metadata.get('institution', 'unknown')
                )
                
                # Convert coordinates to geometries
                geometries = [
                    Point(row['longitude'], row['latitude'])
                    for _, row in df.iterrows()
                ]
                
                gdf = gpd.GeoDataFrame(df, geometry=geometries)
                
                for _, row in gdf.iterrows():
                    sites.append(ValidationSite(
                        geometry=row.geometry,
                        site_type=row['site_type'],
                        dating_method=row.get('dating_method'),
                        dating_confidence=row.get('dating_confidence'),
                        artifacts=row.get('artifacts', '').split(','),
                        source=source
                    ))
                    
            except Exception as e:
                self.logger.warning(f"Failed to load database file {file}: {str(e)}")
                continue
        
        return sites
    
    def _load_historical_maps(self, maps_path: Path) -> List[ValidationSite]:
        """Load digitized historical maps"""
        sites = []
        
        if not maps_path.exists():
            return sites
        
        # Load georeferenced historical map features
        for file in maps_path.glob('*.geojson'):
            try:
                gdf = gpd.read_file(file)
                source = GroundTruthSource(
                    name=file.stem,
                    type='historical_map',
                    confidence=self.confidence_thresholds['historical_map'],
                    date=gdf.metadata.get('map_date', 'unknown'),
                    attribution=gdf.metadata.get('cartographer', 'unknown')
                )
                
                for _, row in gdf.iterrows():
                    sites.append(ValidationSite(
                        geometry=row.geometry,
                        site_type=row['site_type'],
                        source=source
                    ))
                    
            except Exception as e:
                self.logger.warning(f"Failed to load historical map {file}: {str(e)}")
                continue
        
        return sites
    
    def _create_validation_splits(self,
                                sites: List[ValidationSite],
                                val_ratio: float = 0.2,
                                test_ratio: float = 0.1) -> Dict[str, List[ValidationSite]]:
        """Create train/validation/test splits of ground truth data"""
        # Convert to GeoDataFrame for spatial operations
        gdf = gpd.GeoDataFrame([{
            'geometry': site.geometry,
            'site_type': site.site_type,
            'confidence': site.source.confidence
        } for site in sites])
        
        # Stratify by site type and ensure spatial distribution
        train_idx, temp_idx = train_test_split(
            range(len(sites)),
            test_size=(val_ratio + test_ratio),
            stratify=gdf['site_type']
        )
        
        val_size = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=val_size,
            stratify=gdf.iloc[temp_idx]['site_type']
        )
        
        return {
            'train': [sites[i] for i in train_idx],
            'val': [sites[i] for i in val_idx],
            'test': [sites[i] for i in test_idx]
        }
    
    def evaluate_prediction(self,
                          predictions: gpd.GeoDataFrame,
                          ground_truth: List[ValidationSite],
                          iou_threshold: float = 0.5) -> Dict[str, float]:
        """Evaluate model predictions against ground truth data"""
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'mean_iou': 0.0
        }
        
        try:
            # Convert ground truth to GeoDataFrame
            gt_gdf = gpd.GeoDataFrame([{
                'geometry': site.geometry,
                'site_type': site.site_type
            } for site in ground_truth])
            
            # Calculate IoU for each prediction
            ious = []
            true_positives = 0
            
            for pred_idx, pred_row in predictions.iterrows():
                max_iou = 0
                for gt_idx, gt_row in gt_gdf.iterrows():
                    if pred_row.geometry.intersects(gt_row.geometry):
                        intersection = pred_row.geometry.intersection(gt_row.geometry).area
                        union = pred_row.geometry.union(gt_row.geometry).area
                        iou = intersection / union
                        max_iou = max(max_iou, iou)
                
                if max_iou > iou_threshold:
                    true_positives += 1
                ious.append(max_iou)
            
            # Calculate metrics
            if len(predictions) > 0:
                metrics['precision'] = true_positives / len(predictions)
            if len(ground_truth) > 0:
                metrics['recall'] = true_positives / len(ground_truth)
            
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1_score'] = (
                    2 * metrics['precision'] * metrics['recall'] /
                    (metrics['precision'] + metrics['recall'])
                )
            
            metrics['mean_iou'] = np.mean(ious) if ious else 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate predictions: {str(e)}")
            raise
        
        return metrics
