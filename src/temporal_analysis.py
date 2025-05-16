"""
LIDAR temporal analysis utilities
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Polygon, Point
from sklearn.metrics.pairwise import euclidean_distances
import logging
from dataclasses import dataclass

@dataclass
class TemporalFeature:
    feature_id: str
    geometry: Any
    attributes: Dict[str, Any]
    timestamps: List[datetime]
    confidence: float

class TemporalAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger('xingu_pipeline.temporal')
        self.config = config
        self.params = config['temporal_analysis']
    
    def analyze_temporal_changes(self, 
                               time_series_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze changes in LIDAR data over time"""
        try:
            # Sort data by timestamp
            time_series_data.sort(key=lambda x: x['timestamp'])
            
            # Extract and track features over time
            tracked_features = self._track_features(time_series_data)
            
            # Detect significant changes
            elevation_changes = self._detect_elevation_changes(time_series_data)
            
            # Analyze seasonal patterns if enabled
            seasonal_patterns = {}
            if self.params['seasonality']['vegetation_cycle']:
                seasonal_patterns['vegetation'] = self._analyze_vegetation_cycles(
                    time_series_data
                )
            if self.params['seasonality']['water_level']:
                seasonal_patterns['hydrology'] = self._analyze_water_levels(
                    time_series_data
                )
            
            return {
                'tracked_features': tracked_features,
                'elevation_changes': elevation_changes,
                'seasonal_patterns': seasonal_patterns
            }
            
        except Exception as e:
            self.logger.error(f"Temporal analysis failed: {str(e)}")
            raise
    
    def _track_features(self, time_series: List[Dict[str, Any]]) -> List[TemporalFeature]:
        """Track individual features across time periods"""
        tracked = []
        max_offset = self.params['feature_tracking']['max_spatial_offset']
        
        # Initialize feature tracking with first timestamp
        current_features = {}
        
        for period_data in time_series:
            new_features = period_data['archaeological_features']['features']
            timestamp = period_data['timestamp']
            
            if not current_features:
                # First period - initialize tracking
                for idx, feature in new_features.iterrows():
                    feature_id = f"feature_{idx}"
                    tracked.append(TemporalFeature(
                        feature_id=feature_id,
                        geometry=feature.geometry,
                        attributes={
                            'type': feature['type'],
                            'confidence': feature['confidence'],
                            'initial_area': feature.geometry.area
                        },
                        timestamps=[timestamp],
                        confidence=feature['confidence']
                    ))
                    current_features[feature_id] = feature.geometry.centroid
            else:
                # Match new features with existing tracks
                centroids_current = np.array([
                    [p.x, p.y] for p in current_features.values()
                ])
                centroids_new = np.array([
                    [f.geometry.centroid.x, f.geometry.centroid.y] 
                    for _, f in new_features.iterrows()
                ])
                
                # Calculate distances between all pairs
                distances = euclidean_distances(centroids_current, centroids_new)
                
                # Match features based on proximity
                matched_pairs = []
                while distances.size > 0 and np.min(distances) <= max_offset:
                    i, j = np.unravel_index(np.argmin(distances), distances.shape)
                    if distances[i, j] <= max_offset:
                        matched_pairs.append((list(current_features.keys())[i], j))
                        distances[i, :] = float('inf')
                        distances[:, j] = float('inf')
                
                # Update tracked features
                for feature_id, new_idx in matched_pairs:
                    new_feature = new_features.iloc[new_idx]
                    for tracked_feature in tracked:
                        if tracked_feature.feature_id == feature_id:
                            tracked_feature.timestamps.append(timestamp)
                            tracked_feature.geometry = new_feature.geometry
                            tracked_feature.attributes['last_area'] = new_feature.geometry.area
                            tracked_feature.attributes['area_change'] = (
                                new_feature.geometry.area / tracked_feature.attributes['initial_area']
                            )
                            tracked_feature.confidence = min(
                                tracked_feature.confidence,
                                new_feature['confidence']
                            )
                
                # Add new unmatched features as new tracks
                matched_new = {j for _, j in matched_pairs}
                for idx, feature in new_features.iterrows():
                    if idx not in matched_new:
                        feature_id = f"feature_{len(tracked)}_{idx}"
                        tracked.append(TemporalFeature(
                            feature_id=feature_id,
                            geometry=feature.geometry,
                            attributes={
                                'type': feature['type'],
                                'confidence': feature['confidence'],
                                'initial_area': feature.geometry.area
                            },
                            timestamps=[timestamp],
                            confidence=feature['confidence']
                        ))
                
                # Update current features for next iteration
                current_features = {
                    t.feature_id: t.geometry.centroid for t in tracked
                    if timestamp in t.timestamps
                }
        
        return tracked
    
    def _detect_elevation_changes(self, time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect significant changes in elevation between time periods"""
        min_change = self.params['change_detection']['min_elevation_change']
        changes = {
            'erosion': [],
            'accumulation': [],
            'disturbance': []
        }
        
        for i in range(len(time_series) - 1):
            dem1 = time_series[i]['dem']['data']
            dem2 = time_series[i + 1]['dem']['data']
            
            # Calculate elevation difference
            diff = dem2 - dem1
            
            # Classify changes
            erosion = diff < -min_change
            accumulation = diff > min_change
            
            # Create polygons for change areas
            changes['erosion'].extend(self._raster_to_polygons(
                erosion, 
                time_series[i]['dem']['transform'],
                abs(diff[erosion])
            ))
            changes['accumulation'].extend(self._raster_to_polygons(
                accumulation,
                time_series[i]['dem']['transform'],
                abs(diff[accumulation])
            ))
            
            # Detect potential disturbance (rapid elevation changes)
            disturbed = np.logical_or(erosion, accumulation)
            if np.any(disturbed):
                changes['disturbance'].extend(self._raster_to_polygons(
                    disturbed,
                    time_series[i]['dem']['transform'],
                    abs(diff[disturbed])
                ))
        
        return changes
    
    def _analyze_vegetation_cycles(self, time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze seasonal vegetation patterns"""
        vegetation_heights = []
        timestamps = []
        
        for data in time_series:
            if 'advanced_features' in data:
                veg_features = [
                    f for f in data['advanced_features']['vegetation']
                    if f.confidence > self.params['change_detection']['confidence_threshold']
                ]
                
                if veg_features:
                    mean_height = np.mean([
                        f.attributes['height'] for f in veg_features
                    ])
                    vegetation_heights.append(mean_height)
                    timestamps.append(data['timestamp'])
        
        if vegetation_heights:
            # Convert to pandas for time series analysis
            df = pd.DataFrame({
                'height': vegetation_heights
            }, index=pd.DatetimeIndex(timestamps))
            
            # Detect seasonality using rolling statistics
            return {
                'temporal_trend': self._calculate_trend(df),
                'seasonal_pattern': self._detect_seasonal_pattern(df),
                'data': df.to_dict()
            }
        
        return {}
    
    def _analyze_water_levels(self, time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze changes in water features and levels"""
        water_areas = []
        timestamps = []
        
        for data in time_series:
            if 'advanced_features' in data:
                # Look for water-related features in terrain analysis
                water_features = [
                    f for f in data['advanced_features']['terrain']
                    if f.attributes.get('type') == 'water'
                ]
                
                if water_features:
                    total_area = sum(f.geometry.area for f in water_features)
                    water_areas.append(total_area)
                    timestamps.append(data['timestamp'])
        
        if water_areas:
            df = pd.DataFrame({
                'area': water_areas
            }, index=pd.DatetimeIndex(timestamps))
            
            return {
                'temporal_trend': self._calculate_trend(df),
                'seasonal_pattern': self._detect_seasonal_pattern(df),
                'data': df.to_dict()
            }
        
        return {}
    
    def _calculate_trend(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate temporal trend in time series data"""
        x = np.arange(len(df))
        y = df.iloc[:, 0].values
        z = np.polyfit(x, y, 1)
        
        return {
            'slope': float(z[0]),
            'intercept': float(z[1])
        }
    
    def _detect_seasonal_pattern(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect seasonal patterns in time series data"""
        # Resample to monthly frequency
        monthly = df.resample('M').mean()
        
        # Calculate month-wise statistics
        monthly_stats = df.groupby(df.index.month).agg(['mean', 'std']).to_dict()
        
        # Detect periodicity using autocorrelation
        autocorr = pd.Series(df.iloc[:, 0]).autocorr(lag=12)
        
        return {
            'monthly_stats': monthly_stats,
            'seasonality_strength': float(autocorr)
        }
    
    def _raster_to_polygons(self, 
                           mask: np.ndarray, 
                           transform: Any,
                           values: np.ndarray) -> List[Dict[str, Any]]:
        """Convert raster mask to polygons with attributes"""
        import rasterio.features
        
        shapes = rasterio.features.shapes(
            mask.astype('uint8'),
            transform=transform
        )
        
        features = []
        for shape, value in shapes:
            if value == 1:  # Only convert positive mask values
                features.append({
                    'geometry': shape,
                    'properties': {
                        'mean_change': float(np.mean(values))
                    }
                })
        
        return features
