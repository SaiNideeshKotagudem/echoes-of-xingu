"""
Data validation and quality assurance utilities
"""

import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Optional
import rasterio
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    messages: List[str]
    details: Optional[Dict[str, Any]] = None

class DataValidator:
    def __init__(self):
        self.logger = logging.getLogger('xingu_pipeline.validation')
        
        # Define validation thresholds
        self.thresholds = {
            'satellite': {
                'min_coverage': 0.8,  # Minimum valid data coverage
                'max_cloud_cover': 20.0,  # Maximum cloud cover percentage
                'required_bands': {
                    'sentinel2': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
                    'landsat8': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
                }
            },
            'lidar': {
                'min_point_density': 2.0,  # points per square meter
                'max_gap_size': 10.0,  # meters
                'required_classes': [2, 3, 4, 5]  # Required point classifications
            },
            'text': {
                'min_confidence': 0.7,  # Minimum confidence for extracted locations
                'min_context_length': 50  # Minimum context length in characters
            }
        }
    
    def validate_satellite_data(self,
                              data: Dict[str, Any],
                              source: str) -> ValidationResult:
        """Validate satellite imagery data"""
        messages = []
        details = {}
        
        try:
            # Check required bands
            required_bands = set(self.thresholds['satellite']['required_bands'][source])
            available_bands = set(data['data'].keys())
            missing_bands = required_bands - available_bands
            
            if missing_bands:
                messages.append(f"Missing required bands: {missing_bands}")
            
            # Check data coverage
            coverage = self._calculate_data_coverage(data['data'])
            details['coverage'] = coverage
            
            if coverage < self.thresholds['satellite']['min_coverage']:
                messages.append(
                    f"Insufficient data coverage: {coverage:.2f} < "
                    f"{self.thresholds['satellite']['min_coverage']}"
                )
            
            # Check cloud cover
            cloud_cover = data['metadata'].get('cloud_cover', 100.0)
            details['cloud_cover'] = cloud_cover
            
            if cloud_cover > self.thresholds['satellite']['max_cloud_cover']:
                messages.append(
                    f"Excessive cloud cover: {cloud_cover:.1f}% > "
                    f"{self.thresholds['satellite']['max_cloud_cover']}%"
                )
            
            # Validate spatial metadata
            if not self._validate_spatial_metadata(data['metadata']):
                messages.append("Invalid or missing spatial metadata")
            
            is_valid = len(messages) == 0
            
        except Exception as e:
            messages.append(f"Validation failed: {str(e)}")
            is_valid = False
        
        return ValidationResult(is_valid, messages, details)
    
    def validate_lidar_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate LIDAR data products"""
        messages = []
        details = {}
        
        try:
            # Check point density
            density = self._calculate_point_density(data)
            details['point_density'] = density
            
            if density < self.thresholds['lidar']['min_point_density']:
                messages.append(
                    f"Insufficient point density: {density:.1f} pts/m² < "
                    f"{self.thresholds['lidar']['min_point_density']} pts/m²"
                )
            
            # Check classification coverage
            class_coverage = self._check_classification_coverage(data)
            details['classification_coverage'] = class_coverage
            
            missing_classes = set(self.thresholds['lidar']['required_classes']) - \
                            set(class_coverage.keys())
            if missing_classes:
                messages.append(f"Missing required point classifications: {missing_classes}")
            
            # Check for gaps
            max_gap = self._find_max_gap(data)
            details['max_gap'] = max_gap
            
            if max_gap > self.thresholds['lidar']['max_gap_size']:
                messages.append(
                    f"Large data gaps present: {max_gap:.1f}m > "
                    f"{self.thresholds['lidar']['max_gap_size']}m"
                )
            
            is_valid = len(messages) == 0
            
        except Exception as e:
            messages.append(f"Validation failed: {str(e)}")
            is_valid = False
        
        return ValidationResult(is_valid, messages, details)
    
    def validate_extracted_locations(self,
                                  locations: List[Dict[str, Any]]) -> ValidationResult:
        """Validate extracted location information from texts"""
        messages = []
        details = {
            'total_locations': len(locations),
            'valid_locations': 0,
            'invalid_locations': []
        }
        
        try:
            for loc in locations:
                # Check confidence
                if loc.get('confidence', 0) < self.thresholds['text']['min_confidence']:
                    details['invalid_locations'].append({
                        'text': loc['text'],
                        'reason': 'low_confidence',
                        'confidence': loc.get('confidence', 0)
                    })
                    continue
                
                # Check context
                if len(loc.get('context', '')) < self.thresholds['text']['min_context_length']:
                    details['invalid_locations'].append({
                        'text': loc['text'],
                        'reason': 'insufficient_context',
                        'context_length': len(loc.get('context', ''))
                    })
                    continue
                
                # Validate coordinates if present
                if 'coordinates' in loc and not self._validate_coordinates(loc['coordinates']):
                    details['invalid_locations'].append({
                        'text': loc['text'],
                        'reason': 'invalid_coordinates',
                        'coordinates': loc['coordinates']
                    })
                    continue
                
                details['valid_locations'] += 1
            
            if details['valid_locations'] == 0:
                messages.append("No valid locations found")
            
            if len(details['invalid_locations']) > 0:
                messages.append(
                    f"Found {len(details['invalid_locations'])} invalid locations"
                )
            
            is_valid = len(messages) == 0
            
        except Exception as e:
            messages.append(f"Validation failed: {str(e)}")
            is_valid = False
        
        return ValidationResult(is_valid, messages, details)
    
    def _calculate_data_coverage(self, data: Dict[str, np.ndarray]) -> float:
        """Calculate the proportion of valid data"""
        # Use the first band to calculate coverage
        first_band = next(iter(data.values()))
        valid_pixels = np.count_nonzero(~np.isnan(first_band))
        total_pixels = first_band.size
        return valid_pixels / total_pixels
    
    def _validate_spatial_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate spatial metadata"""
        required_fields = ['transform', 'crs']
        return all(field in metadata for field in required_fields)
      def _calculate_point_density(self, data: Dict[str, Any]) -> float:
        """Calculate LIDAR point density (points per square meter)"""
        if 'points' not in data:
            raise ValueError("No point data found")
            
        points = data['points']
        
        # Calculate bounds
        min_x, max_x = np.min(points['X']), np.max(points['X'])
        min_y, max_y = np.min(points['Y']), np.max(points['Y'])
        
        # Calculate area in square meters
        area = (max_x - min_x) * (max_y - min_y)
        
        # Calculate density
        if area > 0:
            return len(points) / area
        else:
            return 0.0
    
    def _check_classification_coverage(self, data: Dict[str, Any]) -> Dict[int, float]:
        """Check coverage of each point classification"""
        if 'points' not in data:
            raise ValueError("No point data found")
            
        points = data['points']
        if 'Classification' not in points.dtype.names:
            raise ValueError("No classification data found")
        
        # Count points in each class
        unique, counts = np.unique(points['Classification'], return_counts=True)
        total_points = len(points)
        
        # Calculate coverage percentages
        coverage = {}
        for class_id, count in zip(unique, counts):
            coverage[int(class_id)] = count / total_points * 100.0
        
        return coverage
    
    def _find_max_gap(self, data: Dict[str, Any]) -> float:
        """Find the largest data gap in the point cloud using KD-tree"""
        if 'points' not in data:
            raise ValueError("No point data found")
            
        points = data['points']
        coords = np.vstack((points['X'], points['Y'])).T
        
        # Build KD-tree
        tree = scipy.spatial.cKDTree(coords)
        
        # For each point, find distance to nearest neighbor
        distances, _ = tree.query(coords, k=2)  # k=2 to get nearest neighbor (first point is self)
        nearest_neighbor_distances = distances[:, 1]  # Take second column (nearest neighbor)
        
        # Return maximum gap
        return np.max(nearest_neighbor_distances)
    
    def _validate_coordinates(self, coords: Dict[str, float]) -> bool:
        """Validate geographic coordinates"""
        if 'latitude' not in coords or 'longitude' not in coords:
            return False
        
        lat = coords['latitude']
        lon = coords['longitude']
        
        return -90 <= lat <= 90 and -180 <= lon <= 180
