"""
LIDAR data processing and fusion utilities
"""

import numpy as np
import laspy
import rasterio
from rasterio.transform import from_origin
import rasterio.features
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pdal
import json
from scipy import ndimage
from scipy.interpolate import griddata
import logging
from skimage import feature, filters, morphology, draw, measure
from sklearn.decomposition import PCA
import numpy.ma as ma
import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon
from .lidar_advanced import AdvancedLidarProcessor, LidarFeature

class LidarProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger('xingu_pipeline.lidar')
        self.config = config
        self.resolution = config.get('resolution', 1.0)  # meters per pixel
        self.advanced_processor = AdvancedLidarProcessor(config)  # Initialize advanced processor
    
    def process_lidar_data(self, 
                          lidar_files: List[Path],
                          bounds: Dict[str, float]) -> Dict[str, Any]:
        """Process LIDAR point clouds and generate derived products"""
        try:
            # Merge and filter point clouds
            merged_points = self._merge_point_clouds(lidar_files)
            filtered_points = self._filter_points(merged_points)
            
            # Extract advanced features
            advanced_features = self.advanced_processor.extract_advanced_features(filtered_points)
            
            # Generate basic products
            products = {
                'dem': self._generate_dem(filtered_points, bounds),
                'dsm': self._generate_dsm(filtered_points, bounds),
                'intensity': self._generate_intensity_map(filtered_points, bounds),
                'metrics': self._calculate_point_metrics(filtered_points),
                'advanced_features': advanced_features
            }
            
            # Generate archaeological feature indicators
            archaeological_products = self._detect_archaeological_features(products)
            products.update(archaeological_products)
            
            # Combine automated detections with advanced feature analysis
            products['archaeological_features'] = self._combine_feature_detections(
                archaeological_products,
                advanced_features
            )
            
            return products
            
        except Exception as e:
            self.logger.error(f"LIDAR processing failed: {str(e)}")
            raise
    
    def _combine_feature_detections(self,
                                  automated_detections: Dict[str, Any],
                                  advanced_features: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Combine automated and advanced feature detections"""
        combined_features = []
        
        # Process automated detections
        for feature_type, feature_mask in automated_detections.items():
            if feature_type in ['geometric_patterns', 'linear_features', 'mounds']:
                # Convert binary masks to polygons
                for geom in self._mask_to_polygons(feature_mask):
                    combined_features.append({
                        'geometry': geom,
                        'properties': {
                            'type': feature_type,
                            'detection_method': 'automated',
                            'confidence': 0.7  # Base confidence for automated detections
                        }
                    })
        
        # Process advanced features
        for feature_type, features in advanced_features.items():
            if feature_type == 'archaeological':
                for feature in features:
                    combined_features.append({
                        'geometry': mapping(feature.geometry),
                        'properties': {
                            'type': feature.type,
                            'detection_method': 'advanced_analysis',
                            'confidence': feature.confidence,
                            **feature.attributes
                        }
                    })
        
        # Create GeoDataFrame of all features
        if combined_features:
            gdf = gpd.GeoDataFrame.from_features(combined_features)
            
            # Merge overlapping features of the same type
            merged_features = {}
            for feature_type in gdf['type'].unique():
                type_features = gdf[gdf['type'] == feature_type]
                dissolved = type_features.dissolve(by='type')
                merged_features[feature_type] = dissolved
            
            return {
                'features': gdf,
                'merged_features': merged_features,
                'confidence_map': self._generate_confidence_map(gdf, self.resolution)
            }
        else:
            return {
                'features': gpd.GeoDataFrame(),
                'merged_features': {},
                'confidence_map': np.zeros((100, 100))  # Empty confidence map
            }
    
    def _mask_to_polygons(self, mask: np.ndarray) -> List[Any]:
        """Convert binary mask to list of polygons"""
        polygons = []
        
        # Find contours in the mask
        contours = measure.find_contours(mask, 0.5)
        
        for contour in contours:
            # Create polygon from contour
            if len(contour) > 3:  # Need at least 4 points for a valid polygon
                poly = Polygon(contour)
                if poly.is_valid and poly.area > 0:
                    polygons.append(poly)
        
        return polygons
    
    def _generate_confidence_map(self, features: gpd.GeoDataFrame, resolution: float) -> np.ndarray:
        """Generate confidence map from detected features"""
        bounds = features.total_bounds
        width = int((bounds[2] - bounds[0]) / resolution)
        height = int((bounds[3] - bounds[1]) / resolution)
        
        confidence_map = np.zeros((height, width))
        
        for _, feature in features.iterrows():
            # Create feature mask
            feature_mask = rasterio.features.rasterize(
                [(feature.geometry, 1)],
                out_shape=(height, width),
                transform=from_origin(bounds[0], bounds[3], resolution, resolution)
            )
            
            # Add confidence values
            confidence_map += feature_mask * feature['confidence']
        
        # Normalize confidence map
        if np.any(confidence_map > 0):
            confidence_map /= np.max(confidence_map)
        
        return confidence_map

    # Other methods remain unchanged
    def _merge_point_clouds(self, lidar_files: List[Path]) -> np.ndarray:
        """Merge multiple LAS/LAZ files into a single point cloud"""
        pipeline = {
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": str(lidar_files[0])  # Start with first file
                }
            ]
        }
        
        # Add remaining files
        for file in lidar_files[1:]:
            pipeline["pipeline"].append({
                "type": "readers.las",
                "filename": str(file)
            })
        
        # Add merge step
        pipeline["pipeline"].append({
            "type": "filters.merge"
        })
        
        # Execute pipeline
        pipeline = pdal.Pipeline(json.dumps(pipeline))
        pipeline.execute()
        
        return pipeline.arrays[0]
    
    def _filter_points(self, points: np.ndarray) -> np.ndarray:
        """Filter and clean point cloud data"""
        pipeline = {
            "pipeline": [
                {
                    "type": "filters.outlier",
                    "method": "statistical",
                    "mean_k": 8,
                    "multiplier": 2.0
                },
                {
                    "type": "filters.range",
                    "limits": "Classification![7:7]"  # Remove noise points
                },
                {
                    "type": "filters.smrf"  # Ground filtering
                }
            ]
        }
        
        pipeline = pdal.Pipeline(json.dumps(pipeline), arrays=[points])
        pipeline.execute()
        
        return pipeline.arrays[0]
    
    def _generate_dem(self, 
                     points: np.ndarray,
                     bounds: Dict[str, float]) -> Dict[str, Any]:
        """Generate Digital Elevation Model from ground points"""
        # Extract ground points
        ground_mask = points['Classification'] == 2
        ground_points = points[ground_mask]
        
        # Create regular grid
        x_range = np.arange(bounds['min_lon'], bounds['max_lon'], self.resolution)
        y_range = np.arange(bounds['min_lat'], bounds['max_lat'], self.resolution)
        xi, yi = np.meshgrid(x_range, y_range)
        
        # Interpolate ground points
        zi = griddata(
            (ground_points['X'], ground_points['Y']),
            ground_points['Z'],
            (xi, yi),
            method='cubic'
        )
        
        # Create metadata
        transform = from_origin(
            bounds['min_lon'],
            bounds['max_lat'],
            self.resolution,
            self.resolution
        )
        
        return {
            'data': zi,
            'transform': transform,
            'resolution': self.resolution
        }
    
    def _generate_dsm(self,
                     points: np.ndarray,
                     bounds: Dict[str, float]) -> Dict[str, Any]:
        """Generate Digital Surface Model using highest points"""
        # Create regular grid
        x_range = np.arange(bounds['min_lon'], bounds['max_lon'], self.resolution)
        y_range = np.arange(bounds['min_lat'], bounds['max_lat'], self.resolution)
        xi, yi = np.meshgrid(x_range, y_range)
        
        # Get highest points
        zi = griddata(
            (points['X'], points['Y']),
            points['Z'],
            (xi, yi),
            method='cubic'
        )
        
        transform = from_origin(
            bounds['min_lon'],
            bounds['max_lat'],
            self.resolution,
            self.resolution
        )
        
        return {
            'data': zi,
            'transform': transform,
            'resolution': self.resolution
        }
    
    def _generate_intensity_map(self,
                              points: np.ndarray,
                              bounds: Dict[str, float]) -> Dict[str, Any]:
        """Generate intensity map from LIDAR returns"""
        x_range = np.arange(bounds['min_lon'], bounds['max_lon'], self.resolution)
        y_range = np.arange(bounds['min_lat'], bounds['max_lat'], self.resolution)
        xi, yi = np.meshgrid(x_range, y_range)
        
        # Interpolate intensity values
        intensity = griddata(
            (points['X'], points['Y']),
            points['Intensity'],
            (xi, yi),
            method='linear'
        )
        
        transform = from_origin(
            bounds['min_lon'],
            bounds['max_lat'],
            self.resolution,
            self.resolution
        )
        
        return {
            'data': intensity,
            'transform': transform,
            'resolution': self.resolution
        }
    
    def _calculate_point_metrics(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate various point cloud metrics useful for archaeology"""
        metrics = {}
        
        # Point density
        metrics['density'] = self._calculate_point_density(points)
        
        # Canopy height model
        metrics['chm'] = self._calculate_canopy_height(points)
        
        # Local relief model
        metrics['local_relief'] = self._calculate_local_relief(points)
        
        return metrics
    
    def _calculate_point_density(self, points: np.ndarray) -> np.ndarray:
        """Calculate point density per unit area"""
        # Implementation depends on specific requirements
        pass
    
    def _calculate_canopy_height(self, points: np.ndarray) -> np.ndarray:
        """Calculate canopy height model"""
        # Implementation depends on specific requirements
        pass
    
    def _calculate_local_relief(self, points: np.ndarray) -> np.ndarray:
        """Calculate local relief model"""
        # Implementation depends on specific requirements
        pass
    
    def _detect_archaeological_features(self, products: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential archaeological features from LIDAR derivatives"""
        features = {}
        
        # Detect geometric patterns (e.g., geoglyphs)
        features['geometric_patterns'] = self._detect_geometric_patterns(
            products['dem']['data'],
            products['intensity']['data']
        )
        
        # Detect linear features (e.g., ancient roads)
        features['linear_features'] = self._detect_linear_features(
            products['local_relief']
        )
        
        # Detect mounds
        features['mounds'] = self._detect_mounds(
            products['dem']['data']
        )
        
        return features
    def _detect_geometric_patterns(self,
                                 dem: np.ndarray,
                                 intensity: np.ndarray) -> np.ndarray:
        """Detect geometric patterns indicative of archaeological features"""
        from skimage import feature, filters, morphology
        from scipy import ndimage
        
        # Create binary mask of potential features
        # 1. Apply Sobel edge detection
        edges = filters.sobel(dem)
        
        # 2. Use intensity data to refine edges
        intensity_edges = filters.sobel(intensity)
        combined_edges = edges * (intensity_edges > np.percentile(intensity_edges, 75))
        
        # 3. Threshold to create binary mask
        thresh = filters.threshold_otsu(combined_edges)
        binary = combined_edges > thresh
        
        # 4. Clean up mask
        binary = morphology.remove_small_objects(binary, min_size=100)
        binary = morphology.remove_small_holes(binary, area_threshold=50)
        
        # 5. Detect geometric shapes using Hough transform
        lines = feature.probabilistic_hough_line(binary)
        
        # Create output mask for geometric patterns
        geometric_mask = np.zeros_like(dem, dtype=bool)
        
        # Add detected lines to mask
        for line in lines:
            rr, cc = line
            geometric_mask[rr, cc] = True
        
        # Look for circular patterns
        circles = feature.blob_log(binary, min_sigma=5, max_sigma=30)
        
        for circle in circles:
            y, x, r = circle
            rr, cc = draw.circle(y, x, r)
            mask = (rr >= 0) & (rr < dem.shape[0]) & (cc >= 0) & (cc < dem.shape[1])
            geometric_mask[rr[mask], cc[mask]] = True
        
        return geometric_mask
    
    def _detect_linear_features(self, relief: np.ndarray) -> np.ndarray:
        """Detect linear features like ancient roads or canals"""
        from skimage import feature, filters, morphology
        
        # 1. Enhance linear features using directional filters
        angles = np.arange(0, 180, 45)
        responses = []
        
        for angle in angles:
            # Create directional filter
            kernel = feature.haar_like_feature_detector(
                relief.shape,
                feature_type='line',
                feature_width=5,
                feature_height=15,
                theta=np.radians(angle)
            )
            
            # Apply filter
            response = ndimage.convolve(relief, kernel)
            responses.append(response)
        
        # 2. Combine responses
        combined = np.maximum.reduce(responses)
        
        # 3. Threshold to create binary mask
        thresh = filters.threshold_otsu(combined)
        binary = combined > thresh
        
        # 4. Clean up and connect features
        binary = morphology.remove_small_objects(binary, min_size=50)
        binary = morphology.skeletonize(binary)
        
        # 5. Filter by length and linearity
        labeled, num_features = ndimage.label(binary)
        linear_mask = np.zeros_like(binary)
        
        for i in range(1, num_features + 1):
            feature_mask = labeled == i
            coords = np.column_stack(np.where(feature_mask))
            
            if len(coords) < 20:  # Skip short features
                continue
                
            # Calculate linearity using PCA
            pca = PCA(n_components=2)
            pca.fit(coords)
            linearity = pca.explained_variance_ratio_[0]
            
            if linearity > 0.8:  # Keep highly linear features
                linear_mask[feature_mask] = True
        
        return linear_mask
    
    def _detect_mounds(self, dem: np.ndarray) -> np.ndarray:
        """Detect potential archaeological mounds"""
        from scipy import ndimage
        from skimage import filters, morphology
        
        # 1. Calculate local relief
        neighborhood_size = 21
        local_mean = ndimage.uniform_filter(dem, neighborhood_size)
        local_relief = dem - local_mean
        
        # 2. Identify potential mounds based on height thresholds
        min_height = 0.5  # meters
        max_height = 5.0  # meters
        mound_candidates = (local_relief >= min_height) & (local_relief <= max_height)
        
        # 3. Apply shape constraints
        labeled, num_features = ndimage.label(mound_candidates)
        mound_mask = np.zeros_like(mound_candidates)
        
        for i in range(1, num_features + 1):
            feature_mask = labeled == i
            
            # Get feature properties
            props = measure.regionprops(feature_mask.astype(int))[0]
            
            # Filter based on shape characteristics
            if (props.area >= 100 and  # Minimum area in pixels
                props.area <= 10000 and  # Maximum area
                props.eccentricity < 0.8 and  # Roughly circular
                props.solidity > 0.7):  # Solid shape
                
                mound_mask[feature_mask] = True
        
        # 4. Clean up results
        mound_mask = morphology.remove_small_objects(mound_mask, min_size=50)
        mound_mask = morphology.remove_small_holes(mound_mask, area_threshold=20)
        
        return mound_mask
