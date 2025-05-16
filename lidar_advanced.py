"""
Advanced LIDAR feature extraction utilities
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pdal
import json
from scipy import ndimage
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import logging
from dataclasses import dataclass

@dataclass
class LidarFeature:
    type: str  # 'vegetation', 'structure', 'terrain'
    geometry: Any
    attributes: Dict[str, Any]
    confidence: float

class AdvancedLidarProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger('xingu_pipeline.lidar_advanced')
        self.config = config
        
        # Feature detection parameters
        self.params = {
            'vegetation': {
                'min_height': 2.0,
                'density_threshold': 0.5
            },
            'structures': {
                'min_size': 10.0,
                'max_size': 1000.0,
                'height_threshold': 1.5
            },
            'terrain': {
                'slope_threshold': 15.0,
                'relief_window': 21
            }
        }
    
    def extract_advanced_features(self, points: np.ndarray) -> Dict[str, List[LidarFeature]]:
        """Extract advanced features from LIDAR point cloud"""
        features = {
            'vegetation': [],
            'structures': [],
            'terrain': [],
            'archaeological': []
        }
        
        try:
            # Remove noise and classify points
            cleaned_points = self._denoise_points(points)
            classified_points = self._classify_points(cleaned_points)
            
            # Extract vegetation features
            features['vegetation'] = self._extract_vegetation(classified_points)
            
            # Extract potential structures
            features['structures'] = self._extract_structures(classified_points)
            
            # Extract terrain features
            features['terrain'] = self._extract_terrain_features(classified_points)
            
            # Identify potential archaeological features
            features['archaeological'] = self._identify_archaeological_features(
                classified_points,
                features['terrain']
            )
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            raise
        
        return features
    
    def _denoise_points(self, points: np.ndarray) -> np.ndarray:
        """Remove noise from point cloud using statistical outlier removal"""
        pipeline = {
            "pipeline": [
                {
                    "type": "filters.statisticaloutlier",
                    "mean_k": 8,
                    "multiplier": 2.0
                },
                {
                    "type": "filters.elmstatic",
                    "cell": 1.0,
                    "class": 7
                }
            ]
        }
        
        pipeline = pdal.Pipeline(json.dumps(pipeline))
        pipeline.execute()
        
        return pipeline.arrays[0]
    
    def _classify_points(self, points: np.ndarray) -> np.ndarray:
        """Classify points using machine learning approach"""
        # Extract point features
        features = self._compute_point_features(points)
        
        # Apply classification
        classified = points.copy()
        
        # Use geometric features for classification
        height_above_ground = points['Z'] - np.min(points['Z'])
        intensity_normalized = (points['Intensity'] - np.mean(points['Intensity'])) / np.std(points['Intensity'])
        
        # Classify vegetation
        vegetation_mask = (height_above_ground > self.params['vegetation']['min_height']) & \
                        (intensity_normalized < 0)
        classified['Classification'][vegetation_mask] = 5  # High vegetation
        
        # Classify potential structures
        structure_mask = (height_above_ground > self.params['structures']['height_threshold']) & \
                        (intensity_normalized > 0)
        classified['Classification'][structure_mask] = 6  # Building
        
        return classified
    
    def _compute_point_features(self, points: np.ndarray) -> np.ndarray:
        """Compute geometric features for each point"""
        # Calculate local point density
        density = self._calculate_local_density(points)
        
        # Calculate surface normals
        normals = self._calculate_surface_normals(points)
        
        # Calculate eigenvalue features
        eigenfeatures = self._calculate_eigenfeatures(points)
        
        return np.column_stack([density, normals, eigenfeatures])
    
    def _calculate_local_density(self, points: np.ndarray) -> np.ndarray:
        """Calculate local point density"""
        from scipy.spatial import cKDTree
        
        tree = cKDTree(points[['X', 'Y', 'Z']])
        density = np.zeros(len(points))
        
        # Count points within radius
        radius = 1.0  # 1 meter radius
        for i in range(len(points)):
            density[i] = len(tree.query_ball_point(points[i], radius))
        
        return density
    
    def _calculate_surface_normals(self, points: np.ndarray) -> np.ndarray:
        """Calculate surface normals using PCA"""
        from scipy.spatial import cKDTree
        
        tree = cKDTree(points[['X', 'Y', 'Z']])
        normals = np.zeros((len(points), 3))
        
        # Calculate normals for each point
        k = 10  # Number of neighbors
        for i in range(len(points)):
            distances, indices = tree.query(points[i], k=k)
            neighbors = points[indices]
            
            # Perform PCA
            pca = PCA(n_components=3)
            pca.fit(neighbors[['X', 'Y', 'Z']])
            
            # Use smallest eigenvector as normal
            normals[i] = pca.components_[-1]
        
        return normals
    
    def _calculate_eigenfeatures(self, points: np.ndarray) -> np.ndarray:
        """Calculate eigenvalue-based features"""
        from scipy.spatial import cKDTree
        
        tree = cKDTree(points[['X', 'Y', 'Z']])
        features = np.zeros((len(points), 3))
        
        k = 10  # Number of neighbors
        for i in range(len(points)):
            distances, indices = tree.query(points[i], k=k)
            neighbors = points[indices]
            
            # Perform PCA
            pca = PCA(n_components=3)
            pca.fit(neighbors[['X', 'Y', 'Z']])
            
            # Calculate eigenvalue features
            eigenvalues = pca.explained_variance_
            features[i] = [
                eigenvalues[0] / sum(eigenvalues),  # Linearity
                eigenvalues[1] / sum(eigenvalues),  # Planarity
                eigenvalues[2] / sum(eigenvalues)   # Sphericity
            ]
        
        return features
    
    def _extract_vegetation(self, points: np.ndarray) -> List[LidarFeature]:
        """Extract vegetation features from classified point cloud"""
        vegetation_features = []
        
        # Get points classified as vegetation
        veg_mask = points['Classification'] == 5  # Standard LIDAR class for high vegetation
        vegetation_points = points[veg_mask]
        
        if len(vegetation_points) == 0:
            return vegetation_features
            
        # Cluster vegetation points into distinct features
        clustering = DBSCAN(
            eps=self.params['vegetation']['min_height'],
            min_samples=5
        ).fit(vegetation_points[['X', 'Y', 'Z']])
        
        unique_labels = np.unique(clustering.labels_)
        
        # Process each vegetation cluster
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
                
            cluster_points = vegetation_points[clustering.labels_ == label]
            
            # Calculate cluster properties
            min_z = np.min(cluster_points['Z'])
            max_z = np.max(cluster_points['Z'])
            height = max_z - min_z
            
            # Create convex hull of points for geometry
            points_2d = [(p['X'], p['Y']) for p in cluster_points]
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(points_2d)
                hull_points = [points_2d[i] for i in hull.vertices]
                from shapely.geometry import Polygon
                geometry = Polygon(hull_points)
                
                # Calculate density and other metrics
                density = len(cluster_points) / geometry.area
                if density >= self.params['vegetation']['density_threshold']:
                    vegetation_features.append(LidarFeature(
                        type='vegetation',
                        geometry=geometry,
                        attributes={
                            'height': height,
                            'density': density,
                            'point_count': len(cluster_points),
                            'mean_intensity': np.mean(cluster_points['Intensity'])
                        },
                        confidence=min(0.9, density / 2)  # Scale confidence with density
                    ))
            except Exception as e:
                self.logger.warning(f"Failed to process vegetation cluster: {str(e)}")
                continue
                
        return vegetation_features
    
    def _extract_structures(self, points: np.ndarray) -> List[LidarFeature]:
        """Extract potential structural features from classified point cloud"""
        structure_features = []
        
        # Get points likely to be structures (excluding vegetation and ground)
        structure_mask = np.isin(points['Classification'], [6, 26])  # Building and archaeological feature classes
        structure_points = points[structure_mask]
        
        if len(structure_points) == 0:
            return structure_features
            
        # Use DBSCAN to cluster points into potential structures
        clustering = DBSCAN(
            eps=2.0,  # 2 meter radius
            min_samples=10
        ).fit(structure_points[['X', 'Y', 'Z']])
        
        unique_labels = np.unique(clustering.labels_)
        
        for label in unique_labels:
            if label == -1:
                continue
                
            cluster_points = structure_points[clustering.labels_ == label]
            
            try:
                # Create 2D footprint
                points_2d = [(p['X'], p['Y']) for p in cluster_points]
                hull = ConvexHull(points_2d)
                hull_points = [points_2d[i] for i in hull.vertices]
                geometry = Polygon(hull_points)
                
                # Calculate structure metrics
                area = geometry.area
                if (area >= self.params['structures']['min_size'] and 
                    area <= self.params['structures']['max_size']):
                    
                    # Calculate height and other properties
                    min_z = np.min(cluster_points['Z'])
                    max_z = np.max(cluster_points['Z'])
                    height = max_z - min_z
                    
                    # Check if height suggests artificial structure
                    if height >= self.params['structures']['height_threshold']:
                        # Calculate geometric properties
                        bbox = geometry.minimum_rotated_rectangle
                        length = bbox.length
                        width = bbox.width
                        elongation = length / width if width > 0 else 1
                        
                        # Calculate confidence based on geometry and point density
                        point_density = len(cluster_points) / area
                        geometry_score = min(1.0, 4.0 / elongation) # Prefer less elongated shapes
                        confidence = min(0.95, (point_density * geometry_score * 0.7 + 0.3))
                        
                        structure_features.append(LidarFeature(
                            type='structure',
                            geometry=geometry,
                            attributes={
                                'area': area,
                                'height': height,
                                'elongation': elongation,
                                'point_count': len(cluster_points),
                                'mean_intensity': np.mean(cluster_points['Intensity']),
                                'min_z': min_z,
                                'max_z': max_z
                            },
                            confidence=confidence
                        ))
                        
            except Exception as e:
                self.logger.warning(f"Failed to process structure cluster: {str(e)}")
                continue
                
        return structure_features
    
    def _extract_terrain_features(self, points: np.ndarray) -> List[LidarFeature]:
        """Extract significant terrain features from ground-classified points"""
        terrain_features = []
        
        # Filter for ground points
        ground_mask = points['Classification'] == 2  # Standard LIDAR ground class
        ground_points = points[ground_mask]
        
        if len(ground_points) == 0:
            return terrain_features
            
        # Create regular grid for DEM generation
        x_min, x_max = np.min(ground_points['X']), np.max(ground_points['X'])
        y_min, y_max = np.min(ground_points['Y']), np.max(ground_points['Y'])
        
        # Grid parameters
        grid_size = 1.0  # 1 meter resolution
        x_grid = np.arange(x_min, x_max + grid_size, grid_size)
        y_grid = np.arange(y_min, y_max + grid_size, grid_size)
        
        # Create DEM using natural neighbor interpolation
        from scipy.interpolate import griddata
        xi, yi = np.meshgrid(x_grid, y_grid)
        points_2d = np.column_stack((ground_points['X'], ground_points['Y']))
        zi = griddata(points_2d, ground_points['Z'], (xi, yi), method='cubic')
        
        # Calculate terrain derivatives
        dx, dy = np.gradient(zi, grid_size)
        slope = np.sqrt(dx**2 + dy**2)
        
        # Calculate terrain roughness (standard deviation of elevation in window)
        from scipy.ndimage import uniform_filter
        window_size = self.params['terrain']['relief_window']
        elevation_mean = uniform_filter(zi, size=window_size)
        elevation_std = np.sqrt(uniform_filter((zi - elevation_mean)**2, size=window_size))
        
        # Identify significant terrain features
        slope_threshold = self.params['terrain']['slope_threshold']
        significant_terrain = np.zeros_like(zi, dtype=bool)
        
        # Look for areas with high slope or unusual roughness
        significant_terrain |= (slope > np.deg2rad(slope_threshold))
        significant_terrain |= (elevation_std > np.mean(elevation_std) + 2*np.std(elevation_std))
        
        # Label connected components
        from scipy.ndimage import label
        labeled_features, num_features = label(significant_terrain)
        
        for feature_id in range(1, num_features + 1):
            try:
                # Get feature mask
                feature_mask = labeled_features == feature_id
                feature_bounds = np.where(feature_mask)
                
                # Convert grid indices to coordinates
                y_coords = y_grid[feature_bounds[0]]
                x_coords = x_grid[feature_bounds[1]]
                z_coords = zi[feature_bounds]
                
                # Create polygon from concave hull
                from shapely.ops import unary_union
                from shapely.geometry import Point, MultiPoint
                points = MultiPoint([(x, y) for x, y in zip(x_coords, y_coords)])
                
                # Use buffer operations to create concave hull
                geometry = points.buffer(grid_size*2).buffer(-grid_size)
                
                if geometry.is_empty or not geometry.is_valid:
                    continue
                    
                # Calculate feature properties
                area = geometry.area
                relief = np.ptp(z_coords)
                mean_slope = np.mean(slope[feature_mask])
                
                # Calculate confidence based on feature characteristics
                size_score = min(1.0, area / 100)  # Larger features more likely to be significant
                relief_score = min(1.0, relief / 2)  # Features with more relief more likely to be significant
                confidence = np.mean([size_score, relief_score])
                
                terrain_features.append(LidarFeature(
                    type='terrain',
                    geometry=geometry,
                    attributes={
                        'area': area,
                        'relief': relief,
                        'mean_slope': mean_slope,
                        'max_slope': np.max(slope[feature_mask]),
                        'roughness': np.mean(elevation_std[feature_mask]),
                        'mean_elevation': np.mean(z_coords)
                    },
                    confidence=confidence
                ))
                
            except Exception as e:
                self.logger.warning(f"Failed to process terrain feature: {str(e)}")
                continue
                
        return terrain_features
    
    def _identify_archaeological_features(self,
                                       points: np.ndarray,
                                       terrain_features: List[LidarFeature]) -> List[LidarFeature]:
        """Identify potential archaeological features by combining multiple feature types"""
        archaeological_features = []
        
        # Process each terrain feature
        for terrain in terrain_features:
            try:
                # Skip features that are too small or large to be archaeological
                if terrain.attributes['area'] < 25 or terrain.attributes['area'] > 10000:
                    continue
                
                # Calculate feature characteristics that suggest human modification
                artificial_indicators = {
                    'geometric_regularity': self._calculate_geometric_regularity(terrain.geometry),
                    'elevation_pattern': self._analyze_elevation_pattern(
                        points, terrain.geometry
                    ),
                    'context_score': self._evaluate_spatial_context(
                        points, terrain.geometry
                    )
                }
                
                # Weight and combine indicators
                indicator_weights = {
                    'geometric_regularity': 0.4,
                    'elevation_pattern': 0.4,
                    'context_score': 0.2
                }
                
                total_score = sum(
                    score * indicator_weights[indicator]
                    for indicator, score in artificial_indicators.items()
                )
                
                if total_score > 0.7:  # Threshold for archaeological feature classification
                    # Determine the likely feature type
                    feature_type = self._classify_archaeological_feature(
                        terrain.attributes,
                        artificial_indicators
                    )
                    
                    archaeological_features.append(LidarFeature(
                        type=feature_type,
                        geometry=terrain.geometry,
                        attributes={
                            **terrain.attributes,
                            **artificial_indicators,
                            'feature_score': total_score
                        },
                        confidence=total_score
                    ))
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze potential archaeological feature: {str(e)}")
                continue
        
        return archaeological_features
        
    def _calculate_geometric_regularity(self, geometry) -> float:
        """Calculate how geometrically regular a feature is"""
        try:
            # Compare with minimum bounding circle
            from shapely.geometry import Point
            centroid = geometry.centroid
            radius = max(
                Point(centroid.x, centroid.y).distance(Point(p[0], p[1]))
                for p in geometry.exterior.coords
            )
            circle = Point(centroid.x, centroid.y).buffer(radius)
            
            # Calculate circularity/regularity metrics
            compactness = geometry.area / circle.area
            perimeter_ratio = geometry.length / (2 * np.pi * radius)
            
            # High scores for both very circular features (like mounds)
            # and rectilinear features (like walls)
            circular_score = compactness
            rectilinear_score = self._calculate_rectilinearity(geometry)
            
            return max(circular_score, rectilinear_score)
            
        except Exception:
            return 0.0
            
    def _analyze_elevation_pattern(self, points: np.ndarray, geometry) -> float:
        """Analyze elevation patterns within the feature"""
        try:
            # Get points within the geometry
            mask = np.array([
                geometry.contains(Point(p['X'], p['Y']))
                for p in points
            ])
            local_points = points[mask]
            
            if len(local_points) < 10:
                return 0.0
                
            # Analyze elevation distribution
            z_values = local_points['Z']
            z_range = np.ptp(z_values)
            z_std = np.std(z_values)
            
            # Calculate metrics that might indicate artificial modification
            relative_height = z_range / z_std if z_std > 0 else 0
            symmetry_score = self._calculate_elevation_symmetry(z_values)
            
            return np.mean([
                min(1.0, relative_height / 5),
                symmetry_score
            ])
            
        except Exception:
            return 0.0
            
    def _evaluate_spatial_context(self, points: np.ndarray, geometry) -> float:
        """Evaluate the spatial context of the feature"""
        try:
            # Create a buffer around the feature
            buffer_dist = np.sqrt(geometry.area)  # Scale with feature size
            context_area = geometry.buffer(buffer_dist)
            
            # Analyze point distribution in context area
            mask = np.array([
                context_area.contains(Point(p['X'], p['Y']))
                for p in points
            ])
            context_points = points[mask]
            
            if len(context_points) < 10:
                return 0.0
                
            # Look for patterns in point distribution
            from scipy.stats import entropy
            z_hist, _ = np.histogram(context_points['Z'], bins='auto', density=True)
            distribution_entropy = entropy(z_hist)
            
            # Calculate isolation or association with other features
            isolation_score = self._calculate_isolation(geometry, context_points)
            
            return np.mean([
                1.0 - min(1.0, distribution_entropy / 2),
                isolation_score
            ])
            
        except Exception:
            return 0.0
            
    def _calculate_rectilinearity(self, geometry) -> float:
        """Calculate how rectilinear a feature is"""
        try:
            # Get minimum rotated rectangle
            min_rect = geometry.minimum_rotated_rectangle
            
            # Calculate area ratio
            area_ratio = geometry.area / min_rect.area
            
            # Get angle differences between segments
            coords = list(geometry.exterior.coords)
            angles = []
            
            for i in range(len(coords) - 2):
                p1 = np.array(coords[i])
                p2 = np.array(coords[i + 1])
                p3 = np.array(coords[i + 2])
                
                # Calculate vectors
                v1 = p2 - p1
                v2 = p3 - p2
                
                # Calculate angle
                angle = np.abs(np.degrees(
                    np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
                ))
                angles.append(min(angle, 180 - angle))
            
            # Score based on prevalence of right angles
            right_angles = sum(1 for a in angles if abs(a - 90) < 15)
            angle_score = right_angles / len(angles) if angles else 0
            
            return np.mean([area_ratio, angle_score])
            
        except Exception:
            return 0.0
            
    def _calculate_elevation_symmetry(self, z_values: np.ndarray) -> float:
        """Calculate the symmetry of elevation distribution"""
        try:
            # Calculate histogram
            hist, bins = np.histogram(z_values, bins='auto', density=True)
            
            # Calculate symmetry around mean
            mean_idx = np.argmax(hist)
            left = hist[:mean_idx]
            right = hist[mean_idx:]
            
            # Compare left and right sides
            min_len = min(len(left), len(right))
            if min_len < 2:
                return 0.0
                
            left = left[-min_len:]
            right = right[:min_len]
            
            # Calculate correlation between left and right sides
            from scipy.stats import pearsonr
            correlation = pearsonr(left, np.flip(right))[0]
            
            return max(0.0, correlation)
            
        except Exception:
            return 0.0
            
    def _calculate_isolation(self, geometry, context_points: np.ndarray) -> float:
        """Calculate how isolated a feature is from other terrain variations"""
        try:
            # Create distance bands around feature
            from shapely.geometry import LineString
            
            distances = []
            centroid = geometry.centroid
            
            for p in context_points:
                point = Point(p['X'], p['Y'])
                if not geometry.contains(point):
                    distances.append(point.distance(centroid))
            
            if not distances:
                return 0.0
                
            # Analyze point distribution with distance
            distances = np.array(distances)
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            # Higher score for more evenly distributed points
            evenness = 1.0 - (std_dist / mean_dist if mean_dist > 0 else 1.0)
            
            return min(1.0, evenness)
            
        except Exception:
            return 0.0
            
    def _classify_archaeological_feature(self,
                                      terrain_attributes: Dict[str, Any],
                                      indicators: Dict[str, float]) -> str:
        """Classify the type of archaeological feature based on its characteristics"""
        # Extract key metrics
        area = terrain_attributes['area']
        relief = terrain_attributes['relief']
        mean_slope = terrain_attributes['mean_slope']
        geometric_score = indicators['geometric_regularity']
        
        # Classification logic
        if geometric_score > 0.8 and area < 100:
            if relief > 1.0:
                return 'mound'
            else:
                return 'foundation'
        elif geometric_score > 0.7 and area > 1000:
            if mean_slope > np.deg2rad(15):
                return 'terrace'
            else:
                return 'earthwork'
        elif geometric_score > 0.6 and relief < 0.5:
            return 'ancient_field'
        else:
            return 'archaeological_feature'  # Generic classification
