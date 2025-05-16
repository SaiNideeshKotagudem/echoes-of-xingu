"""
Geospatial integration and analysis utilities
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from typing import Dict, List, Tuple, Union, Optional, Any
import rasterio
from rasterio.features import geometry_mask
from scipy.spatial import cKDTree
import networkx as nx
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

class GeospatialIntegrator:
    def __init__(self):
        self.crs = 'EPSG:4326'  # WGS84
        self.default_clustering_params = {
            'eps': 0.001,  # Approximately 100m at equator
            'min_samples': 3,
            'metric': 'haversine'
        }
    
    def create_site_predictions(self, 
                              model_output: np.ndarray,
                              bounds: Dict[str, float],
                              probability_threshold: float = 0.7,
                              cluster_predictions: bool = True) -> gpd.GeoDataFrame:
        """Convert model predictions to GeoDataFrame of potential sites with optional clustering"""
        # Get coordinates where probability exceeds threshold
        y_coords, x_coords = np.where(model_output > probability_threshold)
        probs = model_output[y_coords, x_coords]
        
        # Convert pixel coordinates to geographic coordinates
        lon = np.linspace(bounds['min_lon'], bounds['max_lon'], model_output.shape[1])
        lat = np.linspace(bounds['min_lat'], bounds['max_lat'], model_output.shape[0])
        
        # Create points
        points = [Point(lon[x], lat[y]) for y, x in zip(y_coords, x_coords)]
        
        # Create initial GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'geometry': points,
            'probability': probs,
            'cluster_id': -1
        }, crs=self.crs)
        
        if cluster_predictions:
            gdf = self._cluster_predictions(gdf)
        
        return gdf
    
    def _cluster_predictions(self, 
                           gdf: gpd.GeoDataFrame,
                           params: Optional[Dict[str, Any]] = None) -> gpd.GeoDataFrame:
        """Cluster nearby predictions using DBSCAN"""
        # Use default or provided clustering parameters
        cluster_params = self.default_clustering_params.copy()
        if params:
            cluster_params.update(params)
        
        # Extract coordinates for clustering
        coords = np.radians(gdf.geometry.apply(lambda p: [p.x, p.y]).tolist())
        
        # Perform clustering
        clustering = DBSCAN(
            eps=cluster_params['eps'],
            min_samples=cluster_params['min_samples'],
            metric=cluster_params['metric']
        ).fit(coords)
        
        # Add cluster labels to GeoDataFrame
        gdf['cluster_id'] = clustering.labels_
        
        # Calculate cluster centroids and confidences
        clustered_sites = []
        unique_clusters = set(clustering.labels_)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_points = gdf[gdf['cluster_id'] == cluster_id]
            centroid = cluster_points.geometry.unary_union.centroid
            mean_prob = cluster_points['probability'].mean()
            point_count = len(cluster_points)
            
            clustered_sites.append({
                'geometry': centroid,
                'probability': mean_prob,
                'point_count': point_count,
                'cluster_id': cluster_id
            })
        
        # Create new GeoDataFrame with clustered results
        if clustered_sites:
            clustered_gdf = gpd.GeoDataFrame(clustered_sites, crs=self.crs)
            return clustered_gdf
        else:
            return gdf

    def integrate_hydrology(self, 
                          current_rivers: gpd.GeoDataFrame,
                          historical_rivers: Dict[str, gpd.GeoDataFrame],
                          site_predictions: gpd.GeoDataFrame,
                          max_distance: float = 2000) -> gpd.GeoDataFrame:
        """Integrate hydrological data with site predictions"""
        # Create network from current rivers
        G = nx.from_geodataframes(current_rivers, current_rivers)
        
        # Function to find nearest river point
        def nearest_river_point(point: Point, river_gdf: gpd.GeoDataFrame) -> Tuple[float, Point]:
            distances = river_gdf.geometry.distance(point)
            idx = distances.argmin()
            return distances[idx], river_gdf.geometry.iloc[idx]
        
        # Add hydrological attributes to predictions
        site_predictions['current_river_dist'], _ = zip(*[
            nearest_river_point(point, current_rivers)
            for point in site_predictions.geometry
        ])
        
        # Add historical river distances
        for period, river_gdf in historical_rivers.items():
            site_predictions[f'river_dist_{period}'], _ = zip(*[
                nearest_river_point(point, river_gdf)
                for point in site_predictions.geometry
            ])
        
        # Filter by maximum distance criterion
        site_predictions = site_predictions[
            site_predictions['current_river_dist'] <= max_distance
        ]
        
        return site_predictions
    
    def create_linguistic_diffusion_map(self,
                                      linguistic_data: Dict[str, Dict],
                                      site_locations: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Create a linguistic diffusion map"""
        points = []
        attributes = []
        
        # Process linguistic data
        for language, data in linguistic_data.items():
            for location in data['locations']:
                point = Point(location['longitude'], location['latitude'])
                attributes.append({
                    'language': language,
                    'family': data.get('family', 'Unknown'),
                    'period': data.get('period', 'Unknown'),
                    'vocabulary_size': len(data.get('vocabulary', [])),
                })
                points.append(point)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            attributes,
            geometry=points,
            crs=self.crs
        )
        
        # Calculate linguistic density
        gdf['density'] = self._calculate_linguistic_density(gdf)
        
        return gdf
    
    def validate_ring_village_pattern(self,
                                    site_predictions: gpd.GeoDataFrame,
                                    radius_range: Tuple[float, float] = (100, 300),
                                    min_points: int = 5,
                                    max_eccentricity: float = 0.3) -> gpd.GeoDataFrame:
        """Validate predictions based on known ring-village morphology"""
        def calculate_eccentricity(points: np.ndarray) -> float:
            """Calculate the eccentricity of a set of points"""
            if len(points) < 3:
                return 1.0
                
            try:
                # Create convex hull
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                
                # Fit an ellipse to the hull points
                center = np.mean(hull_points, axis=0)
                covariance = np.cov(hull_points.T)
                eigenvals, _ = np.linalg.eig(covariance)
                
                # Calculate eccentricity
                a = np.sqrt(np.max(eigenvals))
                b = np.sqrt(np.min(eigenvals))
                return np.sqrt(1 - (b * b) / (a * a))
            except:
                return 1.0
        
        validated_sites = []
        
        # Create KD-tree for efficient nearest neighbor search
        coords = np.vstack((
            site_predictions.geometry.x,
            site_predictions.geometry.y
        )).T
        tree = cKDTree(coords)
        
        for idx, row in site_predictions.iterrows():
            point = coords[idx]
            
            # Find neighbors within maximum radius
            neighbors_idx = tree.query_ball_point(point, radius_range[1])
            neighbors = coords[neighbors_idx]
            
            if len(neighbors) >= min_points:
                # Calculate point pattern statistics
                distances = np.linalg.norm(neighbors - point, axis=1)
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                eccentricity = calculate_eccentricity(neighbors)
                
                # Validate based on multiple criteria
                if (radius_range[0] <= mean_dist <= radius_range[1] and  # Size criterion
                    std_dist / mean_dist < 0.3 and                        # Regularity criterion
                    eccentricity < max_eccentricity):                     # Shape criterion
                    
                    # Create a polygon representing the village boundary
                    hull = ConvexHull(neighbors)
                    boundary = Polygon(neighbors[hull.vertices])
                    
                    validated_sites.append({
                        'geometry': row.geometry,
                        'probability': row.probability,
                        'mean_radius': mean_dist,
                        'eccentricity': eccentricity,
                        'point_count': len(neighbors),
                        'village_boundary': boundary
                    })
        
        if validated_sites:
            return gpd.GeoDataFrame(validated_sites, crs=self.crs)
        else:
            return gpd.GeoDataFrame([], columns=site_predictions.columns, crs=self.crs)
    
    def _calculate_linguistic_density(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        """Calculate linguistic density using kernel density estimation"""
        from scipy.stats import gaussian_kde
        
        # Get coordinates
        coords = np.vstack((gdf.geometry.x, gdf.geometry.y))
        
        # Calculate kernel density
        kde = gaussian_kde(coords)
        density = kde(coords)
        
        return density
