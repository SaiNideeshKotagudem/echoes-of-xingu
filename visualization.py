"""
Visualization utilities for archaeological site analysis and presentation
"""

import numpy as np
import folium
from keplergl import KeplerGl
import json
from typing import Dict, List, Union, Any
import geopandas as gpd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class ArchaeologicalVisualizer:
    def __init__(self):
        self.map_style = {
            'mapbox': {
                'style': 'mapbox://styles/mapbox/satellite-v9'
            }
        }
        self.feature_colors = {
            'vegetation': '#2ecc71',
            'structure': '#e74c3c',
            'terrain': '#3498db',
            'archaeological': '#f1c40f'
        }
    
    def create_kepler_map(self, data: Dict[str, gpd.GeoDataFrame]) -> KeplerGl:
        """Create a Kepler.gl map with archaeological data layers"""
        map_1 = KeplerGl(height=800)
        
        # Add each dataset as a layer
        for name, gdf in data.items():
            map_1.add_data(data=gdf, name=name)
        
        # Configure map settings
        config = {
            'version': 'v1',
            'config': {
                'visState': {
                    'layers': self._create_layer_configs(data),
                    'filters': []
                }
            }
        }
        
        map_1.config = config
        return map_1
    
    def create_temporal_visualization(self, 
                                   site_data: gpd.GeoDataFrame,
                                   river_data: Dict[str, gpd.GeoDataFrame],
                                   linguistic_data: gpd.GeoDataFrame) -> go.Figure:
        """Create a temporal visualization showing changes over time"""
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'mapbox'}]])
        
        # Add modern satellite basemap
        fig.add_trace(go.Scattermapbox(
            lat=site_data.geometry.y,
            lon=site_data.geometry.x,
            mode='markers',
            marker=dict(size=10),
            text=site_data['site_name'],
            name='Archaeological Sites'
        ))
        
        # Add historical river paths
        for period, river_gdf in river_data.items():
            fig.add_trace(go.Scattermapbox(
                lat=river_gdf.geometry.y,
                lon=river_gdf.geometry.x,
                mode='lines',
                line=dict(width=2),
                name=f'River Path - {period}'
            ))
        
        # Add linguistic diffusion patterns
        fig.add_trace(go.Scattermapbox(
            lat=linguistic_data.geometry.y,
            lon=linguistic_data.geometry.x,
            mode='markers+text',
            marker=dict(size=8),
            text=linguistic_data['language'],
            name='Linguistic Groups'
        ))
        
        # Update layout
        fig.update_layout(
            mapbox=dict(
                style='satellite',
                zoom=8
            ),
            showlegend=True,
            height=800
        )
        
        return fig
    
    def create_3d_terrain_view(self, 
                             dem_data: np.ndarray, 
                             site_locations: List[Dict[str, float]],
                             resolution: float = 30.0) -> go.Figure:
        """Create a 3D terrain visualization with archaeological sites"""
        # Create meshgrid for 3D surface
        y, x = np.mgrid[:dem_data.shape[0], :dem_data.shape[1]]
        
        # Create 3D surface plot
        fig = go.Figure(data=[
            go.Surface(z=dem_data, colorscale='earth'),
            go.Scatter3d(
                x=[site['x'] for site in site_locations],
                y=[site['y'] for site in site_locations],
                z=[site['elevation'] for site in site_locations],
                mode='markers',
                marker=dict(size=5, color='red'),
                name='Archaeological Sites'
            )
        ])
        
        # Update layout
        fig.update_layout(
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            title='3D Terrain View with Archaeological Sites',
            height=800
        )
        
        return fig
    
    def visualize_lidar_features(self,
                               features: Dict[str, Any],
                               dem: Dict[str, Any],
                               confidence_map: np.ndarray,
                               output_path: str = None) -> go.Figure:
        """Create an interactive 3D visualization of LIDAR features"""
        # Create 3D surface from DEM
        y, x = np.mgrid[:dem['data'].shape[0], :dem['data'].shape[1]]
        
        fig = go.Figure()
        
        # Add terrain surface
        fig.add_trace(go.Surface(
            x=x * dem['transform'].a,
            y=y * dem['transform'].e,
            z=dem['data'],
            colorscale='terrain',
            opacity=0.8,
            showscale=False,
            name='Terrain'
        ))
        
        # Add features by type
        gdf = features['features']
        for feature_type, color in self.feature_colors.items():
            type_features = gdf[gdf['type'] == feature_type]
            if not type_features.empty:
                for _, feature in type_features.iterrows():
                    # Get feature bounds
                    bounds = feature.geometry.bounds
                    i_min = int((bounds[1] - dem['transform'].f) / dem['transform'].e)
                    i_max = int((bounds[3] - dem['transform'].f) / dem['transform'].e)
                    j_min = int((bounds[0] - dem['transform'].c) / dem['transform'].a)
                    j_max = int((bounds[2] - dem['transform'].c) / dem['transform'].a)
                    
                    # Extract feature elevation
                    feature_dem = dem['data'][i_min:i_max+1, j_min:j_max+1]
                    y_f, x_f = np.mgrid[i_min:i_max+1, j_min:j_max+1]
                    
                    fig.add_trace(go.Surface(
                        x=x_f * dem['transform'].a,
                        y=y_f * dem['transform'].e,
                        z=feature_dem,
                        colorscale=[[0, color], [1, color]],
                        opacity=feature.confidence,
                        showscale=False,
                        name=f"{feature_type} (conf: {feature.confidence:.2f})"
                    ))
        
        # Add confidence map as a semi-transparent overlay
        if confidence_map is not None:
            fig.add_trace(go.Surface(
                x=x * dem['transform'].a,
                y=y * dem['transform'].e,
                z=dem['data'] + np.max(dem['data']) * 0.1,  # Slightly above terrain
                surfacecolor=confidence_map,
                colorscale='Viridis',
                opacity=0.3,
                name='Confidence Map'
            ))
        
        # Update layout
        fig.update_layout(
            scene=dict(
                aspectratio=dict(x=1.5, y=1.5, z=0.5),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1)
                )
            ),
            title='LIDAR Feature Detection Results',
            showlegend=True,
            height=800
        )
        
        if output_path:
            fig.write_html(output_path)
        
        return fig
    
    def create_temporal_feature_animation(self,
                                       time_series: List[Dict[str, Any]],
                                       output_path: str = None) -> go.Figure:
        """Create an animated visualization of feature changes over time"""
        fig = go.Figure()
        
        # Get overall bounds
        all_features = pd.concat([
            ts['archaeological_features']['features'] for ts in time_series
        ])
        bounds = all_features.total_bounds
        
        # Create frames for each time point
        frames = []
        for ts_data in time_series:
            frame_traces = []
            
            # Add DEM surface
            y, x = np.mgrid[:ts_data['dem']['data'].shape[0], :ts_data['dem']['data'].shape[1]]
            frame_traces.append(go.Surface(
                x=x * ts_data['dem']['transform'].a,
                y=y * ts_data['dem']['transform'].e,
                z=ts_data['dem']['data'],
                colorscale='terrain',
                opacity=0.8,
                showscale=False
            ))
            
            # Add features by type
            gdf = ts_data['archaeological_features']['features']
            for feature_type, color in self.feature_colors.items():
                type_features = gdf[gdf['type'] == feature_type]
                if not type_features.empty:
                    xs, ys, zs = [], [], []
                    confidences = []
                    for _, feature in type_features.iterrows():
                        x, y = feature.geometry.exterior.xy
                        z = [ts_data['dem']['data'][
                            int((yi - ts_data['dem']['transform'].f) / ts_data['dem']['transform'].e),
                            int((xi - ts_data['dem']['transform'].c) / ts_data['dem']['transform'].a)
                        ] for xi, yi in zip(x, y)]
                        xs.extend(x)
                        ys.extend(y)
                        zs.extend(z)
                        confidences.extend([feature.confidence] * len(x))
                    
                    frame_traces.append(go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=confidences,
                            colorscale=[[0, color], [1, color]],
                            opacity=0.7
                        ),
                        name=feature_type
                    ))
            
            frames.append(go.Frame(
                data=frame_traces,
                name=str(ts_data['timestamp'])
            ))
        
        # Add initial state
        fig.add_traces(frames[0].data)
        
        # Configure animation
        fig.frames = frames
        fig.update_layout(
            scene=dict(
                aspectratio=dict(x=1.5, y=1.5, z=0.5),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1)
                )
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 1000, 'redraw': True},
                        'fromcurrent': True
                    }]
                }]
            }],
            sliders=[{
                'currentvalue': {'prefix': 'Time: '},
                'steps': [{'args': [[f.name], {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate'
                }],
                          'label': f.name,
                          'method': 'animate'} for f in frames]
            }]
        )
        
        if output_path:
            fig.write_html(output_path)
        
        return fig
    
    def visualize_temporal_changes(self,
                                temporal_results: Dict[str, Any],
                                output_path: str = None) -> go.Figure:
        """Create a comprehensive visualization of temporal analysis results"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Feature Changes Over Time',
                'Elevation Changes',
                'Seasonal Patterns',
                'Feature Confidence Evolution'
            ],
            specs=[[{'type': 'scatter'}, {'type': 'heatmap'}],
                  [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Plot tracked features
        tracked_features = temporal_results['tracked_features']
        timestamps = sorted(list(set(
            ts for feature in tracked_features 
            for ts in feature.timestamps
        )))
        
        feature_counts = {t: {
            'vegetation': 0, 'structure': 0, 'terrain': 0, 'archaeological': 0
        } for t in timestamps}
        
        for feature in tracked_features:
            for ts in feature.timestamps:
                feature_counts[ts][feature.attributes['type']] += 1
        
        for feature_type, color in self.feature_colors.items():
            counts = [feature_counts[ts][feature_type] for ts in timestamps]
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=counts,
                    name=f"{feature_type} features",
                    line=dict(color=color)
                ),
                row=1, col=1
            )
        
        # Plot elevation changes
        changes = temporal_results['elevation_changes']
        fig.add_trace(
            go.Heatmap(
                z=changes['data'],
                colorscale='RdBu',
                zmin=-np.max(abs(changes['data'])),
                zmax=np.max(abs(changes['data'])),
                name='Elevation Changes'
            ),
            row=1, col=2
        )
        
        # Plot seasonal patterns
        seasonal = temporal_results['seasonal_patterns']
        if 'vegetation' in seasonal:
            veg_data = seasonal['vegetation']['data']
            fig.add_trace(
                go.Scatter(
                    x=list(veg_data.keys()),
                    y=list(veg_data.values()),
                    name='Vegetation Height',
                    line=dict(color=self.feature_colors['vegetation'])
                ),
                row=2, col=1
            )
        
        if 'hydrology' in seasonal:
            hydro_data = seasonal['hydrology']['data']
            fig.add_trace(
                go.Scatter(
                    x=list(hydro_data.keys()),
                    y=list(hydro_data.values()),
                    name='Water Level',
                    line=dict(color=self.feature_colors['terrain'])
                ),
                row=2, col=1
            )
        
        # Plot confidence evolution
        for feature in tracked_features:
            if len(feature.timestamps) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=feature.timestamps,
                        y=[feature.confidence] * len(feature.timestamps),
                        name=f"{feature.attributes['type']} {feature.feature_id}",
                        line=dict(
                            color=self.feature_colors[feature.attributes['type']],
                            dash='dot'
                        )
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text='Temporal Analysis Results'
        )
        
        fig.update_xaxes(title_text='Time', row=2, col=1)
        fig.update_xaxes(title_text='Time', row=2, col=2)
        fig.update_yaxes(title_text='Count', row=1, col=1)
        fig.update_yaxes(title_text='Feature Confidence', row=2, col=2)
        
        if output_path:
            fig.write_html(output_path)
        
        return fig
    
    def _create_layer_configs(self, data: Dict[str, gpd.GeoDataFrame]) -> List[Dict]:
        """Create layer configurations for Kepler.gl"""
        layers = []
        
        layer_types = {
            'sites': {
                'type': 'point',
                'color': [255, 0, 0]
            },
            'rivers': {
                'type': 'line',
                'color': [0, 0, 255]
            },
            'linguistics': {
                'type': 'point',
                'color': [0, 255, 0]
            }
        }
        
        for name, gdf in data.items():
            layer_type = next((k for k in layer_types.keys() if k in name.lower()), 'default')
            config = layer_types.get(layer_type, {'type': 'point', 'color': [255, 255, 255]})
            
            layers.append({
                'id': name,
                'type': config['type'],
                'config': {
                    'color': config['color'],
                    'opacity': 0.8,
                    'visibility': True
                }
            })
        
        return layers
