"""
Satellite data processing utilities
"""

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from pathlib import Path
import geopandas as gpd
from typing import Dict, List, Tuple, Any
from datetime import datetime
import os
from shapely.geometry import box

class SatelliteDataProcessor:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.supported_sources = {
            'sentinel2': {
                'bands': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
                'resolution': 10
            },
            'landsat8': {
                'bands': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
                'resolution': 30
            }
        }
    
    def fetch_sentinel2_imagery(self, 
                              bounds: Dict[str, float],
                              time_range: Dict[str, str]) -> Dict[str, Any]:
        """
        Fetch Sentinel-2 imagery from Planetary Computer
        """
        try:
            import planetary_computer as pc
            from pystac_client import Client
            
            # Create search bbox
            bbox = [
                bounds['min_lon'], bounds['min_lat'],
                bounds['max_lon'], bounds['max_lat']
            ]
            
            # Search for imagery
            catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
            
            search_results = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=f"{time_range['start']}/{time_range['end']}",
                query={"eo:cloud_cover": {"lt": 20}}
            )
            
            # Get first item with minimal cloud cover
            items = list(search_results.get_items())
            if not items:
                raise ValueError("No suitable Sentinel-2 imagery found")
            
            item = items[0]
            signed_item = pc.sign(item)
            
            # Download and process bands
            data = {}
            for band in self.supported_sources['sentinel2']['bands']:
                href = signed_item.assets[band].href
                with rasterio.open(href) as src:
                    data[band] = src.read(1)
            
            return {
                'data': data,
                'metadata': {
                    'date': item.datetime.strftime('%Y-%m-%d'),
                    'cloud_cover': item.properties['eo:cloud_cover'],
                    'transform': item.assets[band].transform,
                    'crs': item.assets[band].crs
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Error fetching Sentinel-2 imagery: {str(e)}")
    
    def fetch_landsat8_imagery(self,
                             bounds: Dict[str, float],
                             time_range: Dict[str, str]) -> Dict[str, Any]:
        """
        Fetch Landsat-8 imagery from Planetary Computer
        """
        try:
            import planetary_computer as pc
            from pystac_client import Client
            
            # Create search bbox
            bbox = [
                bounds['min_lon'], bounds['min_lat'],
                bounds['max_lon'], bounds['max_lat']
            ]
            
            # Search for imagery
            catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
            
            search_results = catalog.search(
                collections=["landsat-8-c2-l2"],
                bbox=bbox,
                datetime=f"{time_range['start']}/{time_range['end']}",
                query={"eo:cloud_cover": {"lt": 20}}
            )
            
            # Get first item with minimal cloud cover
            items = list(search_results.get_items())
            if not items:
                raise ValueError("No suitable Landsat-8 imagery found")
            
            item = items[0]
            signed_item = pc.sign(item)
            
            # Download and process bands
            data = {}
            for band in self.supported_sources['landsat8']['bands']:
                href = signed_item.assets[band].href
                with rasterio.open(href) as src:
                    data[band] = src.read(1)
            
            return {
                'data': data,
                'metadata': {
                    'date': item.datetime.strftime('%Y-%m-%d'),
                    'cloud_cover': item.properties['eo:cloud_cover'],
                    'transform': item.assets[band].transform,
                    'crs': item.assets[band].crs
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Error fetching Landsat-8 imagery: {str(e)}")
    
    def preprocess_imagery(self, data: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess satellite imagery for model input
        """
        # Stack bands
        bands = []
        for band_name in sorted(data['data'].keys()):
            band_data = data['data'][band_name]
            
            # Normalize band values
            band_norm = self._normalize_band(band_data)
            bands.append(band_norm)
        
        # Create multi-band array
        multi_band = np.stack(bands, axis=0)
        
        # Calculate indices
        ndvi = self._calculate_ndvi(data['data'])
        ndwi = self._calculate_ndwi(data['data'])
        
        # Add indices to stack
        multi_band = np.vstack([multi_band, ndvi[np.newaxis, :, :], ndwi[np.newaxis, :, :]])
        
        # Update metadata
        metadata = data['metadata']
        metadata['band_info'] = {
            'count': multi_band.shape[0],
            'indices': ['ndvi', 'ndwi']
        }
        
        return multi_band, metadata
    
    def _normalize_band(self, band_data: np.ndarray) -> np.ndarray:
        """Normalize band values to 0-1 range"""
        min_val = np.nanpercentile(band_data, 1)
        max_val = np.nanpercentile(band_data, 99)
        normalized = (band_data - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)
    
    def _calculate_ndvi(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate Normalized Difference Vegetation Index"""
        nir = data.get('B8', data.get('B5'))  # Sentinel-2 B8 or Landsat-8 B5
        red = data.get('B4')  # Both use B4 for red
        
        ndvi = (nir - red) / (nir + red + 1e-8)
        return np.clip(ndvi, -1, 1)
    
    def _calculate_ndwi(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate Normalized Difference Water Index"""
        nir = data.get('B8', data.get('B5'))  # Sentinel-2 B8 or Landsat-8 B5
        green = data.get('B3')  # Both use B3 for green
        
        ndwi = (green - nir) / (green + nir + 1e-8)
        return np.clip(ndwi, -1, 1)
