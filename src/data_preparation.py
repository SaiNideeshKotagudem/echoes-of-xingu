"""
Dataset preparation and caching utilities
"""

import numpy as np
from pathlib import Path
import json
import shutil
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import rasterio
import geopandas as gpd
from shapely.geometry import box

from .satellite_processing import SatelliteDataProcessor

@dataclass
class TileDimensions:
    width: int
    height: int
    overlap: int

class DatasetPreparator:
    def __init__(self, data_root: Path, config_path: str):
        self.logger = logging.getLogger('xingu_pipeline.data_preparation')
        self.data_root = Path(data_root)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize satellite processor
        self.satellite_processor = SatelliteDataProcessor(
            api_key=self.config.get('planetary_computer_api_key')
        )
        
        # Create necessary directories
        self._create_directories()
    
    def prepare_dataset(self, 
                       tile_dims: TileDimensions,
                       validation_split: float = 0.2,
                       clear_cache: bool = False) -> None:
        """Prepare satellite imagery dataset for training"""
        if clear_cache:
            self._clear_processed_data()
        
        try:
            # 1. Process training data
            self.logger.info("Processing satellite imagery...")
            tile_info = self._process_satellite_data(tile_dims)
            
            # 2. Generate target masks
            self.logger.info("Generating target masks...")
            self._generate_target_masks(tile_info)
            
            # 3. Create train/val split
            self.logger.info("Creating dataset splits...")
            self._create_dataset_splits(tile_info, validation_split)
            
            self.logger.info("Dataset preparation completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Dataset preparation failed: {str(e)}")
            raise
    
    def _create_directories(self):
        """Create necessary directory structure"""
        dirs = [
            self.data_root / 'processed' / 'images',
            self.data_root / 'processed' / 'masks'
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _clear_processed_data(self):
        """Clear existing processed data"""
        processed_dir = self.data_root / 'processed'
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
        self._create_directories()
    
    def _process_satellite_data(self, tile_dims: TileDimensions) -> List[Dict]:
        """Process and tile satellite imagery"""
        # Get study area bounds
        bounds = self.config['study_area']['bounds']
        time_range = self.config['study_area']['time_range']
        
        # Fetch satellite data
        sentinel_data = self.satellite_processor.fetch_sentinel2_imagery(
            bounds=bounds,
            time_range=time_range
        )
        
        # Process satellite data
        processed_data, metadata = self.satellite_processor.preprocess_imagery(
            sentinel_data
        )
        
        # Create tiles
        tile_info = []
        for i in range(0, processed_data.shape[1] - tile_dims.height + 1, 
                      tile_dims.height - tile_dims.overlap):
            for j in range(0, processed_data.shape[2] - tile_dims.width + 1,
                         tile_dims.width - tile_dims.overlap):
                
                # Extract tile
                tile = processed_data[:, 
                                    i:i + tile_dims.height,
                                    j:j + tile_dims.width]
                
                # Calculate tile bounds
                tile_bounds = self._calculate_tile_bounds(
                    i, j, tile_dims, bounds, processed_data.shape
                )
                
                # Save tile
                tile_name = f"tile_{i}_{j}.npy"
                tile_path = self.data_root / 'processed' / 'images' / tile_name
                np.save(tile_path, tile)
                
                tile_info.append({
                    'input_file': f"images/{tile_name}",
                    'bounds': tile_bounds
                })
        
        return tile_info
    
    def _calculate_tile_bounds(self,
                             row: int,
                             col: int,
                             tile_dims: TileDimensions,
                             study_bounds: Dict[str, float],
                             image_shape: Tuple[int, int, int]) -> Dict[str, float]:
        """Calculate geographic bounds for a tile"""
        # Calculate proportions
        row_prop = row / image_shape[1]
        col_prop = col / image_shape[2]
        height_prop = tile_dims.height / image_shape[1]
        width_prop = tile_dims.width / image_shape[2]
        
        # Calculate bounds
        lon_range = study_bounds['max_lon'] - study_bounds['min_lon']
        lat_range = study_bounds['max_lat'] - study_bounds['min_lat']
        
        return {
            'min_lon': study_bounds['min_lon'] + col_prop * lon_range,
            'max_lon': study_bounds['min_lon'] + (col_prop + width_prop) * lon_range,
            'min_lat': study_bounds['min_lat'] + row_prop * lat_range,
            'max_lat': study_bounds['min_lat'] + (row_prop + height_prop) * lat_range
        }
    
    def _generate_target_masks(self, tile_info: List[Dict]) -> None:
        """Generate target masks for each tile"""
        # Load known archaeological sites
        sites_path = self.data_root / 'raw' / 'known_sites.geojson'
        sites_gdf = gpd.read_file(sites_path)
        
        for tile in tile_info:
            # Create mask for tile
            tile_bounds = tile['bounds']
            tile_box = box(
                tile_bounds['min_lon'],
                tile_bounds['min_lat'],
                tile_bounds['max_lon'],
                tile_bounds['max_lat']
            )
            
            # Find sites within tile
            sites_in_tile = sites_gdf[sites_gdf.geometry.intersects(tile_box)]
            
            # Generate binary mask
            mask = self._create_site_mask(sites_in_tile, tile_bounds, tile['input_file'])
            
            # Save mask
            mask_name = tile['input_file'].replace('images/', 'masks/')
            mask_path = self.data_root / 'processed' / mask_name
            np.save(mask_path, mask)
            
            # Update tile info
            tile['target_file'] = mask_name
    
    def _create_site_mask(self,
                         sites: gpd.GeoDataFrame,
                         bounds: Dict[str, float],
                         input_file: str) -> np.ndarray:
        """Create binary mask for archaeological sites"""
        # Load input tile to get dimensions
        input_path = self.data_root / 'processed' / input_file
        input_data = np.load(input_path)
        height, width = input_data.shape[1:]
        
        # Create empty mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Add each site to mask
        for _, site in sites.iterrows():
            # Convert site coordinates to pixel coordinates
            x = int((site.geometry.x - bounds['min_lon']) / 
                   (bounds['max_lon'] - bounds['min_lon']) * width)
            y = int((site.geometry.y - bounds['min_lat']) /
                   (bounds['max_lat'] - bounds['min_lat']) * height)
            
            # Add site to mask with a small radius
            radius = 5  # pixels
            y_indices, x_indices = np.ogrid[:height, :width]
            distances = np.sqrt((x_indices - x)**2 + (y_indices - y)**2)
            mask[distances <= radius] = 1
        
        return mask
    
    def _create_dataset_splits(self,
                             tile_info: List[Dict],
                             validation_split: float) -> None:
        """Create train/validation splits"""
        # Split tiles
        train_tiles, val_tiles = train_test_split(
            tile_info,
            test_size=validation_split,
            random_state=42
        )
        
        # Save split information
        splits = {
            'train': train_tiles,
            'val': val_tiles
        }
        
        for split_name, tiles in splits.items():
            split_file = self.data_root / f'{split_name}_data.json'
            with open(split_file, 'w') as f:
                json.dump(tiles, f)
