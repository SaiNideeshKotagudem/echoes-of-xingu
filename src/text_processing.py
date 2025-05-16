"""
Text processing utilities for historical and indigenous narratives
"""

import re
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import spacy
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

@dataclass
class SpatialReference:
    text: str
    location_type: str
    coordinates: Optional[Dict[str, float]]
    confidence: float
    context: str

class HistoricalTextProcessor:
    def __init__(self):
        # Load NLP models
        self.nlp = spacy.load("pt_core_news_lg")  # Portuguese model
        self.ner_pipeline = pipeline(
            "token-classification",
            model="joeddav/xlm-roberta-large-xnli",
            tokenizer="joeddav/xlm-roberta-large-xnli"
        )
        
        # Load custom location dictionaries
        self.location_patterns = self._load_location_patterns()
    
    def process_colonial_text(self, text: str) -> List[SpatialReference]:
        """Process colonial era text to extract spatial references"""
        # Normalize text
        text = self._normalize_text(text)
        
        # Split into sentences
        doc = self.nlp(text)
        
        spatial_refs = []
        for sent in doc.sents:
            # Look for location patterns
            locations = self._extract_location_patterns(sent.text)
            
            # Use NER to find additional locations
            ner_locations = self._extract_ner_locations(sent.text)
            
            # Merge and deduplicate findings
            all_locations = self._merge_locations(locations, ner_locations)
            
            # Add context
            for loc in all_locations:
                spatial_refs.append(
                    SpatialReference(
                        text=loc['text'],
                        location_type=loc['type'],
                        coordinates=loc.get('coordinates'),
                        confidence=loc['confidence'],
                        context=sent.text
                    )
                )
        
        return spatial_refs
    
    def process_indigenous_narrative(self, text: str) -> List[SpatialReference]:
        """Process indigenous narratives to extract spatial references"""
        # Similar to colonial text but with indigenous-specific patterns
        spatial_refs = []
        
        # Normalize text
        text = self._normalize_text(text)
        
        # Process each paragraph
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            # Look for cosmological references
            cosmic_refs = self._extract_cosmological_references(para)
            
            # Look for natural feature references
            natural_refs = self._extract_natural_features(para)
            
            # Look for movement patterns
            movement_refs = self._extract_movement_patterns(para)
            
            # Combine all references
            refs = cosmic_refs + natural_refs + movement_refs
            
            for ref in refs:
                spatial_refs.append(
                    SpatialReference(
                        text=ref['text'],
                        location_type=ref['type'],
                        coordinates=ref.get('coordinates'),
                        confidence=ref['confidence'],
                        context=para
                    )
                )
        
        return spatial_refs
    
    def extract_spatial_markers(self, text: str) -> Dict[str, Any]:
        """Extract and structure spatial markers from text"""
        # Process text with NLP pipeline
        doc = self.nlp(text)
        
        markers = {
            'landmarks': [],
            'routes': [],
            'settlements': [],
            'sacred_sites': []
        }
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['LOC', 'FAC']:
                # Determine marker type
                marker_type = self._classify_marker_type(ent.text)
                
                # Create marker entry
                marker = {
                    'text': ent.text,
                    'context': ent.sent.text,
                    'type': marker_type,
                    'confidence': self._calculate_confidence(ent)
                }
                
                markers[marker_type + 's'].append(marker)
        
        return markers
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for processing"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _load_location_patterns(self) -> Dict[str, List[str]]:
        """Load location pattern dictionaries"""
        patterns = {
            'landmarks': [
                r'monte\s+(\w+)',
                r'rio\s+(\w+)',
                r'serra\s+(\w+)'
            ],
            'directions': [
                r'(\d+)\s+léguas?\s+ao\s+(norte|sul|leste|oeste)',
                r'(norte|sul|leste|oeste)\s+de\s+(\w+)'
            ]
        }
        return patterns
    
    def _extract_location_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract locations using pattern matching"""
        locations = []
        
        for category, patterns in self.location_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    locations.append({
                        'text': match.group(0),
                        'type': category,
                        'confidence': 0.8,
                        'span': match.span()
                    })
        
        return locations
    
    def _extract_ner_locations(self, text: str) -> List[Dict[str, Any]]:
        """Extract locations using NER"""
        entities = self.ner_pipeline(text)
        
        locations = []
        for ent in entities:
            if ent['entity'] in ['B-LOC', 'I-LOC', 'B-GPE', 'I-GPE']:
                locations.append({
                    'text': ent['word'],
                    'type': 'location',
                    'confidence': ent['score'],
                    'span': (ent['start'], ent['end'])
                })
        
        return locations
    
    def _merge_locations(self, 
                        pattern_locs: List[Dict[str, Any]], 
                        ner_locs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge and deduplicate location references"""
        merged = []
        seen_spans = set()
        
        # Sort all locations by start position
        all_locs = sorted(pattern_locs + ner_locs, key=lambda x: x['span'][0])
        
        for loc in all_locs:
            span = loc['span']
            if not any(self._spans_overlap(span, s) for s in seen_spans):
                merged.append(loc)
                seen_spans.add(span)
        
        return merged
    
    def _spans_overlap(self, span1: tuple, span2: tuple) -> bool:
        """Check if two spans overlap"""
        return not (span1[1] <= span2[0] or span2[1] <= span1[0])
    
    def _extract_cosmological_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract references to cosmological features"""
        # Implement pattern matching for cosmic/sacred references
        patterns = [
            r'casa\s+dos\s+espíritos',
            r'portal\s+sagrado',
            r'morada\s+dos\s+ancestrais'
        ]
        
        refs = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                refs.append({
                    'text': match.group(0),
                    'type': 'sacred_site',
                    'confidence': 0.7
                })
        
        return refs
    
    def _extract_natural_features(self, text: str) -> List[Dict[str, Any]]:
        """Extract references to natural features"""
        # Implement pattern matching for natural features
        patterns = [
            r'(rio|igarapé)\s+(\w+)',
            r'(monte|serra)\s+(\w+)',
            r'(lago|lagoa)\s+(\w+)'
        ]
        
        refs = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                refs.append({
                    'text': match.group(0),
                    'type': 'natural_feature',
                    'confidence': 0.8
                })
        
        return refs
    
    def _extract_movement_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract references to movement and paths"""
        # Implement pattern matching for movement descriptions
        patterns = [
            r'caminho\s+(\w+)',
            r'rota\s+(\w+)',
            r'trilha\s+(\w+)'
        ]
        
        refs = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                refs.append({
                    'text': match.group(0),
                    'type': 'route',
                    'confidence': 0.7
                })
        
        return refs
    
    def _classify_marker_type(self, text: str) -> str:
        """Classify the type of spatial marker"""
        landmark_patterns = ['monte', 'rio', 'serra']
        settlement_patterns = ['aldeia', 'vila', 'cidade']
        route_patterns = ['caminho', 'trilha', 'rota']
        sacred_patterns = ['sagrado', 'ritual', 'ceremonial']
        
        text_lower = text.lower()
        
        if any(pattern in text_lower for pattern in sacred_patterns):
            return 'sacred_site'
        elif any(pattern in text_lower for pattern in landmark_patterns):
            return 'landmark'
        elif any(pattern in text_lower for pattern in settlement_patterns):
            return 'settlement'
        elif any(pattern in text_lower for pattern in route_patterns):
            return 'route'
        else:
            return 'landmark'  # Default to landmark
    
    def _calculate_confidence(self, entity) -> float:
        """Calculate confidence score for an entity"""
        # Simple heuristic based on entity label and context
        base_score = 0.7
        
        # Adjust based on entity label
        if entity.label_ == 'LOC':
            base_score += 0.1
        elif entity.label_ == 'FAC':
            base_score += 0.05
        
        # Adjust based on presence in known patterns
        text_lower = entity.text.lower()
        if any(pattern in text_lower for pattern in ['rio', 'monte', 'serra']):
            base_score += 0.1
        
        return min(base_score, 1.0)  # Cap at 1.0
