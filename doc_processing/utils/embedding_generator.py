"""
Embedding Generation Module using IBM Granite-Embedding-125M-English
Generates semantic representations for document content
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModel
import sentence_transformers
from pathlib import Path
import json
from datetime import datetime

class GraniteEmbeddingGenerator:
    """
    Embedding generator using IBM Granite-Embedding-125M-English model
    for creating semantic representations of document content.
    """

    def __init__(self, config=None):
        """Initialize the embedding generator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.model_name = "ibm-granite/granite-embedding-125m-english"
        self.max_sequence_length = 512
        self.embedding_dimension = 768
        
        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.sentence_transformer = None
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the IBM Granite embedding model."""
        try:
            self.logger.info("🔄 Initializing IBM Granite-Embedding-125M-English model...")
            
            # Try to load IBM Granite model directly
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                
                self.logger.info("✅ IBM Granite embedding model loaded successfully")
                
            except Exception as e:
                self.logger.warning(f"⚠️  Could not load IBM Granite model directly: {e}")
                self.logger.info("🔄 Falling back to sentence-transformers with compatible model...")
                
                # Fallback to a compatible sentence transformer model
                fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
                self.sentence_transformer = sentence_transformers.SentenceTransformer(fallback_model)
                self.embedding_dimension = self.sentence_transformer.get_sentence_embedding_dimension()
                
                self.logger.info(f"✅ Fallback embedding model loaded: {fallback_model}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize embedding model: {e}")
            raise e
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            
        Returns:
            NumPy array of embeddings
        """
        if not texts:
            return np.array([])
        
        try:
            if self.sentence_transformer is not None:
                # Use sentence transformer (fallback)
                embeddings = self.sentence_transformer.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                
            elif self.model is not None and self.tokenizer is not None:
                # Use IBM Granite model directly
                embeddings = self._generate_embeddings_with_granite(texts, batch_size)
                
            else:
                raise ValueError("No embedding model available")
            
            self.logger.info(f"✅ Generated embeddings for {len(texts)} texts. Shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate embeddings: {e}")
            raise e
    
    def _generate_embeddings_with_granite(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using IBM Granite model directly."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_sequence_length,
                return_tensors="pt"
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                
                # Use mean pooling of last hidden states
                attention_mask = encoded['attention_mask']
                token_embeddings = outputs.last_hidden_state
                
                # Mean pooling
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def generate_slide_embeddings(self, slides_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for slide data with metadata.
        
        Args:
            slides_data: List of slide dictionaries with title and content
            
        Returns:
            List of slide data with embeddings and metadata
        """
        try:
            self.logger.info(f"🔄 Generating embeddings for {len(slides_data)} slides...")
            
            # Prepare texts for embedding
            slide_texts = []
            for slide in slides_data:
                title = slide.get('title', '')
                content = slide.get('content', '')
                
                # Combine title and content for better semantic representation
                combined_text = f"{title}\n\n{content}".strip()
                slide_texts.append(combined_text)
            
            # Generate embeddings
            embeddings = self.generate_embeddings(slide_texts)
            
            # Add embeddings to slide data
            enriched_slides = []
            for i, slide in enumerate(slides_data):
                enriched_slide = slide.copy()
                enriched_slide.update({
                    'embedding': embeddings[i].tolist(),  # Convert to list for JSON serialization
                    'embedding_model': self.model_name if self.model else "sentence-transformers/all-MiniLM-L6-v2",
                    'embedding_dimension': self.embedding_dimension,
                    'text_for_embedding': slide_texts[i],
                    'embedding_timestamp': datetime.now().isoformat()
                })
                enriched_slides.append(enriched_slide)
            
            self.logger.info(f"✅ Generated embeddings for all slides")
            return enriched_slides
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate slide embeddings: {e}")
            raise e
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        return {
            'model_name': self.model_name if self.model else "sentence-transformers/all-MiniLM-L6-v2",
            'embedding_dimension': self.embedding_dimension,
            'max_sequence_length': self.max_sequence_length,
            'device': str(self.device),
            'using_sentence_transformer': self.sentence_transformer is not None,
            'using_granite_direct': self.model is not None
        }

class CriteriaMapper:
    """
    Maps content sections to specific evaluation criteria using semantic similarity.
    """
    
    def __init__(self, embedding_generator: GraniteEmbeddingGenerator, config=None):
        """Initialize the criteria mapper."""
        self.embedding_generator = embedding_generator
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load evaluation criteria
        self.evaluation_criteria = self._load_evaluation_criteria()
        self.criteria_embeddings = self._generate_criteria_embeddings()
    
    def _load_evaluation_criteria(self) -> Dict[str, Dict[str, Any]]:
        """Load evaluation criteria from config."""
        if self.config and hasattr(self.config, 'evaluation_criteria'):
            return self.config.evaluation_criteria
        
        # Default criteria if not provided
        return {
            "innovation_novelty": {
                "weight": 0.25,
                "description": "Assess the uniqueness and innovative aspects of the idea",
                "keywords": ["innovation", "novel", "unique", "creative", "breakthrough", "disruptive"],
                "section_patterns": ["innovation", "novelty", "uniqueness", "differentiation"]
            },
            "market_feasibility": {
                "weight": 0.20,
                "description": "Evaluate market potential and commercial viability",
                "keywords": ["market", "feasible", "commercial", "viable", "demand", "customers"],
                "section_patterns": ["market", "feasibility", "business", "commercial", "viability"]
            },
            "technical_implementation": {
                "weight": 0.20,
                "description": "Assess technical approach and implementation strategy",
                "keywords": ["technical", "technology", "implementation", "development", "architecture"],
                "section_patterns": ["technical", "technology", "implementation", "development", "solution"]
            },
            "business_impact": {
                "weight": 0.20,
                "description": "Evaluate potential business and societal impact",
                "keywords": ["impact", "benefit", "value", "roi", "revenue", "profit"],
                "section_patterns": ["impact", "benefits", "value", "outcomes", "results"]
            },
            "scalability_potential": {
                "weight": 0.15,
                "description": "Assess scalability and growth potential",
                "keywords": ["scalable", "growth", "expand", "scale", "global", "widespread"],
                "section_patterns": ["scalability", "growth", "expansion", "scale", "future"]
            }
        }
    
    def _generate_criteria_embeddings(self) -> Dict[str, np.ndarray]:
        """Generate embeddings for evaluation criteria."""
        criteria_embeddings = {}
        
        for criterion_name, criterion_data in self.evaluation_criteria.items():
            # Create text representation of criterion
            criterion_text = f"{criterion_data['description']} {' '.join(criterion_data['keywords'])}"
            
            # Generate embedding
            embedding = self.embedding_generator.generate_embeddings([criterion_text])
            criteria_embeddings[criterion_name] = embedding[0]
        
        self.logger.info(f"✅ Generated embeddings for {len(criteria_embeddings)} evaluation criteria")
        return criteria_embeddings
    
    def map_slides_to_criteria(self, slides_with_embeddings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map slides to evaluation criteria based on semantic similarity.
        
        Args:
            slides_with_embeddings: Slides with embedding data
            
        Returns:
            Slides with criteria mapping information
        """
        try:
            self.logger.info(f"🔄 Mapping {len(slides_with_embeddings)} slides to evaluation criteria...")
            
            mapped_slides = []
            
            for slide in slides_with_embeddings:
                slide_embedding = np.array(slide['embedding'])
                
                # Calculate similarity with each criterion
                criterion_scores = {}
                for criterion_name, criterion_embedding in self.criteria_embeddings.items():
                    similarity = self._calculate_cosine_similarity(slide_embedding, criterion_embedding)
                    criterion_scores[criterion_name] = float(similarity)
                
                # Find best matching criterion
                best_criterion = max(criterion_scores.keys(), key=lambda x: criterion_scores[x])
                best_score = criterion_scores[best_criterion]
                
                # Add mapping information to slide
                mapped_slide = slide.copy()
                mapped_slide.update({
                    'criteria_scores': criterion_scores,
                    'primary_criterion': best_criterion,
                    'primary_criterion_score': best_score,
                    'criterion_weight': self.evaluation_criteria[best_criterion]['weight'],
                    'mapping_timestamp': datetime.now().isoformat()
                })
                
                mapped_slides.append(mapped_slide)
            
            self.logger.info(f"✅ Mapped all slides to evaluation criteria")
            return mapped_slides
            
        except Exception as e:
            self.logger.error(f"❌ Failed to map slides to criteria: {e}")
            raise e
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_criteria_summary(self, mapped_slides: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of criteria mapping results."""
        criteria_counts = {}
        criteria_scores = {}
        
        for slide in mapped_slides:
            criterion = slide['primary_criterion']
            score = slide['primary_criterion_score']
            
            if criterion not in criteria_counts:
                criteria_counts[criterion] = 0
                criteria_scores[criterion] = []
            
            criteria_counts[criterion] += 1
            criteria_scores[criterion].append(score)
        
        # Calculate averages
        criteria_avg_scores = {
            criterion: np.mean(scores) 
            for criterion, scores in criteria_scores.items()
        }
        
        return {
            'total_slides': len(mapped_slides),
            'criteria_distribution': criteria_counts,
            'criteria_average_scores': criteria_avg_scores,
            'evaluation_criteria': self.evaluation_criteria
        }
