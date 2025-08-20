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
        
        # Comprehensive evaluation criteria based on hackathon requirements
        return {
            "certification": {
                "weight": 0.05,
                "description": "Presence and validity of certification or credentials",
                "keywords": ["certification", "certificate", "credential", "verified", "validated", "accredited"],
                "section_patterns": ["certification", "certificate", "credentials", "verification", "validation"],
                "evaluation_questions": [
                    "Are proper certifications or credentials present?",
                    "Is the certification valid and relevant to the solution?"
                ]
            },
            "novelty_uniqueness": {
                "weight": 0.15,
                "description": "Novelty and uniqueness of the problem statement and solution approach",
                "keywords": ["novel", "unique", "innovative", "original", "creative", "breakthrough", "distinctive", "problem statement", "realistic", "feasible"],
                "section_patterns": ["novelty", "uniqueness", "innovation", "problem", "statement", "differentiation"],
                "evaluation_questions": [
                    "Is the problem statement clearly defined and realistic?",
                    "Is the technology suggested feasible to implement?",
                    "Is there a working prototype?",
                    "How have you tested your solution for any bias?"
                ]
            },
            "presentation_quality": {
                "weight": 0.15,
                "description": "Quality, clarity, and structure of the presentation",
                "keywords": ["presentation", "clear", "concise", "structured", "why", "what", "how", "services", "LLM", "training", "architecture"],
                "section_patterns": ["presentation", "overview", "introduction", "explanation", "structure"],
                "evaluation_questions": [
                    "Is the presentation clear, concise and structured?",
                    "Is 'Why', 'What', and 'How' explained?",
                    "Are services used clearly mentioned?",
                    "Is the LLM used specified?",
                    "Is training content used described?",
                    "Is the architecture explained?"
                ]
            },
            "technical_architecture": {
                "weight": 0.20,
                "description": "Technical architecture design and implementation details",
                "keywords": ["technical", "architecture", "system", "design", "implementation", "technology", "stack", "infrastructure", "components"],
                "section_patterns": ["technical", "architecture", "system", "design", "implementation", "technology"],
                "evaluation_questions": [
                    "Is the technical architecture clearly presented?",
                    "Are system components and their interactions explained?",
                    "Is the technology stack appropriate for the solution?"
                ]
            },
            "ethical_considerations": {
                "weight": 0.10,
                "description": "Ethical implications and bias considerations",
                "keywords": ["ethical", "ethics", "bias", "gender", "neutral", "racial", "religion", "fair", "responsible", "inclusive"],
                "section_patterns": ["ethics", "ethical", "bias", "fairness", "responsibility", "considerations"],
                "evaluation_questions": [
                    "Is it ethical to implement the idea?",
                    "Have you considered factors like gender neutrality?",
                    "Are racial or religious biases being avoided?",
                    "What bias testing methodology was used?"
                ]
            },
            "impact_scalability": {
                "weight": 0.15,
                "description": "Solution impact, scalability, and real-world applicability",
                "keywords": ["impact", "scalable", "better", "deviation", "positive", "negative", "end-users", "Google", "search", "improvement"],
                "section_patterns": ["impact", "scalability", "benefits", "outcomes", "users", "application"],
                "evaluation_questions": [
                    "How is the solution making things better?",
                    "What sort of deviation (positive/negative) can be seen by adopting your solution?",
                    "Will it be scalable? Where else can this be used to make an impact?",
                    "What can your solution do that just a simple Google search does not do?",
                    "How would end-users use it?"
                ]
            },
            "completeness_implementation": {
                "weight": 0.20,
                "description": "Completeness of design, solution implementation, and use of required technologies",
                "keywords": ["DPK", "IBM", "Granite", "RAG", "Agentic", "AWS", "must-have", "traditional AI", "LLM", "Gen AI", "implementation", "complete"],
                "section_patterns": ["implementation", "solution", "technology", "DPK", "IBM", "AWS", "Granite", "RAG"],
                "evaluation_questions": [
                    "Have you used one of the must-haves in your solution?",
                    "What are the must-have techs that have been used in your solution?",
                    "If using Traditional AI, is DPK usage present (mandatory)?",
                    "How different is the solution compared to other LLMs being used?",
                    "How is this solution better than using LLMs/Gen AI?",
                    "Is IBM or AWS or both mentioned and used?",
                    "Is RAG/Agentic usage clearly demonstrated?"
                ]
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
        Map slides to evaluation criteria based on semantic similarity and detect missing information.
        
        Args:
            slides_with_embeddings: Slides with embedding data
            
        Returns:
            Slides with criteria mapping information and missing elements detection
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
                
                # Detect missing elements for this slide's content
                missing_elements = self._detect_missing_elements(slide, best_criterion)
                
                # Generate detailed feedback (non-truncated)
                detailed_feedback = self._generate_detailed_feedback(slide, best_criterion, missing_elements)
                
                # Add mapping information to slide
                mapped_slide = slide.copy()
                mapped_slide.update({
                    'criteria_scores': criterion_scores,
                    'primary_criterion': best_criterion,
                    'primary_criterion_score': best_score,
                    'criterion_weight': self.evaluation_criteria[best_criterion]['weight'],
                    'missing_elements': missing_elements,
                    'detailed_feedback': detailed_feedback,
                    'evaluation_questions': self.evaluation_criteria[best_criterion]['evaluation_questions'],
                    'mapping_timestamp': datetime.now().isoformat()
                })
                
                mapped_slides.append(mapped_slide)
            
            self.logger.info(f"✅ Mapped all slides to evaluation criteria with missing elements detection")
            return mapped_slides
            
        except Exception as e:
            self.logger.error(f"❌ Failed to map slides to criteria: {e}")
            raise e
    
    def _detect_missing_elements(self, slide: Dict[str, Any], criterion: str) -> List[str]:
        """Detect missing elements based on the evaluation criterion."""
        missing_elements = []
        content = slide.get('content', '').lower()
        title = slide.get('title', '').lower()
        combined_text = f"{title} {content}".lower()
        
        criterion_data = self.evaluation_criteria[criterion]
        
        # Check for specific missing elements based on criterion
        if criterion == "certification":
            if not any(keyword in combined_text for keyword in ["certificate", "certification", "credential", "verified"]):
                missing_elements.append("Certification or credentials not mentioned")
        
        elif criterion == "novelty_uniqueness":
            checks = [
                ("prototype", ["prototype", "working prototype", "demo", "demonstration"]),
                ("bias_testing", ["bias test", "bias testing", "bias methodology", "bias evaluation"]),
                ("feasibility", ["feasible", "feasibility", "realistic", "achievable"]),
                ("problem_definition", ["problem statement", "problem definition", "clear problem"])
            ]
            for check_name, keywords in checks:
                if not any(keyword in combined_text for keyword in keywords):
                    missing_elements.append(f"Missing {check_name.replace('_', ' ')}")
        
        elif criterion == "presentation_quality":
            checks = [
                ("why_explanation", ["why", "reason", "motivation", "purpose"]),
                ("what_explanation", ["what", "solution", "approach", "methodology"]),
                ("how_explanation", ["how", "implementation", "process", "steps"]),
                ("services_used", ["service", "API", "platform", "cloud"]),
                ("llm_mentioned", ["LLM", "language model", "GPT", "BERT", "Granite"]),
                ("architecture_diagram", ["architecture", "diagram", "system design", "technical overview"])
            ]
            for check_name, keywords in checks:
                if not any(keyword in combined_text for keyword in keywords):
                    missing_elements.append(f"Missing {check_name.replace('_', ' ')}")
        
        elif criterion == "technical_architecture":
            checks = [
                ("system_components", ["component", "module", "service", "microservice"]),
                ("technology_stack", ["technology", "stack", "framework", "library"]),
                ("data_flow", ["data flow", "pipeline", "process", "workflow"])
            ]
            for check_name, keywords in checks:
                if not any(keyword in combined_text for keyword in keywords):
                    missing_elements.append(f"Missing {check_name.replace('_', ' ')}")
        
        elif criterion == "ethical_considerations":
            checks = [
                ("ethics_discussion", ["ethical", "ethics", "moral", "responsible"]),
                ("bias_considerations", ["bias", "fair", "fairness", "neutral"]),
                ("gender_neutrality", ["gender", "gender neutral", "inclusive"]),
                ("racial_religious_bias", ["racial", "religion", "cultural", "diversity"])
            ]
            for check_name, keywords in checks:
                if not any(keyword in combined_text for keyword in keywords):
                    missing_elements.append(f"Missing {check_name.replace('_', ' ')}")
        
        elif criterion == "impact_scalability":
            checks = [
                ("impact_measurement", ["impact", "benefit", "improvement", "better"]),
                ("scalability_discussion", ["scalable", "scale", "growth", "expand"]),
                ("user_experience", ["user", "end-user", "customer", "experience"]),
                ("differentiation", ["different", "unique", "advantage", "compared to"])
            ]
            for check_name, keywords in checks:
                if not any(keyword in combined_text for keyword in keywords):
                    missing_elements.append(f"Missing {check_name.replace('_', ' ')}")
        
        elif criterion == "completeness_implementation":
            checks = [
                ("ibm_aws_mention", ["IBM", "AWS", "Amazon", "Watson"]),
                ("dpk_usage", ["DPK", "Data Prep Kit", "data preparation"]),
                ("granite_usage", ["Granite", "IBM Granite", "granite model"]),
                ("rag_agentic", ["RAG", "agentic", "retrieval", "augmented"]),
                ("must_have_tech", ["must-have", "required technology", "mandatory"]),
                ("llm_comparison", ["compared to", "different from", "better than", "LLM"])
            ]
            for check_name, keywords in checks:
                if not any(keyword in combined_text for keyword in keywords):
                    missing_elements.append(f"Missing {check_name.replace('_', ' ')}")
        
        return missing_elements
    
    def _generate_detailed_feedback(self, slide: Dict[str, Any], criterion: str, missing_elements: List[str]) -> str:
        """Generate comprehensive, non-truncated feedback for the slide."""
        criterion_data = self.evaluation_criteria[criterion]
        content = slide.get('content', '')
        title = slide.get('title', '')
        
        feedback_parts = []
        
        # Add criterion-specific feedback
        feedback_parts.append(f"EVALUATION CRITERION: {criterion.replace('_', ' ').title()}")
        feedback_parts.append(f"DESCRIPTION: {criterion_data['description']}")
        feedback_parts.append("")
        
        # Add evaluation questions
        feedback_parts.append("EVALUATION QUESTIONS:")
        for question in criterion_data['evaluation_questions']:
            feedback_parts.append(f"• {question}")
        feedback_parts.append("")
        
        # Add strengths (what's present)
        strengths = []
        keywords_found = []
        for keyword in criterion_data['keywords']:
            if keyword.lower() in f"{title} {content}".lower():
                keywords_found.append(keyword)
        
        if keywords_found:
            strengths.append(f"Contains relevant keywords: {', '.join(keywords_found)}")
        
        if len(content) > 100:
            strengths.append("Substantial content provided")
        
        if title:
            strengths.append("Clear slide title provided")
        
        if strengths:
            feedback_parts.append("STRENGTHS:")
            for strength in strengths:
                feedback_parts.append(f"✓ {strength}")
            feedback_parts.append("")
        
        # Add areas for improvement (missing elements)
        if missing_elements:
            feedback_parts.append("AREAS FOR IMPROVEMENT:")
            for missing in missing_elements:
                feedback_parts.append(f"⚠ {missing}")
            feedback_parts.append("")
        
        # Add specific recommendations
        feedback_parts.append("RECOMMENDATIONS:")
        if criterion == "certification":
            feedback_parts.append("• Include relevant certifications or credentials")
            feedback_parts.append("• Validate authenticity of any certificates presented")
        elif criterion == "novelty_uniqueness":
            feedback_parts.append("• Clearly define the problem statement")
            feedback_parts.append("• Provide evidence of working prototype")
            feedback_parts.append("• Document bias testing methodology")
            feedback_parts.append("• Demonstrate technical feasibility")
        elif criterion == "presentation_quality":
            feedback_parts.append("• Structure presentation with clear Why, What, How")
            feedback_parts.append("• Specify services and LLMs used")
            feedback_parts.append("• Include architecture diagrams")
            feedback_parts.append("• Describe training content used")
        elif criterion == "technical_architecture":
            feedback_parts.append("• Provide detailed system architecture")
            feedback_parts.append("• Explain component interactions")
            feedback_parts.append("• Specify technology stack")
        elif criterion == "ethical_considerations":
            feedback_parts.append("• Address ethical implications")
            feedback_parts.append("• Consider gender neutrality")
            feedback_parts.append("• Avoid racial/religious biases")
            feedback_parts.append("• Document bias testing approach")
        elif criterion == "impact_scalability":
            feedback_parts.append("• Quantify solution impact")
            feedback_parts.append("• Demonstrate scalability potential")
            feedback_parts.append("• Explain user experience")
            feedback_parts.append("• Differentiate from simple search solutions")
        elif criterion == "completeness_implementation":
            feedback_parts.append("• Mention IBM or AWS usage")
            feedback_parts.append("• Document DPK implementation")
            feedback_parts.append("• Specify RAG/Agentic usage")
            feedback_parts.append("• Compare with other LLM solutions")
        
        return "\n".join(feedback_parts)
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_criteria_summary(self, mapped_slides: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive summary of criteria mapping results."""
        criteria_counts = {}
        criteria_scores = {}
        missing_elements_summary = {}
        
        for slide in mapped_slides:
            criterion = slide['primary_criterion']
            score = slide['primary_criterion_score']
            missing = slide.get('missing_elements', [])
            
            if criterion not in criteria_counts:
                criteria_counts[criterion] = 0
                criteria_scores[criterion] = []
                missing_elements_summary[criterion] = []
            
            criteria_counts[criterion] += 1
            criteria_scores[criterion].append(score)
            missing_elements_summary[criterion].extend(missing)
        
        # Calculate averages
        criteria_avg_scores = {
            criterion: np.mean(scores) 
            for criterion, scores in criteria_scores.items()
        }
        
        # Count most common missing elements
        overall_missing = {}
        for criterion_missing in missing_elements_summary.values():
            for missing_item in criterion_missing:
                overall_missing[missing_item] = overall_missing.get(missing_item, 0) + 1
        
        # Calculate coverage scores
        coverage_scores = {}
        for criterion, missing_list in missing_elements_summary.items():
            total_checks = len(self.evaluation_criteria[criterion]['evaluation_questions'])
            missing_count = len(set(missing_list))
            coverage = max(0, (total_checks - missing_count) / total_checks) if total_checks > 0 else 1.0
            coverage_scores[criterion] = coverage
        
        return {
            'total_slides': len(mapped_slides),
            'criteria_distribution': criteria_counts,
            'criteria_average_scores': criteria_avg_scores,
            'criteria_coverage_scores': coverage_scores,
            'most_common_missing_elements': dict(sorted(overall_missing.items(), key=lambda x: x[1], reverse=True)[:10]),
            'missing_elements_by_criterion': missing_elements_summary,
            'evaluation_criteria': self.evaluation_criteria,
            'evaluation_summary': {
                'certification_coverage': coverage_scores.get('certification', 0),
                'novelty_uniqueness_coverage': coverage_scores.get('novelty_uniqueness', 0),
                'presentation_quality_coverage': coverage_scores.get('presentation_quality', 0),
                'technical_architecture_coverage': coverage_scores.get('technical_architecture', 0),
                'ethical_considerations_coverage': coverage_scores.get('ethical_considerations', 0),
                'impact_scalability_coverage': coverage_scores.get('impact_scalability', 0),
                'completeness_implementation_coverage': coverage_scores.get('completeness_implementation', 0)
            }
        }
