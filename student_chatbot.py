import os
import re
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Sentiment Analysis with Transformers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    SENTIMENT_SUPPORT = True
except ImportError:
    SENTIMENT_SUPPORT = False
    print("⚠️  Transformers not installed. Run: pip install transformers torch")

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables.")

# Core dependencies - with type: ignore for Pylance warnings
import google.generativeai as genai  # type: ignore

# PDF Processing
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠️  PyMuPDF not installed. Run: pip install pymupdf")

# OCR for Images
try:
    from PIL import Image
    import pytesseract
    OCR_SUPPORT = True
except ImportError:
    OCR_SUPPORT = False
    print("⚠️  PIL/pytesseract not installed. Run: pip install pillow pytesseract")

# Vector Search
try:
    import faiss  # type: ignore
    import numpy as np
    FAISS_SUPPORT = True
except ImportError:
    FAISS_SUPPORT = False
    print("⚠️  FAISS not installed. Run: pip install faiss-cpu numpy")

# ==================== CONFIGURATION ====================
class Config:
    """Configuration for the RAG system"""
    
    # Gemini API Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
  
    MODEL_NAME: str = "gemini-2.5-flash"
    
    # Using Gemini's embedding model (without models/ prefix - it's added in the API calls)
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    
    # Generation Parameters
    TEMPERATURE: float = 0.7
    MAX_OUTPUT_TOKENS: int = 4096          # For RAG / LLM routes
    SENTIMENT_MAX_OUTPUT_TOKENS: int = 2048 # Full motivational/emotional responses — no cutoff
    SENTIMENT_TEMPERATURE: float = 0.75    # Warm, expressive emotional support tone
    CONVERSATIONAL_MAX_OUTPUT_TOKENS: int = 300  # Greetings — must be enough for 3 full sentences
    TOP_P: float = 0.95
    TOP_K: int = 40
    
    # RAG Parameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.25
    
    # Storage Paths
    UPLOAD_FOLDER: str = "chat_uploads"
    VECTOR_DB_PATH: str = "vector_database"
    INDEX_FILE: str = "faiss_index.bin"
    METADATA_FILE: str = "chunk_metadata.pkl"
    
    # Intent Classification Keywords
    DOCUMENT_QUERY_KEYWORDS: List[str] = [
        "according to", "in the document", "from the pdf", "in the file",
        "based on the document", "the text says", "mentioned in", "from the image",
        "in the picture", "what does the document", "summarize", "summary"
    ]
    
    # Sentiment/Emotional Query Keywords
    SENTIMENT_QUERY_KEYWORDS: List[str] = [
        "i feel", "i'm feeling", "feeling", "i am sad", "i am happy", "i am worried",
        "i'm sad", "i'm happy", "i'm worried", "i'm stressed", "i'm anxious",
        "i'm frustrated", "i'm confused", "i'm tired", "i'm exhausted",
        "help me feel", "i need support", "i'm struggling", "i'm overwhelmed",
        "i'm nervous", "i'm depressed", "i'm angry", "i'm upset", "cheer me up",
        "motivate me", "i'm scared", "i'm afraid", "i'm lonely"
    ]
    
    # Sentiment Analysis Configuration (Transformer-based)
    SENTIMENT_MODEL: str = "distilbert-base-uncased-finetuned-sst-2-english"
    EMOTION_MODEL: str = "j-hartmann/emotion-english-distilroberta-base"
    SENTIMENT_THRESHOLD: float = 0.6  # Confidence threshold for sentiment routing
    USE_TRANSFORMER_SENTIMENT: bool = True  # Toggle transformer vs keyword-based


# ==================== FLASK COMPATIBILITY CLASSES ====================
class Memory:
    """Memory class for Flask compatibility"""
    
    def __init__(self, chat_history: List[Dict[str, Any]]):
        self.conversation_history: List[Dict[str, Any]] = chat_history
    
    def add_interaction(self, message: str, response: str, intent: str) -> None:
        """Add interaction to conversation history"""
        self.conversation_history.append({
            'query': message,
            'response': response,
            'route': intent,
            'timestamp': datetime.now().isoformat()
        })


class Preprocessor:
    """Preprocessor class for Flask compatibility"""
    
    def clean_and_tokenize(self, text: str) -> Tuple[str, List[str]]:
        """Clean and tokenize text"""
        return (text.lower(), text.split())


class IntentClassifier:
    """Intent classifier class for Flask compatibility"""
    
    def __init__(self, get_route_func):
        self._get_route = get_route_func
    
    def classify(self, query: str, tokens: List[str]) -> str:
        """Classify intent based on route"""
        return self._get_route(query).lower()


class EmotionDetector:
    """Enhanced emotion detector with transformer support"""
    
    def __init__(self, detect_emotion_func):
        self._detect_emotion = detect_emotion_func
        
        # Initialize transformer analyzer if available
        if Config.USE_TRANSFORMER_SENTIMENT and SENTIMENT_SUPPORT:
            self.transformer_analyzer = TransformerSentimentAnalyzer()
        else:
            self.transformer_analyzer = None
    
    def detect(self, query: str) -> str:
        """Detect emotion from query using transformer or fallback"""
        if self.transformer_analyzer:
            result = self.transformer_analyzer.analyze_emotion(query)
            return result['emotion']
        else:
            return self._detect_emotion(query)
    
    def detect_with_confidence(self, query: str) -> Tuple[str, float]:
        """Detect emotion with confidence score"""
        if self.transformer_analyzer:
            result = self.transformer_analyzer.analyze_emotion(query)
            return result['emotion'], result['confidence']
        else:
            emotion = self._detect_emotion(query)
            return emotion, 0.7
    
    def get_analysis_report(self, query: str) -> str:
        """Get detailed sentiment analysis report"""
        if self.transformer_analyzer:
            return self.transformer_analyzer.get_detailed_analysis(query)
        else:
            emotion = self._detect_emotion(query)
            return f"Detected emotion: {emotion} (keyword-based)"


class ToneAdjuster:
    """Tone adjuster class for Flask compatibility"""
    
    def get_motivational_prefix(self, emotion: str) -> str:
        """Get motivational prefix based on emotion"""
        return f"I understand you're feeling {emotion}."


class TransformerSentimentAnalyzer:
    """
    Transformer-based sentiment and emotion detection using Hugging Face models
    
    Architecture:
    - Primary: Emotion detection (6 emotions: sadness, joy, anger, fear, surprise, neutral)
    - Fallback: Sentiment analysis (positive, negative, neutral)
    """
    
    def __init__(self):
        """Initialize the transformer models"""
        self.emotion_pipeline = None
        self.sentiment_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not SENTIMENT_SUPPORT:
            print("⚠️  Transformer sentiment analysis not available")
            return
        
        try:
            print("🔄 Loading sentiment analysis models...")

            # Cast pipeline to Any so Pylance does not complain about overload
            # resolution for task-literal arguments or return-type subscripts.
            _pipeline: Any = pipeline  # type: ignore[assignment]

            # Load emotion detection model (primary)
            self.emotion_pipeline: Any = _pipeline(
                "text-classification",
                model=Config.EMOTION_MODEL,
                device=0 if self.device == "cuda" else -1,
                top_k=None,
            )

            # Load sentiment model (fallback)
            self.sentiment_pipeline: Any = _pipeline(
                "sentiment-analysis",
                model=Config.SENTIMENT_MODEL,
                device=0 if self.device == "cuda" else -1,
            )

            print(f"✓ Sentiment models loaded on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading sentiment models: {e}")
            self.emotion_pipeline = None
            self.sentiment_pipeline = None
    
    def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """
        Analyze emotion using transformer model
        
        Returns:
            dict: {
                'emotion': str,      # Primary emotion (sad, happy, angry, etc.)
                'confidence': float, # Confidence score (0-1)
                'all_scores': dict   # All emotion scores
            }
        """
        if not self.emotion_pipeline:
            return self._fallback_emotion_detection(text)
        
        try:
            # emotion_pipeline is typed Any, so subscripts are unchecked by Pylance.
            raw_results: Any = self.emotion_pipeline(text[:512])
            items: List[Any] = list(raw_results[0])

            # Extract top emotion
            top_emotion: Any = max(items, key=lambda x: float(x["score"]))
            top_label: str = str(top_emotion["label"])
            top_score: float = float(top_emotion["score"])

            # Map emotion labels to our system
            emotion_mapping: Dict[str, str] = {
                'joy': 'happy',
                'sadness': 'sad',
                'anger': 'angry',
                'fear': 'anxious',
                'surprise': 'neutral',
                'neutral': 'neutral',
                'disgust': 'frustrated',
            }

            mapped_emotion = emotion_mapping.get(top_label.lower(), 'neutral')

            # Create scores dictionary
            all_scores: Dict[str, float] = {
                emotion_mapping.get(str(item["label"]).lower(), str(item["label"])): float(item["score"])
                for item in items
            }

            return {
                'emotion': mapped_emotion,
                'confidence': top_score,
                'all_scores': all_scores,
                'raw_label': top_label,
            }
            
        except Exception as e:
            print(f"⚠️  Emotion detection error: {e}")
            return self._fallback_emotion_detection(text)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment polarity using transformer model
        
        Returns:
            dict: {
                'sentiment': str,    # positive, negative, or neutral
                'confidence': float  # Confidence score (0-1)
            }
        """
        if not self.sentiment_pipeline:
            return {'sentiment': 'neutral', 'confidence': 0.5}
        
        try:
            # sentiment_pipeline is typed Any; index and key access are unchecked.
            raw: Any = self.sentiment_pipeline(text[:512])
            item: Any = raw[0]
            return {
                'sentiment': str(item["label"]).lower(),
                'confidence': float(item["score"]),
            }
            
        except Exception as e:
            print(f"⚠️  Sentiment detection error: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.5}
    
    def should_route_to_sentiment(self, text: str) -> Tuple[bool, str, float]:
        """
        Decide if query should be routed to SENTIMENT based on emotion analysis
        
        Returns:
            tuple: (should_route: bool, emotion: str, confidence: float)
        """
        emotion_result = self.analyze_emotion(text)
        
        emotion = emotion_result['emotion']
        confidence = emotion_result['confidence']
        
        # Route to SENTIMENT if:
        # 1. Strong negative emotion detected (sad, angry, anxious, frustrated)
        # 2. Confidence is above threshold
        negative_emotions = ['sad', 'angry', 'anxious', 'frustrated', 'stressed']
        
        if emotion in negative_emotions and confidence >= Config.SENTIMENT_THRESHOLD:
            return True, emotion, confidence
        
        # Also check if explicitly asking for help
        help_keywords = ['i feel', 'i am', "i'm", 'help me', 'support']
        if any(keyword in text.lower() for keyword in help_keywords):
            if emotion in negative_emotions:
                return True, emotion, confidence
        
        return False, emotion, confidence
    
    def _fallback_emotion_detection(self, text: str) -> Dict[str, Any]:
        """Fallback to keyword-based emotion detection"""
        text_lower = text.lower()
        
        emotions = {
            'sad': ['sad', 'depressed', 'down', 'unhappy', 'miserable'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'irritated'],
            'anxious': ['anxious', 'worried', 'nervous', 'scared', 'afraid'],
            'frustrated': ['frustrated', 'stuck', 'confused'],
            'happy': ['happy', 'excited', 'glad', 'great', 'wonderful'],
            'stressed': ['stressed', 'overwhelmed', 'pressure']
        }
        
        for emotion, keywords in emotions.items():
            if any(keyword in text_lower for keyword in keywords):
                return {
                    'emotion': emotion,
                    'confidence': 0.7,
                    'all_scores': {},
                    'raw_label': emotion
                }
        
        return {
            'emotion': 'neutral',
            'confidence': 0.5,
            'all_scores': {},
            'raw_label': 'neutral'
        }
    
    def get_detailed_analysis(self, text: str) -> str:
        """
        Get a human-readable analysis report
        """
        emotion_result = self.analyze_emotion(text)
        sentiment_result = self.analyze_sentiment(text)
        
        report = f"""
📊 Sentiment Analysis Report:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Primary Emotion: {emotion_result['emotion'].upper()}
Confidence: {emotion_result['confidence']:.2%}
Sentiment: {sentiment_result['sentiment'].upper()}

All Emotion Scores:"""
        
        if emotion_result['all_scores']:
            for emotion, score in sorted(
                emotion_result['all_scores'].items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                bar = "█" * int(score * 20)
                report += f"\n  {emotion:12s} {bar} {score:.2%}"
        
        return report


# ==================== TEXT EXTRACTION ====================
class DocumentProcessor:
    """Extract and clean text from PDFs and Images with proper error handling"""
    
    @staticmethod
    def extract_from_pdf(filepath: str) -> str:
        """
        Extract text from PDF using PyMuPDF with FIXED error handling
        
        FIXED ISSUES:
        - Document is now kept open during entire extraction
        - Proper page count before closing document
        - Better error handling for corrupted PDFs
        """
        if not PDF_SUPPORT:
            raise RuntimeError("PDF support not available. Install PyMuPDF.")
        
        doc = None
        try:
            # Validate file exists
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"PDF file not found: {filepath}")
            
            # Open document
            doc = fitz.open(filepath)
            total_pages: int = len(doc)
            
            if total_pages == 0:
                raise ValueError("PDF has no pages")
            
            text_parts: List[str] = []
            
            # Extract text from all pages
            for page_num in range(total_pages):
                page = doc[page_num]
                text: str = page.get_text("text") # type: ignore
                if text.strip():
                    # Clean up the text
                    text = re.sub(r'\n+', '\n', text)
                    text = re.sub(r' +', ' ', text)
                    text_parts.append(f"=== Page {page_num + 1} ===\n{text}")
            
            # Join all text parts
            full_text: str = "\n\n".join(text_parts)
            
            # Close document AFTER extraction is complete
            doc.close()
            
            print(f"✓ Extracted {len(full_text)} characters from PDF ({total_pages} pages)")
            return full_text
            
        except Exception as e:
            print(f"❌ Error extracting PDF: {e}")
            # Ensure document is closed even if error occurs
            if doc is not None:
                try:
                    doc.close()
                except:
                    pass
            return ""
    
    @staticmethod
    def extract_from_image(filepath: str) -> str:
        """Extract text from image using OCR with better preprocessing"""
        if not OCR_SUPPORT:
            raise RuntimeError("OCR support not available. Install Pillow and pytesseract.")
        
        try:
            # Validate file exists
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Image file not found: {filepath}")
            
            img = Image.open(filepath)
            
            # Better image preprocessing for OCR
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to grayscale
            img = img.convert('L')
            
            # Extract text with better config
            custom_config: str = r'--oem 3 --psm 6'
            text: str = pytesseract.image_to_string(img, config=custom_config)
            
            print(f"✓ Extracted {len(text)} characters from image via OCR")
            return text.strip()
            
        except Exception as e:
            print(f"❌ Error extracting image text: {e}")
            return ""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        # Normalize line breaks
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'"=\n]+', '', text)
        return text.strip()


# ==================== TEXT CHUNKING ====================
class TextChunker:
    """Split text into semantic chunks with overlap"""
    
    @staticmethod
    def chunk_by_sentences(text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """
        Create overlapping chunks based on sentence boundaries
        Returns list of dicts with: {text, start_char, end_char, sentences}
        """
        # Better sentence splitting
        sentences: List[str] = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        chunks: List[Dict[str, Any]] = []
        current_chunk: List[str] = []
        current_size: int = 0
        chunk_start: int = 0
        
        for sentence in sentences:
            sentence_size: int = len(sentence)
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_size + sentence_size > chunk_size and current_chunk:
                chunk_text: str = ' '.join(current_chunk)
                chunk_end: int = chunk_start + len(chunk_text)
                
                chunks.append({
                    'text': chunk_text,
                    'start_char': chunk_start,
                    'end_char': chunk_end,
                    'sentences': len(current_chunk)
                })
                
                # Start new chunk with overlap
                overlap_sentences: List[str] = []
                overlap_size: int = 0
                for sent in reversed(current_chunk):
                    if overlap_size + len(sent) <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_size += len(sent)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
                chunk_start = chunk_end - overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'start_char': chunk_start,
                'end_char': chunk_start + len(chunk_text),
                'sentences': len(current_chunk)
            })
        
        return chunks


# ==================== GEMINI INTERFACE ====================
class GeminiInterface:
    """Interface for Gemini API with embeddings and generation"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)  # type: ignore
        
        # Initialize embedding model - just store the name
        self.embedding_model_name: str = Config.EMBEDDING_MODEL
        
        # Detect the actual embedding dimension by generating a test embedding
        print(f"✓ Detecting embedding dimension for {Config.EMBEDDING_MODEL}...")
        try:
            test_result = genai.embed_content(  # type: ignore
                model=f"models/{Config.EMBEDDING_MODEL}",
                content="test",
                task_type="retrieval_document"
            )
            self.embedding_dimension: int = len(test_result['embedding'])
            print(f"✓ Detected embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            print(f"⚠️  Could not detect dimension, defaulting to 768: {e}")
            self.embedding_dimension = 768
        
        print(f"✓ Using Gemini Embedding Model: {Config.EMBEDDING_MODEL}")
        
        # ── Main generation model (RAG / LLM routes — full token budget) ──────
        generation_config: Dict[str, Any] = {
            "temperature": Config.TEMPERATURE,
            "top_p": Config.TOP_P,
            "top_k": Config.TOP_K,
            "max_output_tokens": Config.MAX_OUTPUT_TOKENS,
        }
        self.model = genai.GenerativeModel(  # type: ignore
            model_name=Config.MODEL_NAME,
            generation_config=generation_config  # type: ignore
        )

        # ── Sentiment model — initialized ONCE, reused on every sentiment call ─
        # Uses lower temperature (0.4) to stay focused and avoid verbose output.
        # max_output_tokens reads from SENTIMENT_MAX_OUTPUT_TOKENS in Config.
        sentiment_gen_config: Dict[str, Any] = {
            "temperature": Config.SENTIMENT_TEMPERATURE,
            "top_p": Config.TOP_P,
            "top_k": Config.TOP_K,
            "max_output_tokens": Config.SENTIMENT_MAX_OUTPUT_TOKENS,
        }
        self.sentiment_model = genai.GenerativeModel(  # type: ignore
            model_name=Config.MODEL_NAME,
            generation_config=sentiment_gen_config  # type: ignore
        )

        # ── Conversational model — for greetings / small-talk ─────────────────
        # Very low token cap so greetings like "hello" never get long definitions.
        conv_gen_config: Dict[str, Any] = {
            "temperature": Config.TEMPERATURE,
            "top_p": Config.TOP_P,
            "top_k": Config.TOP_K,
            "max_output_tokens": Config.CONVERSATIONAL_MAX_OUTPUT_TOKENS,
        }
        self.conversational_model = genai.GenerativeModel(  # type: ignore
            model_name=Config.MODEL_NAME,
            generation_config=conv_gen_config  # type: ignore
        )

        print(f"✓ Initialized Gemini model: {Config.MODEL_NAME}")
        print(f"✓ Sentiment model ready (max_tokens={Config.SENTIMENT_MAX_OUTPUT_TOKENS}, temp={Config.SENTIMENT_TEMPERATURE})")
        print(f"✓ Conversational model ready (max_tokens={Config.CONVERSATIONAL_MAX_OUTPUT_TOKENS})")
    
    def generate_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for multiple texts using Gemini's embedContent API"""
        embeddings: List[List[float]] = []
        
        if show_progress:
            print(f"🔄 Generating embeddings for {len(texts)} texts...")
        
        for i, text in enumerate(texts):
            try:
                # Use the embedContent API directly with models/ prefix
                result = genai.embed_content(  # type: ignore
                    model=f"models/{Config.EMBEDDING_MODEL}",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
                
                if show_progress and (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{len(texts)}")
                    
            except Exception as e:
                print(f"❌ Error generating embedding for text {i}: {e}")
                # Use zero vector as fallback with correct dimension
                embeddings.append([0.0] * self.embedding_dimension)
        
        if show_progress:
            print(f"✓ Generated {len(embeddings)} embeddings")
        
        return np.array(embeddings, dtype=np.float32)
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a query"""
        try:
            result = genai.embed_content(  # type: ignore
                model=f"models/{Config.EMBEDDING_MODEL}",
                content=query,
                task_type="retrieval_query"
            )
            return np.array([result['embedding']], dtype=np.float32)
        except Exception as e:
            print(f"❌ Error generating query embedding: {e}")
            return np.zeros((1, self.embedding_dimension), dtype=np.float32)
    
    def generate_with_context(self, query: str, context: str) -> str:
        """Generate answer using retrieved context"""
        prompt: str = f"""You are a highly knowledgeable AI assistant. Use the provided context to answer the user's question in detail.

Context from documents:
{context}

User Question: {query}

Instructions:
- Provide a comprehensive, detailed answer based on the context
- Explain concepts thoroughly with examples when relevant
- If the context doesn't fully answer the question, use your knowledge to supplement
- Be specific and cite relevant information from the context
- Structure your response clearly with proper explanations

Answer:"""
        
        try:
            response = self.model.generate_content(prompt)  # type: ignore
            return response.text  # type: ignore
        except Exception as e:
            return f"Error generating response: {e}"
    
    def generate_greeting(self, query: str) -> str:
        """Generate a SHORT friendly reply for greetings only — 3 sentences max"""
        prompt: str = f"""Reply to this student greeting in exactly 3 SHORT sentences. Student said: "{query}"
Sentence 1: Greet back (max 6 words).
Sentence 2: Say ready to help (max 8 words).  
Sentence 3: Ask what they need (max 8 words).
Total response must be under 40 words. No lists, no headers."""

        try:
            response = self.conversational_model.generate_content(prompt)  # type: ignore
            text = response.text.strip()  # type: ignore
            # Safety net: if response is still too long, return a fixed reply
            if len(text.split()) > 50:
                return f"Hello! I'm here to help with your studies. What would you like to know today?"
            return text
        except Exception as e:
            return "Hello! I'm here to help with your studies. What would you like to know today?"

    def generate_direct(self, query: str) -> str:
        """Generate answer without context (direct LLM)"""
        prompt: str = f"""You are a helpful AI assistant. Answer the following question clearly and comprehensively.

User Question: {query}

Provide a detailed, well-structured answer with explanations and examples where appropriate.

Answer:"""
        
        try:
            response = self.model.generate_content(prompt)  # type: ignore
            return response.text  # type: ignore
        except Exception as e:
            return f"Error generating response: {e}"
    
    def generate_sentiment_response(self, query: str, emotion: str) -> str:
        """Generate empathetic response for emotional/sentiment queries"""
        prompt = f"""You are a caring, energetic, and empathetic student counselor and life coach. A student has reached out to you.

STUDENT'S MESSAGE: "{query}"
DETECTED EMOTION: {emotion}

STRICT RULES — YOU MUST FOLLOW THESE:
- DO NOT define any words or explain concepts academically.
- DO NOT give dry or textbook-style explanations.
- ALWAYS respond with warmth, energy, and genuine care like a best friend who also happens to be a great motivator.

YOUR RESPONSE MUST INCLUDE ALL OF THESE:
1. **Acknowledge** — Warmly recognize what the student is feeling (2-3 sentences).
2. **Validate** — Reassure them it is completely okay to feel this way (1-2 sentences).
3. **Motivate** — Give 3-5 powerful, specific motivational points or uplifting thoughts tailored to their situation. Be enthusiastic and genuine.
4. **Practical Tips** — Offer 2-3 concrete, actionable steps they can take right now to feel better or move forward.
5. **Encouragement** — End with a strong, uplifting closing message (2-3 sentences) that leaves them feeling empowered.

Aim for a response of around 200-300 words. Use a warm, conversational, and energetic tone — like a friend who truly believes in them."""

        try:
            # ✅ Uses pre-initialized self.sentiment_model — no new object created per call
            response = self.sentiment_model.generate_content(prompt)  # type: ignore
            return response.text  # type: ignore
        except Exception as e:
            print(f"❌ Error generating sentiment response: {e}")
            return "I'm here for you. It sounds like you're going through a tough time. Would you like to talk more about what's on your mind?"


# ==================== VECTOR DATABASE ====================
class VectorDatabase:
    """FAISS-based vector storage with metadata"""
    
    def __init__(self, dimension: int = 768):
        self.dimension: int = dimension
        # Using cosine similarity (inner product after normalization)
        self.index = faiss.IndexFlatIP(dimension)  # type: ignore
        self.chunk_metadata: List[Dict[str, Any]] = []
        print(f"✓ Initialized FAISS index (dimension: {dimension})")
    
    def add_vectors(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add vectors with metadata to the database"""
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)  # type: ignore[call-arg]
        
        # Add to index
        self.index.add(embeddings)  # type: ignore[call-arg]
        
        # Store metadata
        self.chunk_metadata.extend(metadata)
        
        print(f"✓ Added {len(embeddings)} embeddings to index")
        print(f"  Total vectors in index: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar vectors"""
        if self.index.ntotal == 0:
            return []
        
        # Normalize query
        faiss.normalize_L2(query_embedding) 
        
        # Search
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)  # type: ignore
        
        # Get results with metadata
        results: List[Tuple[Dict[str, Any], float]] = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.chunk_metadata):
                results.append((self.chunk_metadata[int(idx)], float(score)))
        
        return results
    
    def save(self, save_dir: str) -> None:
        """Save index and metadata to disk with proper error handling"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS index
        index_path: str = os.path.join(save_dir, Config.INDEX_FILE)
        try:
            faiss.write_index(self.index, index_path)  # type: ignore
            print(f"✓ Saved vector index to {index_path}")
        except Exception as e:
            print(f"❌ Error saving FAISS index: {e}")
        
        # Save metadata
        metadata_path: str = os.path.join(save_dir, Config.METADATA_FILE)
        try:
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.chunk_metadata, f)
            print(f"✓ Saved metadata to {metadata_path}")
        except Exception as e:
            print(f"❌ Error saving metadata: {e}")
    
    def load(self, save_dir: str) -> bool:
        """Load index and metadata from disk"""
        index_path: str = os.path.join(save_dir, Config.INDEX_FILE)
        metadata_path: str = os.path.join(save_dir, Config.METADATA_FILE)
        
        try:
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                self.index = faiss.read_index(index_path)  # type: ignore
                with open(metadata_path, 'rb') as f:
                    self.chunk_metadata = pickle.load(f)
                print(f"✓ Loaded vector database from {save_dir}")
                print(f"  Total vectors: {self.index.ntotal}")
                return True
        except Exception as e:
            print(f"❌ Error loading vector database: {e}")
        
        return False


# ==================== RAG PIPELINE ====================
class RAGPipeline:
    """Complete RAG pipeline: Extract → Chunk → Embed → Store → Retrieve"""
    
    def __init__(self, api_key: str):
        self.llm: GeminiInterface = GeminiInterface(api_key)
        # Initialize VectorDatabase with the correct dimension from the embedding model
        self.vector_db: VectorDatabase = VectorDatabase(dimension=self.llm.embedding_dimension)
        self.document_registry: Dict[str, Dict[str, Any]] = {}
        
        # Try to load existing database
        if os.path.exists(Config.VECTOR_DB_PATH):
            if self.vector_db.load(Config.VECTOR_DB_PATH):
                # Load document registry
                registry_path: str = os.path.join(Config.VECTOR_DB_PATH, "document_registry.pkl")
                if os.path.exists(registry_path):
                    try:
                        with open(registry_path, 'rb') as f:
                            self.document_registry = pickle.load(f)
                    except:
                        pass
    
    def process_document(self, filepath: str, source_name: Optional[str] = None) -> bool:
        """
        Process a document: Extract → Chunk → Embed → Store
        
        FIXED ISSUES:
        - Better file path validation for Windows
        - Proper error handling throughout the pipeline
        - Files are properly closed after processing
        """
        print(f"\n📄 Processing document: {os.path.basename(filepath)}")
        
        # Validate file path (works on Windows and Unix)
        filepath = os.path.abspath(filepath)
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return False
        
        source_name = source_name or os.path.basename(filepath)
        
        # Check if already processed
        file_hash: str = self._get_file_hash(filepath)
        if file_hash in self.document_registry:
            print(f"⚠️  Document already processed: {source_name}")
            return True
        
        # Step 1: Extract text
        file_ext: str = Path(filepath).suffix.lower()
        
        text: str = ""
        if file_ext == '.pdf':
            text = DocumentProcessor.extract_from_pdf(filepath)
        elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            text = DocumentProcessor.extract_from_image(filepath)
        else:
            print(f"❌ Unsupported file type: {file_ext}")
            return False
        
        if not text:
            print("❌ No text extracted from document")
            return False
        
        # Step 2: Clean text
        text = DocumentProcessor.clean_text(text)
        
        # Step 3: Chunk text
        chunks: List[Dict[str, Any]] = TextChunker.chunk_by_sentences(
            text,
            Config.CHUNK_SIZE,
            Config.CHUNK_OVERLAP
        )
        print(f"✓ Created {len(chunks)} chunks from text")
        
        # Step 4: Generate embeddings
        chunk_texts: List[str] = [chunk['text'] for chunk in chunks]
        embeddings: np.ndarray = self.llm.generate_embeddings(chunk_texts)
        
        # Step 5: Prepare metadata
        metadata: List[Dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            metadata.append({
                'text': chunk['text'],
                'source': source_name,
                'chunk_id': i,
                'start_char': chunk['start_char'],
                'end_char': chunk['end_char'],
                'file_hash': file_hash
            })
        
        # Step 6: Add to vector database
        self.vector_db.add_vectors(embeddings, metadata)
        
        # Register document
        self.document_registry[file_hash] = {
            'source': source_name,
            'filepath': filepath,
            'num_chunks': len(chunks),
            'text_length': len(text),
            'processed_at': datetime.now().isoformat()
        }
        
        # Save database
        self.vector_db.save(Config.VECTOR_DB_PATH)
        
        # Save document registry
        registry_path: str = os.path.join(Config.VECTOR_DB_PATH, "document_registry.pkl")
        try:
            with open(registry_path, 'wb') as f:
                pickle.dump(self.document_registry, f)
        except Exception as e:
            print(f"⚠️  Warning: Could not save document registry: {e}")
        
        print("✅ Document processed successfully!")
        return True
    
    def retrieve(self, query: str) -> Tuple[str, List[Tuple[Dict[str, Any], float]]]:
        """Retrieve relevant context for a query"""
        if self.vector_db.index.ntotal == 0:
            return "", []
        
        # Generate query embedding
        query_embedding: np.ndarray = self.llm.generate_query_embedding(query)
        
        # Search for similar chunks
        results: List[Tuple[Dict[str, Any], float]] = self.vector_db.search(
            query_embedding, 
            k=Config.TOP_K_RESULTS
        )
        
        # Filter by similarity threshold
        filtered_results: List[Tuple[Dict[str, Any], float]] = [
            (chunk, score) for chunk, score in results
            if score >= Config.SIMILARITY_THRESHOLD
        ]
        
        if not filtered_results:
            return "", []
        
        # Build context
        context_parts: List[str] = []
        for chunk, score in filtered_results:
            context_parts.append(f"[From {chunk['source']}]\n{chunk['text']}")
        
        context: str = "\n\n---\n\n".join(context_parts)
        return context, filtered_results
    
    def _get_file_hash(self, filepath: str) -> str:
        """Generate hash of file for deduplication"""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        except Exception as e:
            print(f"⚠️  Warning: Could not hash file: {e}")
            return f"error_{os.path.basename(filepath)}"
        return hash_md5.hexdigest()


# ==================== 3-WAY QUERY ROUTER ====================
class ThreeWayRouter:
    """Intelligent 3-way routing: SENTIMENT / LLM / RAG"""
    
    @staticmethod
    def classify_query(
        query: str,
        has_documents: bool,
        llm: Optional[Any] = None,
        sentiment_analyzer: Optional[Any] = None
    ) -> Tuple[str, str]:
        """
        Classify query into one of three routes.

        Returns: (route_type, reason)
        route_type: "SENTIMENT" | "LLM" | "RAG"
        reason: explanation for the routing decision

        ROUTING PRIORITY (deterministic, top-down):
          0. Greeting / small-talk  → LLM  (fast-path, no model call)
          1. Keyword sentiment match → SENTIMENT
          2. Personal-emotion combo  → SENTIMENT
          3. Transformer sentiment   → SENTIMENT (uses pre-built analyzer, NOT a new one)
          4. No documents            → LLM
          5. Doc-specific keywords   → RAG
          6. LLM intent classifier   → RAG | LLM
          7. Fallback                → LLM
        """
        query_lower: str = query.lower().strip()

        print(f"🔍 Router Debug: Analyzing query...")
        print(f"   Query (lowercase): {query_lower}")

        # ── PRIORITY 0: Greetings / small-talk → LLM immediately ──────────────
        greeting_patterns = [
            'hello', 'hi', 'hii', 'hiii', 'hiiii', 'hey', 'good morning', 'good afternoon',
            'good evening', 'good night', 'how are you', 'how are u',
            "what's up", 'whats up', 'sup', 'hiya', 'howdy',
            "how's it going", 'how r u', 'greetings'
        ]
        # A greeting is the ENTIRE query or starts with one of these
        is_greeting = any(
            query_lower == g or query_lower.startswith(g + ' ') or query_lower.startswith(g + '?') or query_lower.startswith(g + '!')
            for g in greeting_patterns
        )
        if is_greeting:
            print(f"   ✓ Greeting detected → LLM (small-talk)")
            return "LLM", "Greeting / small-talk"

        # ── PRIORITY 1: Hard-coded sentiment keyword match ─────────────────────
        sentiment_detected = False
        matched_keyword = None
        for keyword in Config.SENTIMENT_QUERY_KEYWORDS:
            if keyword in query_lower:
                sentiment_detected = True
                matched_keyword = keyword
                break

        if sentiment_detected:
            print(f"   ✓ Matched sentiment keyword: '{matched_keyword}' → SENTIMENT")
            return "SENTIMENT", f"Matched sentiment keyword: '{matched_keyword}'"

        # ── PRIORITY 2: Personal pronoun + emotion word combo ─────────────────
        emotion_words = [
            'feel', 'feeling', 'sad', 'happy', 'stressed', 'anxious',
            'worried', 'frustrated', 'confused', 'tired', 'exhausted',
            'overwhelmed', 'nervous', 'depressed', 'angry', 'upset',
            'lonely', 'scared', 'afraid', 'motivate', 'cheer', 'cry',
            'crying', 'hopeless', 'helpless', 'terrible', 'awful',
            'miserable', 'bored', 'excited', 'disappointed', 'hurt'
        ]
        personal_indicators = ["i'm", "i am", "im ", "i feel", "i've been", "i have been"]

        matched_personal = [p for p in personal_indicators if p in query_lower]
        matched_emotion  = [e for e in emotion_words if e in query_lower]

        if matched_personal and matched_emotion:
            print(f"   ✓ Personal-emotion combo: {matched_personal} + {matched_emotion} → SENTIMENT")
            return "SENTIMENT", f"Personal emotion expression detected ({matched_emotion[0]})"

        print(f"   ✗ No keyword sentiment indicators found")

        # ── PRIORITY 3: Transformer model (use the PRE-BUILT instance) ────────
        # Never create a new TransformerSentimentAnalyzer here — pass it in via
        # the `sentiment_analyzer` parameter to avoid inconsistency and slowness.
        if sentiment_analyzer is not None:
            try:
                should_route, emotion, confidence = sentiment_analyzer.should_route_to_sentiment(query)
                print(f"   🤖 Transformer: {emotion} (confidence: {confidence:.2%})")
                if should_route:
                    print(f"   → DECISION: SENTIMENT route (transformer)")
                    return "SENTIMENT", f"Transformer detected {emotion} (confidence: {confidence:.2%})"
                else:
                    print(f"   🤖 Transformer below threshold — not routing to SENTIMENT")
            except Exception as e:
                print(f"⚠️  Transformer routing failed: {e}")
        elif Config.USE_TRANSFORMER_SENTIMENT and SENTIMENT_SUPPORT:
            # Warn that no pre-built analyzer was passed — don't create a new one
            print(f"   ⚠️  No pre-built sentiment_analyzer passed to router — skipping transformer check")

        # ── PRIORITY 4: No documents → LLM ───────────────────────────────────
        if not has_documents:
            print(f"   → DECISION: LLM route (no documents)")
            return "LLM", "No documents uploaded"

        # ── PRIORITY 5: Explicit document-reference keywords → RAG ────────────
        matched_doc_kw = next(
            (kw for kw in Config.DOCUMENT_QUERY_KEYWORDS if kw in query_lower), None
        )
        if matched_doc_kw:
            print(f"   → DECISION: RAG route (doc keyword: '{matched_doc_kw}')")
            return "RAG", f"Document keyword detected: '{matched_doc_kw}'"

        # ── PRIORITY 6: LLM intent classifier (RAG vs LLM only) ──────────────
        if llm is not None:
            print(f"   📞 Asking LLM to classify (RAG vs LLM)...")
            try:
                classification_prompt = f"""You are a precise query classifier for a document-based chatbot.
The user has uploaded documents. Decide if the query is asking about the uploaded document content (RAG) or is a general knowledge question (LLM).

Rules:
- If the query asks about topics/facts that exist independently (e.g. "what is java", "explain recursion") → LLM
- If the query is about content that would specifically exist in an uploaded document → RAG
- When in doubt, choose LLM

Query: "{query}"

Respond with EXACTLY ONE WORD — either RAG or LLM (no punctuation, no explanation):"""

                response = llm.model.generate_content(classification_prompt)  # type: ignore
                classification: str = response.text.strip().upper()  # type: ignore
                classification = classification.replace(".", "").replace(",", "").strip()
                # Only accept clean single-word answers
                if classification not in ("RAG", "LLM"):
                    classification = "LLM"

                print(f"   LLM classifier responded: '{classification}'")

                if classification == "RAG":
                    print(f"   → DECISION: RAG route (LLM classified)")
                    return "RAG", "LLM classified as document query"
                else:
                    print(f"   → DECISION: LLM route (LLM classified)")
                    return "LLM", "LLM classified as general knowledge"

            except Exception as e:
                print(f"⚠️  LLM classification failed: {e}")

        # ── PRIORITY 7: Fallback → LLM ────────────────────────────────────────
        print(f"   → DECISION: LLM route (default fallback)")
        return "LLM", "Default to general knowledge"


# ==================== OLD QUERY ROUTER (KEPT FOR COMPATIBILITY) ====================
class QueryRouter:
    """
    Intelligent routing between RAG and direct LLM using 3-layer decision:

    Layer 1 — No documents loaded              → always LLM
    Layer 2 — Explicit document keywords found → always RAG
    Layer 3 — LLM intent classification:
               Ask Gemini whether the query needs the uploaded document
               or is a general knowledge question, then route accordingly.
               Falls back to similarity threshold if LLM call fails.
    """

    @staticmethod
    def should_use_rag(
        query: str,
        has_documents: bool,
        vector_db: Optional[Any] = None,
        llm: Optional[Any] = None,
    ) -> bool:
        """
        Determine if RAG should be used.

        Parameters
        ----------
        query         : raw user query string
        has_documents : True when at least one document is indexed
        vector_db     : VectorDatabase instance (for similarity fallback)
        llm           : GeminiInterface instance (for intent classification)
        """
        # --- Layer 1: no documents at all → LLM ---
        if not has_documents:
            return False

        query_lower: str = query.lower()

        # --- Layer 2: explicit document reference keywords → RAG ---
        doc_keywords_present: bool = any(
            keyword in query_lower
            for keyword in Config.DOCUMENT_QUERY_KEYWORDS
        )
        if doc_keywords_present:
            print("🗝️  Explicit document keyword detected → RAG")
            return True

        # --- Layer 3: LLM-based intent classification ---
        if llm is not None:
            try:
                intent_prompt: str = f"""You are a query classifier for a document chatbot.
The user has uploaded a document. Decide if the query below is asking about
the uploaded document's content, OR if it is a general knowledge question
that can be answered without the document.

Rules:
- If the query asks about topics, facts, or concepts that exist independently
  (e.g. "what is java", "what is python", "who is einstein", "explain gravity")
  → answer: GENERAL
- If the query is about content, details, topics, or information that would
  specifically exist in an uploaded document (e.g. asking about chapters,
  names, data, findings, events described in the document)
  → answer: DOCUMENT
- When in doubt and the query seems self-contained as a general question → GENERAL

Query: "{query}"

Reply with exactly one word — either GENERAL or DOCUMENT."""

                classification_response = llm.model.generate_content(intent_prompt)  # type: ignore
                classification: str = classification_response.text.strip().upper()  # type: ignore
                print(f"🤖 Intent classification: {classification}")

                if "DOCUMENT" in classification:
                    print("✅ Classified as document query → RAG")
                    return True
                else:
                    print("💡 Classified as general query → LLM")
                    return False

            except Exception as e:
                print(f"⚠️  Intent classification failed: {e} — using similarity fallback")

        # --- Similarity fallback if LLM classification fails ---
        if vector_db is not None and llm is not None:
            try:
                query_embedding: np.ndarray = llm.generate_query_embedding(query)
                results = vector_db.search(query_embedding, k=1)
                if results:
                    best_score: float = results[0][1]
                    print(f"📊 Similarity fallback score: {best_score:.4f} (threshold: {Config.SIMILARITY_THRESHOLD})")
                    return best_score >= Config.SIMILARITY_THRESHOLD
            except Exception as e:
                print(f"⚠️  Similarity fallback failed: {e}")

        return False


# ==================== MAIN CHATBOT CLASS ====================
class EnhancedRAGChatbot:
    """Main chatbot class integrating RAG pipeline with Gemini Pro"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key: str = api_key or Config.GEMINI_API_KEY
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not provided")
        
        # Initialize RAG pipeline
        self.rag: RAGPipeline = RAGPipeline(self.api_key)
        
        # Initialize LLM
        self.llm: GeminiInterface = GeminiInterface(self.api_key)
        
        # Query router - 3-way routing
        self.router: ThreeWayRouter = ThreeWayRouter()
        
        # Chat history
        self.chat_history: List[Dict[str, Any]] = []
        
        # Flask compatibility - create memory object
        self.memory: Memory = Memory(self.chat_history)
        
        # Flask compatibility - create objects for attributes expected by Flask
        self.preprocessor: Preprocessor = Preprocessor()
        
        self.intent_classifier: IntentClassifier = IntentClassifier(self.get_route_type)
        
        self.emotion_detector: EmotionDetector = EmotionDetector(self._detect_emotion)
        
        self.tone_adjuster: ToneAdjuster = ToneAdjuster()
        
        # Initialize transformer sentiment analyzer
        if Config.USE_TRANSFORMER_SENTIMENT and SENTIMENT_SUPPORT:
            print("🔄 Initializing transformer sentiment analyzer...")
            self.sentiment_analyzer = TransformerSentimentAnalyzer()
            print("✓ Transformer sentiment analyzer ready")
        else:
            self.sentiment_analyzer = None
            print("⚠️  Using keyword-based emotion detection")
        
        print("\n" + "="*70)
        print("Enhanced RAG Chatbot with 3-Way Routing (SENTIMENT/LLM/RAG)")
        print("="*70)
    
    def upload_document(self, filepath: str, source_name: Optional[str] = None) -> bool:
        """Upload and process a document from chat interface"""
        return self.rag.process_document(filepath, source_name)
    
    def _detect_emotion(self, query: str) -> str:
        """Detect specific emotion from query"""
        query_lower = query.lower()
        
        emotions = {
            'sad': ['sad', 'depressed', 'down', 'unhappy'],
            'stressed': ['stressed', 'overwhelmed', 'pressure'],
            'anxious': ['anxious', 'worried', 'nervous', 'scared'],
            'frustrated': ['frustrated', 'annoyed', 'irritated'],
            'confused': ['confused', 'lost', 'don\'t understand'],
            'tired': ['tired', 'exhausted', 'burnt out'],
            'lonely': ['lonely', 'alone', 'isolated'],
            'happy': ['happy', 'excited', 'glad', 'great'],
        }
        
        for emotion, keywords in emotions.items():
            if any(keyword in query_lower for keyword in keywords):
                return emotion
        
        return 'mixed emotions'
    
    def query(self, user_query: str) -> str:
        """Process user query through 3-way intelligent routing"""
        print(f"\n{'='*70}")
        print(f"📝 Query: {user_query}")
        print(f"{'='*70}")
        
        # Classify query into SENTIMENT / LLM / RAG
        has_docs: bool = self.rag.vector_db and self.rag.vector_db.index.ntotal > 0
        route_type, reason = self.router.classify_query(
            user_query,
            has_docs,
            llm=self.llm,
            sentiment_analyzer=self.sentiment_analyzer  # pass pre-built instance, never create new one
        )
        
        print(f"\n🔀 ROUTING DECISION: {route_type}")
        print(f"   └─ Reason: {reason}\n")
        
        response: str = ""
        
        # Route 1: SENTIMENT - Emotional support with transformer analysis
        if route_type == "SENTIMENT":
            print("💚 Route: SENTIMENT (Emotional Support)")
            
            # Use transformer for detailed emotion analysis if available
            if self.sentiment_analyzer:
                emotion_result = self.sentiment_analyzer.analyze_emotion(user_query)
                emotion = emotion_result['emotion']
                confidence = emotion_result['confidence']
                
                print(f"   └─ Detected: {emotion} (confidence: {confidence:.2%})")
                
                # Show detailed analysis for high confidence
                if confidence >= 0.8:
                    print("   └─ High confidence emotion detection")
            else:
                # Fallback to keyword-based detection
                emotion = self._detect_emotion(user_query)
                print(f"   └─ Detected: {emotion} (keyword-based)")
            
            response = self.llm.generate_sentiment_response(user_query, emotion)
        
        # Route 2: RAG - Document-based
        elif route_type == "RAG":
            print("📚 Route: RAG (Document-Based Answer)")
            context, results = self.rag.retrieve(user_query)
            
            if context:
                print(f"✓ Retrieved {len(results)} relevant chunks")
                response = self.llm.generate_with_context(user_query, context)
                
                sources: List[str] = list(set([chunk['source'] for chunk, _ in results]))
                source_info: str = f"\n\n📚 **Sources:** {', '.join(sources)}"
                response = response + source_info
            else:
                print("⚠️  No relevant context found, falling back to LLM")
                response = self.llm.generate_direct(user_query)
        
        # Route 3: LLM - Greeting (short) OR General Knowledge (full answer)
        else:  # route_type == "LLM"
            greeting_patterns = [
                'hello', 'hi', 'hii', 'hiii', 'hiiii', 'hey', 'good morning', 'good afternoon',
                'good evening', 'good night', 'how are you', 'how are u',
                "what's up", 'whats up', 'sup', 'hiya', 'howdy',
                "how's it going", 'how r u', 'greetings'
            ]
            query_lower = user_query.lower().strip()
            is_greeting = any(
                query_lower == g or query_lower.startswith(g + ' ')
                or query_lower.startswith(g + '?') or query_lower.startswith(g + '!')
                for g in greeting_patterns
            )
            if is_greeting:
                print("💬 Route: LLM (Greeting — short reply)")
                response = self.llm.generate_greeting(user_query)
            else:
                print("🧠 Route: LLM (General Knowledge)")
                response = self.llm.generate_direct(user_query)
        
        # Save to history
        self.chat_history.append({
            'query': user_query,
            'response': response,
            'route': route_type,
            'timestamp': datetime.now().isoformat()
        })
        
        # Cache route so get_route_type() doesn't make a second API call
        self._last_route: str = route_type
        
        return response
    
    def process_query(self, user_query: str) -> str:
        """Flask-compatible wrapper for query method"""
        return self.query(user_query)
    
    def get_route_type(self, user_query: str) -> str:
        """Return cached route from last query() — avoids a second API call"""
        if hasattr(self, '_last_route'):
            route = self._last_route
            del self._last_route  # clear after use
            return route
        # Fallback: keyword-only routing, no LLM call
        has_docs: bool = self.rag.vector_db and self.rag.vector_db.index.ntotal > 0
        route_type, _ = self.router.classify_query(
            user_query, has_docs, llm=None, sentiment_analyzer=None
        )
        return route_type
    
    def run_interactive(self) -> None:
        """Run interactive chat session with FIXED file upload handling"""
        print("\n" + "="*70)
        print("Interactive RAG Chatbot with 3-Way Routing")
        print("="*70)
        print("Routes:")
        print("  💚 SENTIMENT - Emotional support & counseling")
        print("  📚 RAG - Document-based answers")
        print("  🧠 LLM - General knowledge")
        print("\nCommands:")
        print("  - Type your question to get an answer")
        print("  - 'analyze <text>' to see detailed sentiment analysis")
        print("  - 'upload <filepath>' to add a document")
        print("  - 'stats' to see database statistics")
        print("  - 'quit' to exit")
        print("="*70 + "\n")
        
        while True:
            try:
                user_input: str = input("\n🎓 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == 'quit':
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'stats':
                    self._print_stats()
                    continue
                
                if user_input.lower().startswith('analyze '):
                    text_to_analyze = user_input[8:].strip()
                    if self.sentiment_analyzer:
                        analysis = self.sentiment_analyzer.get_detailed_analysis(text_to_analyze)
                        print(analysis)
                    else:
                        print("⚠️  Transformer sentiment analysis not available")
                        print("     Using keyword-based detection:")
                        emotion = self._detect_emotion(text_to_analyze)
                        print(f"     Detected emotion: {emotion}")
                    continue
                
                if user_input.lower().startswith('upload '):
                    # FIXED: Better file path handling for Windows
                    filepath: str = user_input[7:].strip()
                    
                    # Remove quotes if present (common on Windows)
                    filepath = filepath.strip('"').strip("'")
                    
                    # Convert to absolute path
                    filepath = os.path.abspath(os.path.expanduser(filepath))
                    
                    if os.path.exists(filepath):
                        self.upload_document(filepath)
                    else:
                        print(f"❌ File not found: {filepath}")
                        print(f"   Searched at: {filepath}")
                    continue
                
                # Process query
                response = self.query(user_input)
                print(f"\n🤖 Assistant:\n{response}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def _print_stats(self) -> None:
        """Print database statistics"""
        print("\n" + "="*70)
        print("Database Statistics")
        print("="*70)
        
        if self.rag.vector_db:
            print(f"Total vectors in index: {self.rag.vector_db.index.ntotal}")
            print(f"Documents processed: {len(self.rag.document_registry)}")
            
            if self.rag.document_registry:
                print("\nProcessed Documents:")
                for doc_hash, info in self.rag.document_registry.items():
                    print(f"  • {info['source']}: {info['num_chunks']} chunks ({info['text_length']} chars)")
        else:
            print("Vector database not initialized")
        
        print("="*70)


# ==================== MAIN FUNCTION ====================
def main() -> None:
    """Main entry point"""
    print("\n" + "="*70)
    print("Enhanced RAG Chatbot with 3-Way Routing (Gemini)")
    print("="*70)
    
    # Check API key
    api_key: str = Config.GEMINI_API_KEY
    if not api_key:
        print("\n❌ ERROR: GEMINI_API_KEY not found!")
        print("Please set your API key in .env file:")
        print("  GEMINI_API_KEY=your_api_key_here")
        print("\nGet your API key from: https://aistudio.google.com/app/apikey")
        return
    
    # Check dependencies
    if not FAISS_SUPPORT:
        print("\n❌ ERROR: Missing required dependencies")
        print("Please install:")
        print("  pip install faiss-cpu numpy")
        return
    
    # Initialize chatbot
    try:
        chatbot: EnhancedRAGChatbot = EnhancedRAGChatbot(api_key)
        chatbot.run_interactive()
    except Exception as e:
        print(f"\n❌ Error initializing chatbot: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


# ==================== FLASK COMPATIBILITY ====================
# Create alias for Flask server compatibility
StudentChatbot = EnhancedRAGChatbot