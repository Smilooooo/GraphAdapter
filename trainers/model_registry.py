"""
Model Registry for loading different CLIP-like foundation models.

Supports:
- OpenAI CLIP (original)
- open_clip models (BiomedCLIP, QuiltNet, etc.)
- HuggingFace transformers models (PLIP, etc.)
- Custom models (CONCH, etc.)

Usage:
    from trainers.model_registry import load_clip_model
    
    model, preprocess, tokenizer = load_clip_model(cfg)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable, Any

# Local CLIP module (original OpenAI)
from clip import clip


# =============================================================================
# Model Configuration Registry
# =============================================================================

MODEL_CONFIGS = {
    # OpenAI CLIP models (default)
    "RN50": {"framework": "openai_clip", "feature_dim": 1024},
    "RN101": {"framework": "openai_clip", "feature_dim": 512},
    "RN50x4": {"framework": "openai_clip", "feature_dim": 640},
    "RN50x16": {"framework": "openai_clip", "feature_dim": 768},
    "ViT-B/32": {"framework": "openai_clip", "feature_dim": 512},
    "ViT-B/16": {"framework": "openai_clip", "feature_dim": 512},
    
    # BiomedCLIP (open_clip)
    "biomedclip": {
        "framework": "open_clip",
        "hub_name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "feature_dim": 512,
    },
    
    # PLIP (transformers)
    "plip": {
        "framework": "transformers",
        "hub_name": "vinid/plip",
        "feature_dim": 512,
    },
    
    # QuiltNet (open_clip)
    "quiltnet": {
        "framework": "open_clip",
        "hub_name": "hf-hub:wisdomik/QuiltNet-B-32",
        "feature_dim": 512,
    },
    
    # CONCH (custom loader)
    "conch": {
        "framework": "conch",
        "hub_name": "MahmoodLab/CONCH",
        "feature_dim": 512,
    },
}


# =============================================================================
# Wrapper Classes for Unified Interface
# =============================================================================

class CLIPModelWrapper(nn.Module):
    """
    Unified wrapper that provides consistent interface across different CLIP implementations.
    
    Attributes:
        visual: Image encoder
        token_embedding: Token embedding layer (for text)
        transformer: Text transformer
        positional_embedding: Positional embeddings for text
        ln_final: Final layer norm for text
        text_projection: Text projection matrix
        logit_scale: Learnable temperature parameter
        dtype: Model dtype (float16 or float32)
    """
    
    def __init__(self, model, tokenizer, framework: str, feature_dim: int):
        super().__init__()
        self.framework = framework
        self._feature_dim = feature_dim
        self._tokenizer = tokenizer
        self._model = model
        
        # Set up unified interface based on framework
        if framework == "openai_clip":
            self._setup_openai_clip(model)
        elif framework == "open_clip":
            self._setup_open_clip(model)
        elif framework == "transformers":
            self._setup_transformers(model)
        elif framework == "conch":
            self._setup_conch(model)
        else:
            raise ValueError(f"Unknown framework: {framework}")
    
    def _setup_openai_clip(self, model):
        """Setup for original OpenAI CLIP."""
        self.visual = model.visual
        self.token_embedding = model.token_embedding
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.logit_scale = model.logit_scale
        self.dtype = model.dtype
        
    def _setup_open_clip(self, model):
        """Setup for open_clip models (BiomedCLIP, QuiltNet)."""
        self.visual = model.visual
        self.token_embedding = model.token_embedding
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.logit_scale = model.logit_scale
        # open_clip models are typically float32
        self.dtype = next(model.parameters()).dtype
        
    def _setup_transformers(self, model):
        """Setup for HuggingFace transformers models (PLIP)."""
        # transformers CLIPModel has different structure
        self.visual = TransformersVisualWrapper(model)
        self.token_embedding = TransformersTokenEmbeddingWrapper(model)
        self.transformer = TransformersTextTransformerWrapper(model)
        # These need special handling for transformers
        self.positional_embedding = model.text_model.embeddings.position_embedding.weight
        self.ln_final = model.text_model.final_layer_norm
        self.text_projection = model.text_projection
        self.logit_scale = model.logit_scale
        self.dtype = next(model.parameters()).dtype
        
    def _setup_conch(self, model):
        """Setup for CONCH model."""
        # CONCH follows open_clip interface
        self._setup_open_clip(model)
    
    @property
    def feature_dim(self) -> int:
        return self._feature_dim
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode images to feature vectors."""
        return self.visual(image.type(self.dtype))
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode tokenized text to feature vectors."""
        if self.framework == "transformers":
            return self._model.get_text_features(text)
        else:
            # OpenAI CLIP / open_clip style
            x = self.token_embedding(text).type(self.dtype)
            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
            return x
    
    def tokenize(self, texts, context_length: int = 77):
        """Tokenize text using the appropriate tokenizer."""
        if self.framework == "transformers":
            return self._tokenizer(texts, padding=True, return_tensors="pt")["input_ids"]
        elif self.framework == "open_clip":
            import open_clip
            return open_clip.tokenize(texts, context_length=context_length)
        else:
            return clip.tokenize(texts, context_length=context_length)


class TransformersVisualWrapper(nn.Module):
    """Wrapper to make transformers vision encoder match CLIP interface."""
    
    def __init__(self, clip_model):
        super().__init__()
        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(x)
        pooled = outputs.pooler_output
        return self.visual_projection(pooled)


class TransformersTokenEmbeddingWrapper(nn.Module):
    """Wrapper for transformers token embedding."""
    
    def __init__(self, clip_model):
        super().__init__()
        self.embeddings = clip_model.text_model.embeddings
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embeddings.token_embedding(x)


class TransformersTextTransformerWrapper(nn.Module):
    """Wrapper for transformers text encoder."""
    
    def __init__(self, clip_model):
        super().__init__()
        self.encoder = clip_model.text_model.encoder
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is in LND format, encoder expects NLD
        x = x.permute(1, 0, 2)
        outputs = self.encoder(inputs_embeds=x)
        x = outputs.last_hidden_state
        return x.permute(1, 0, 2)  # Back to LND


# =============================================================================
# Model Loading Functions
# =============================================================================

def _load_openai_clip(backbone_name: str, device: str = "cpu") -> Tuple[Any, Callable, Callable]:
    """Load original OpenAI CLIP model."""
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        model = torch.jit.load(model_path, map_location=device).eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location=device)
    
    model = clip.build_model(state_dict or model.state_dict())
    
    # Get preprocessing
    _, preprocess = clip.load(backbone_name, device=device, jit=False)
    
    return model, preprocess, clip.tokenize


def _load_open_clip(hub_name: str, device: str = "cpu") -> Tuple[Any, Callable, Callable]:
    """Load open_clip model from HuggingFace Hub."""
    try:
        import open_clip
    except ImportError:
        raise ImportError(
            "open_clip is required for this model. "
            "Install with: pip install open_clip_torch"
        )
    
    model, _, preprocess = open_clip.create_model_and_transforms(hub_name)
    tokenizer = open_clip.get_tokenizer(hub_name)
    model = model.to(device)
    
    return model, preprocess, tokenizer


def _load_transformers(hub_name: str, device: str = "cpu") -> Tuple[Any, Callable, Callable]:
    """Load HuggingFace transformers CLIP model."""
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        raise ImportError(
            "transformers is required for this model. "
            "Install with: pip install transformers"
        )
    
    model = CLIPModel.from_pretrained(hub_name)
    processor = CLIPProcessor.from_pretrained(hub_name)
    model = model.to(device)
    
    # Create preprocess function from processor
    def preprocess(image):
        return processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
    
    return model, preprocess, processor.tokenizer


def _load_conch(hub_name: str, device: str = "cpu") -> Tuple[Any, Callable, Callable]:
    """Load CONCH model."""
    try:
        from conch.open_clip_custom import create_model_from_pretrained
    except ImportError:
        # Fallback to open_clip if conch package not available
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                f"hf-hub:{hub_name}"
            )
            tokenizer = open_clip.get_tokenizer(f"hf-hub:{hub_name}")
            return model.to(device), preprocess, tokenizer
        except:
            raise ImportError(
                "CONCH model requires either the conch package or open_clip. "
                "See: https://github.com/mahmoodlab/CONCH"
            )
    
    model, preprocess = create_model_from_pretrained(f"hf_hub:{hub_name}")
    model = model.to(device)
    
    import open_clip
    tokenizer = open_clip.get_tokenizer(f"hf-hub:{hub_name}")
    
    return model, preprocess, tokenizer


def load_clip_model(cfg, device: str = "cpu") -> CLIPModelWrapper:
    """
    Load a CLIP-like model based on configuration.
    
    Args:
        cfg: Configuration object with MODEL.BACKBONE settings
        device: Device to load model on
        
    Returns:
        CLIPModelWrapper: Unified model wrapper with consistent interface
        
    Config options:
        cfg.MODEL.BACKBONE.NAME: Model name (e.g., "RN50", "biomedclip", "plip")
        cfg.MODEL.BACKBONE.HUB_NAME: (Optional) Override HuggingFace hub name
        cfg.MODEL.BACKBONE.FRAMEWORK: (Optional) Override framework
        cfg.MODEL.BACKBONE.FEATURE_DIM: (Optional) Override feature dimension
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    
    # Check if it's a known model
    if backbone_name in MODEL_CONFIGS:
        config = MODEL_CONFIGS[backbone_name].copy()
    else:
        # Try to use config values
        config = {
            "framework": getattr(cfg.MODEL.BACKBONE, "FRAMEWORK", "openai_clip"),
            "hub_name": getattr(cfg.MODEL.BACKBONE, "HUB_NAME", backbone_name),
            "feature_dim": getattr(cfg.MODEL.BACKBONE, "FEATURE_DIM", 512),
        }
    
    # Allow config overrides
    if hasattr(cfg.MODEL.BACKBONE, "FRAMEWORK") and cfg.MODEL.BACKBONE.FRAMEWORK:
        config["framework"] = cfg.MODEL.BACKBONE.FRAMEWORK
    if hasattr(cfg.MODEL.BACKBONE, "HUB_NAME") and cfg.MODEL.BACKBONE.HUB_NAME:
        config["hub_name"] = cfg.MODEL.BACKBONE.HUB_NAME
    if hasattr(cfg.MODEL.BACKBONE, "FEATURE_DIM") and cfg.MODEL.BACKBONE.FEATURE_DIM:
        config["feature_dim"] = cfg.MODEL.BACKBONE.FEATURE_DIM
    
    framework = config["framework"]
    feature_dim = config["feature_dim"]
    hub_name = config.get("hub_name", backbone_name)
    
    print(f"Loading model: {backbone_name}")
    print(f"  Framework: {framework}")
    print(f"  Feature dim: {feature_dim}")
    if hub_name != backbone_name:
        print(f"  Hub name: {hub_name}")
    
    # Load based on framework
    if framework == "openai_clip":
        model, preprocess, tokenizer = _load_openai_clip(backbone_name, device)
    elif framework == "open_clip":
        model, preprocess, tokenizer = _load_open_clip(hub_name, device)
    elif framework == "transformers":
        model, preprocess, tokenizer = _load_transformers(hub_name, device)
    elif framework == "conch":
        model, preprocess, tokenizer = _load_conch(hub_name, device)
    else:
        raise ValueError(f"Unknown framework: {framework}")
    
    # Wrap in unified interface
    wrapped = CLIPModelWrapper(model, tokenizer, framework, feature_dim)
    
    return wrapped


def get_feature_dim(cfg) -> int:
    """Get the feature dimension for a model configuration."""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    
    if backbone_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[backbone_name]["feature_dim"]
    
    if hasattr(cfg.MODEL.BACKBONE, "FEATURE_DIM"):
        return cfg.MODEL.BACKBONE.FEATURE_DIM
    
    # Default
    return 512
