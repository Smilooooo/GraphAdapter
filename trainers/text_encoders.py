"""
Text Encoder wrappers for different CLIP-like models.

Provides unified interface for text encoding across:
- OpenAI CLIP
- open_clip (BiomedCLIP, QuiltNet)
- HuggingFace transformers (PLIP)
- Custom models (CONCH)
"""

import torch
import torch.nn as nn
from typing import List, Optional

from clip import clip


class BaseTextEncoder(nn.Module):
    """Base class for text encoders with unified interface."""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self._feature_dim = feature_dim
    
    @property
    def feature_dim(self) -> int:
        return self._feature_dim
    
    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        """
        Encode text prompts to feature vectors.
        
        Args:
            prompts: Embedded prompts [batch_size, seq_len, embed_dim]
            tokenized_prompts: Tokenized text indices [batch_size, seq_len]
            
        Returns:
            Text features [batch_size, feature_dim]
        """
        raise NotImplementedError


class OpenAICLIPTextEncoder(BaseTextEncoder):
    """Text encoder for original OpenAI CLIP models."""
    
    def __init__(self, clip_model):
        # Determine feature dim from text_projection
        feature_dim = clip_model.text_projection.shape[1]
        super().__init__(feature_dim)
        
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # Take features from the EOT embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class OpenCLIPTextEncoder(BaseTextEncoder):
    """Text encoder for standard open_clip models (with exposed transformer)."""
    
    def __init__(self, clip_model, feature_dim: int = 512):
        super().__init__(feature_dim)
        
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = next(clip_model.parameters()).dtype
        self.attn_mask = getattr(clip_model, 'attn_mask', None)

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # Some open_clip models use attention mask
        if self.attn_mask is not None:
            x = self.transformer(x, attn_mask=self.attn_mask)
        else:
            x = self.transformer(x)
            
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # Take features from the EOT embedding
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class DirectOpenCLIPTextEncoder(BaseTextEncoder):
    """
    Text encoder for open_clip models that don't expose internal components.
    
    Used for models like BiomedCLIP (CustomTextCLIP) that use HuggingFace
    text encoders internally and don't expose token_embedding, transformer, etc.
    
    This encoder uses the model's encode_text() method directly.
    """
    
    def __init__(self, clip_model, feature_dim: int = 512, tokenizer=None, context_length: int = 256):
        super().__init__(feature_dim)
        
        # Store the underlying model for direct encoding
        self._model = clip_model._model if hasattr(clip_model, '_model') else clip_model
        self._tokenizer = tokenizer if tokenizer is not None else clip_model._tokenizer if hasattr(clip_model, '_tokenizer') else None
        self._context_length = context_length  # BiomedCLIP uses 256
        self.dtype = next(self._model.parameters()).dtype
    
    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        """
        For DirectOpenCLIPTextEncoder, we ignore the prompts parameter and use
        tokenized_prompts directly with encode_text().
        
        Note: This means learnable prompt embeddings (CoOp-style) won't work with
        this encoder. Use for inference/feature extraction only.
        """
        # Use the model's built-in encode_text
        with torch.no_grad():
            text_features = self._model.encode_text(tokenized_prompts)
        return text_features
    
    def encode_text_direct(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        Directly encode text strings (simpler path for text feature extraction).
        Uses the proper tokenizer with correct context_length.
        """
        if self._tokenizer is not None:
            # Use the model's specific tokenizer (e.g., BiomedCLIP with context_length=256)
            tokens = self._tokenizer(texts, context_length=self._context_length).to(device)
        else:
            # Fallback to generic open_clip tokenizer
            import open_clip
            tokens = open_clip.tokenize(texts, context_length=self._context_length).to(device)
        
        with torch.no_grad():
            text_features = self._model.encode_text(tokens)
        return text_features


class TransformersTextEncoder(BaseTextEncoder):
    """Text encoder for HuggingFace transformers CLIP models (PLIP, etc.)."""
    
    def __init__(self, clip_model, tokenizer, feature_dim: int = 512):
        super().__init__(feature_dim)
        
        self.text_model = clip_model.text_model
        self.text_projection = clip_model.text_projection
        self.tokenizer = tokenizer
        self.dtype = next(clip_model.parameters()).dtype
        
        # Store reference to full model for direct encoding
        self._clip_model = clip_model

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        """
        For transformers models, we use the text_model directly.
        Note: prompts here are already embedded, but transformers expects token IDs.
        This method provides a compatible interface.
        """
        # For transformers, we need special handling since the architecture differs
        # We'll use the embedded prompts with the encoder
        
        # Add positional embeddings
        position_ids = torch.arange(prompts.shape[1], device=prompts.device).unsqueeze(0)
        position_embeddings = self.text_model.embeddings.position_embedding(position_ids)
        x = prompts + position_embeddings
        
        # Create attention mask from tokenized_prompts
        attention_mask = (tokenized_prompts != 0).float()
        
        # Extend attention mask for transformer
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Pass through encoder
        encoder_outputs = self.text_model.encoder(
            inputs_embeds=x,
            attention_mask=extended_attention_mask,
        )
        
        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.text_model.final_layer_norm(last_hidden_state)
        
        # Pool at EOS token position
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0]),
            tokenized_prompts.argmax(dim=-1)
        ]
        
        # Project to shared space
        text_features = self.text_projection(pooled_output)
        
        return text_features
    
    def encode_text_direct(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        Directly encode text strings (simpler path for text feature extraction).
        For transformers CLIP, get_text_features() returns BaseModelOutputWithPooling.
        We need to extract the pooler_output tensor.
        """
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # get_text_features returns BaseModelOutputWithPooling, extract pooler_output
            outputs = self._clip_model.get_text_features(**inputs)
            text_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
        
        return text_features


def create_text_encoder(clip_model, framework: str, feature_dim: int, tokenizer=None, context_length: int = 77) -> BaseTextEncoder:
    """
    Factory function to create the appropriate text encoder.
    
    Args:
        clip_model: The loaded CLIP model
        framework: One of "openai_clip", "open_clip", "transformers", "conch"
        feature_dim: Output feature dimension
        tokenizer: Tokenizer (required for transformers framework, optional for open_clip)
        context_length: Max context length (default 77, BiomedCLIP uses 256)
        
    Returns:
        BaseTextEncoder: Appropriate text encoder instance
    """
    if framework == "openai_clip":
        return OpenAICLIPTextEncoder(clip_model)
    elif framework in ("open_clip", "conch"):
        # Check if model exposes transformer (standard CLIP architecture)
        # or if it's a CustomTextCLIP (like BiomedCLIP) that uses HuggingFace internally
        underlying_model = clip_model._model if hasattr(clip_model, '_model') else clip_model
        
        if hasattr(underlying_model, 'transformer') and underlying_model.transformer is not None:
            # Standard open_clip model with exposed components
            return OpenCLIPTextEncoder(underlying_model, feature_dim)
        else:
            # CustomTextCLIP (BiomedCLIP, etc.) - use direct encoding
            print(f"  Using DirectOpenCLIPTextEncoder (CustomTextCLIP detected, context_length={context_length})")
            # Pass tokenizer for proper encoding (e.g., BiomedCLIP needs context_length=256)
            return DirectOpenCLIPTextEncoder(clip_model, feature_dim, tokenizer=tokenizer, context_length=context_length)
    elif framework == "transformers":
        if tokenizer is None:
            raise ValueError("tokenizer is required for transformers framework")
        return TransformersTextEncoder(clip_model, tokenizer, feature_dim)
    else:
        raise ValueError(f"Unknown framework: {framework}")


class UnifiedTextEncoder(nn.Module):
    """
    High-level text encoder that handles all the complexity.
    
    Use this for simpler integration - it handles tokenization and encoding
    in a single interface.
    """
    
    def __init__(self, clip_model, framework: str, feature_dim: int, tokenizer=None):
        super().__init__()
        self.framework = framework
        self._feature_dim = feature_dim
        self._tokenizer = tokenizer
        self._clip_model = clip_model
        
        # Create the underlying encoder
        self.encoder = create_text_encoder(clip_model, framework, feature_dim, tokenizer)
        self.dtype = self.encoder.dtype
        
    @property
    def feature_dim(self) -> int:
        return self._feature_dim
    
    def tokenize(self, texts: List[str], context_length: int = 77) -> torch.Tensor:
        """Tokenize text strings."""
        if self.framework == "transformers":
            return self._tokenizer(texts, padding=True, max_length=context_length, 
                                   truncation=True, return_tensors="pt")["input_ids"]
        elif self.framework in ("open_clip", "conch"):
            import open_clip
            return open_clip.tokenize(texts, context_length=context_length)
        else:
            return clip.tokenize(texts, context_length=context_length)
    
    def get_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        """Get token embeddings from token IDs."""
        if self.framework == "transformers":
            return self._clip_model.text_model.embeddings.token_embedding(tokens)
        else:
            # Handle models without token_embedding (e.g., BiomedCLIP with CustomTextCLIP)
            if hasattr(self._clip_model, 'token_embedding') and self._clip_model.token_embedding is not None:
                return self._clip_model.token_embedding(tokens)
            else:
                # For CustomTextCLIP models, token_embedding is not exposed
                # Return None to signal that direct encode_text should be used
                return None
    
    def encode(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        Full pipeline: tokenize and encode texts to features.
        
        Args:
            texts: List of text strings
            device: Target device
            
        Returns:
            Text features [len(texts), feature_dim]
        """
        if self.framework == "transformers":
            # Use direct encoding for transformers (simpler)
            return self.encoder.encode_text_direct(texts, device)
        
        # For CLIP-style models
        tokens = self.tokenize(texts).to(device)
        embeddings = self.get_embeddings(tokens)
        
        # If embeddings is None, use direct encode_text (for CustomTextCLIP models like BiomedCLIP)
        if embeddings is None:
            import open_clip
            return self._clip_model._model.encode_text(tokens)
        
        embeddings = embeddings.type(self.dtype)
        return self.encoder(embeddings, tokens)
    
    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-embedded prompts."""
        return self.encoder(prompts, tokenized_prompts)
