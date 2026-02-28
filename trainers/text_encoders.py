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
    """Text encoder for open_clip models (BiomedCLIP, QuiltNet, etc.)."""
    
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
        """
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(device)
        text_features = self._clip_model.get_text_features(**inputs)
        return text_features


def create_text_encoder(clip_model, framework: str, feature_dim: int, tokenizer=None) -> BaseTextEncoder:
    """
    Factory function to create the appropriate text encoder.
    
    Args:
        clip_model: The loaded CLIP model
        framework: One of "openai_clip", "open_clip", "transformers", "conch"
        feature_dim: Output feature dimension
        tokenizer: Tokenizer (required for transformers framework)
        
    Returns:
        BaseTextEncoder: Appropriate text encoder instance
    """
    if framework == "openai_clip":
        return OpenAICLIPTextEncoder(clip_model)
    elif framework in ("open_clip", "conch"):
        return OpenCLIPTextEncoder(clip_model, feature_dim)
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
            return self._clip_model.token_embedding(tokens)
    
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
        embeddings = self.get_embeddings(tokens).type(self.dtype)
        
        return self.encoder(embeddings, tokens)
    
    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-embedded prompts."""
        return self.encoder(prompts, tokenized_prompts)
