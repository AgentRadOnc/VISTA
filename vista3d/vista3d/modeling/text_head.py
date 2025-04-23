from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from monai.utils import optional_import

from .sam_blocks import MLP, PositionEmbeddingRandom, TwoWayTransformer

rearrange, _ = optional_import("einops", name="rearrange")


class Text_Mapping(nn.Module):
    """3D Text Prompt Encoder for volumetric segmentation based on CLIP text embeddings.
    
    This module processes text prompts and maps them to visual features for segmentation
    in 3D medical volumes. It uses a transformer-based architecture similar to the 
    Point_Mapping_SAM module but with text embeddings instead of point coordinates.
    
    Attributes:
        max_text_len (int): Maximum number of tokens in text embeddings to process.
        text_projection (nn.Linear): Projects text embeddings to the transformer dimension.
        feat_downsample (nn.Sequential): Feature downsampling path using 3D convolutions.
        transformer (TwoWayTransformer): Cross-attention transformer for text-image interaction.
        mask_tokens (nn.Embedding): Output tokens for mask generation.
        output_upscaling (nn.Sequential): Upsampling path to restore resolution.
        output_hypernetworks_mlps (MLP): Hypernetwork for primary mask prediction.
        num_add_mask_tokens (int): Number of additional mask tokens for multiple predictions.
        output_add_hypernetworks_mlps (nn.ModuleList): Hypernetworks for additional masks.
        class_aware (bool): Whether to use class-specific handling.
        n_classes (int): Total number of semantic classes supported.
        last_supported (int): Index threshold for differentiating pre-trained vs. zero-shot classes.
        class_embeddings (nn.Embedding): Semantic class embeddings.
    """
    
    def __init__(
        self,
        feature_size,
        text_embedding_dim=512,
        max_text_len=77,
        num_add_mask_tokens=2,
        n_classes=512,
        last_supported=132,
        class_aware=False,
    ):
        """Initialize the 3D Text Mapping model.
        
        Args:
            feature_size (int): Dimensionality of image feature vectors.
            text_embedding_dim (int): Dimensionality of text embeddings from CLIP.
            max_text_len (int): Maximum number of tokens in text embeddings.
            num_add_mask_tokens (int): Number of additional mask tokens for multi-mask prediction.
            n_classes (int): Total number of semantic classes supported.
            last_supported (int): Index threshold for differentiating between pre-trained
                and zero-shot classes.
            class_aware (bool): Whether to use class-specific handling for prompts.
        """
        super().__init__()
        transformer_dim = feature_size
        self.max_text_len = max_text_len
        self.class_aware = class_aware
        
        # Project text embeddings to transformer dimension
        self.text_projection = nn.Linear(text_embedding_dim, transformer_dim)
        
        # Feature downsampling path
        self.feat_downsample = nn.Sequential(
            nn.Conv3d(
                in_channels=feature_size,
                out_channels=feature_size,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm3d(feature_size),
            nn.GELU(),
            nn.Conv3d(
                in_channels=feature_size,
                out_channels=transformer_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.InstanceNorm3d(feature_size),
        )

        # Transformer for cross-attention between text and image features
        self.transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=transformer_dim,
            mlp_dim=512,
            num_heads=8,  # Increased from 4 to 8 for better text-image interaction
        )
        
        # Positional encoding for image features
        self.pe_layer = PositionEmbeddingRandom(transformer_dim // 2)
        
        # Learnable tokens for mask generation
        self.mask_tokens = nn.Embedding(1, transformer_dim)

        # Output upsampling path
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose3d(
                transformer_dim,
                transformer_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.InstanceNorm3d(transformer_dim),
            nn.GELU(),
            nn.Conv3d(
                transformer_dim, transformer_dim, kernel_size=3, stride=1, padding=1
            ),
        )

        # Hypernetwork for mask prediction
        self.output_hypernetworks_mlps = MLP(
            transformer_dim, transformer_dim, transformer_dim, 3
        )

        # Multi-mask output
        self.num_add_mask_tokens = num_add_mask_tokens
        self.output_add_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim, 3)
                for i in range(self.num_add_mask_tokens)
            ]
        )
        
        # Class embeddings for semantic segmentation (if class_aware=True)
        if self.class_aware:
            self.n_classes = n_classes
            self.last_supported = last_supported
            self.class_embeddings = nn.Embedding(n_classes, transformer_dim)
            # Special embeddings for zero-shot classes
            self.zeroshot_embed = nn.Embedding(1, transformer_dim)
            # Embedding for supported classes
            self.supported_embed = nn.Embedding(1, transformer_dim)

    def forward(self, out, text_embeddings, class_vector=None):
        """Forward pass for the Text Mapping model.
        
        Args:
            out (torch.Tensor): Input feature map from the backbone network of shape [B, C, H, W, D].
            text_embeddings (torch.Tensor): Text embeddings from CLIP encoder of shape [B, L, D_text].
            class_vector (torch.Tensor, optional): Class indices for class-aware operation.
        
        Returns:
            torch.Tensor: Predicted segmentation masks of shape [B, N_masks, H, W, D].
        """
        # Feature downsampling
        out_low = self.feat_downsample(out)
        out_shape = out.shape[-3:]
        out = None  # Free memory
        torch.cuda.empty_cache()
        
        # Project text embeddings to transformer dimension
        # First cut to max length if needed
        if text_embeddings.shape[1] > self.max_text_len:
            text_embeddings = text_embeddings[:, :self.max_text_len, :]
        
        # Project CLIP text embeddings to transformer dimension
        text_embedding = self.text_projection(text_embeddings)
        
        # Initialize mask tokens
        output_tokens = self.mask_tokens.weight.unsqueeze(0).expand(
            text_embedding.size(0), -1, -1
        )
        
        # Prepare tokens for transformer
        if self.class_aware and class_vector is not None:
            # Process class-specific information when class_aware is enabled
            class_embeddings = []
            for i in class_vector:
                if i > self.last_supported:
                    # Handle zero-shot classes
                    class_embeddings.append(self.zeroshot_embed.weight)
                else:
                    # Handle pre-trained classes
                    class_embeddings.append(self.supported_embed.weight)
            class_embeddings = torch.stack(class_embeddings)
            
            # Concatenate tokens (mask tokens + text embeddings + class embeddings)
            tokens_all = torch.cat(
                (output_tokens, text_embedding, class_embeddings), dim=1
            )
        else:
            # Just use mask tokens and text embeddings
            tokens_all = torch.cat(
                (output_tokens, text_embedding), dim=1
            )
        
        # Process through transformer in batches to manage memory usage
        masks = []
        batch_size = tokens_all.shape[0]
        for i in range(0, batch_size, 8):  # Process 8 samples at a time
            # Clear memory from previous iterations
            src, upscaled_embedding, hyper_in = None, None, None
            torch.cuda.empty_cache()
            
            # Get current batch
            end_idx = min(i + 8, batch_size)
            tokens = tokens_all[i:end_idx]
            
            # Prepare image features with positional encodings
            src = torch.repeat_interleave(out_low[i:end_idx], 1, dim=0)
            pos_src = self.pe_layer(out_low.shape[-3:]).unsqueeze(0)
            pos_src = torch.repeat_interleave(pos_src, end_idx - i, dim=0)
            
            b, c, h, w, d = src.shape
            
            # Apply transformer for cross-attention
            hs, src = self.transformer(src, pos_src, tokens)
            
            # Extract mask tokens that now contain text-conditioned information
            mask_tokens_out = hs[:, :1, :]
            
            # Generate projection weights with hypernetwork
            hyper_in = self.output_hypernetworks_mlps(mask_tokens_out)
            
            # Reshape features and upsample
            src = src.transpose(1, 2).view(b, c, h, w, d)
            upscaled_embedding = self.output_upscaling(src)
            
            b, c, h, w, d = upscaled_embedding.shape
            
            # Generate mask predictions
            batch_masks = (hyper_in @ upscaled_embedding.view(b, c, h * w * d)).view(
                b, -1, h, w, d
            )
            masks.append(batch_masks)
            
        # Combine batch results
        masks = torch.cat(masks, dim=0)
        return masks
```