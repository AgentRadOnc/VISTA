from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from monai.utils import optional_import

from .sam_blocks import MLP, PositionEmbeddingRandom, TwoWayTransformer

rearrange, _ = optional_import("einops", name="rearrange")


class Point_Mapping_SAM(nn.Module):
    """3D Point Prompt Encoder for volumetric segmentation based on SAM architecture.
    
    This module processes point prompts and optional class information to generate
    segmentation masks in 3D medical volumes. It uses a transformer-based architecture
    with positional encodings and hypernetworks for mask prediction.
    
    Attributes:
        max_prompt (int): Maximum number of point prompts to process in a batch.
        feat_downsample (nn.Sequential): Feature downsampling path using 3D convolutions.
        mask_downsample (nn.Conv3d): Mask downsampling convolution.
        transformer (TwoWayTransformer): Cross-attention transformer for feature-prompt interaction.
        pe_layer (PositionEmbeddingRandom): Random Fourier feature positional encoding.
        point_embeddings (nn.ModuleList): Embeddings for foreground/background points.
        not_a_point_embed (nn.Embedding): Embedding for ignored points (label -1).
        special_class_embed (nn.Embedding): Embedding for special class points (labels 2-3).
        mask_tokens (nn.Embedding): Output tokens for mask generation.
        output_upscaling (nn.Sequential): Upsampling path to restore resolution.
        output_hypernetworks_mlps (MLP): Hypernetwork for primary mask prediction.
        num_add_mask_tokens (int): Number of additional mask tokens for multiple predictions.
        output_add_hypernetworks_mlps (nn.ModuleList): Hypernetworks for additional masks.
        n_classes (int): Total number of semantic classes supported.
        last_supported (int): Index threshold for differentiating pre-trained vs. zero-shot classes.
        class_embeddings (nn.Embedding): Semantic class embeddings.
        zeroshot_embed (nn.Embedding): Embedding for zero-shot classes.
        supported_embed (nn.Embedding): Embedding for supported classes.
    """
    
    def __init__(
        self,
        feature_size,
        max_prompt=32,
        num_add_mask_tokens=2,
        n_classes=512,
        last_supported=132,
    ):
        """Initialize the 3D Point Mapping SAM model.
        
        Args:
            feature_size (int): Dimensionality of feature vectors.
            max_prompt (int): Maximum number of point prompts to process in one batch
                for memory optimization.
            num_add_mask_tokens (int): Number of additional mask tokens for multi-mask prediction.
            n_classes (int): Total number of semantic classes supported.
            last_supported (int): Index threshold for differentiating between pre-trained
                and zero-shot classes.
        """
        super().__init__()
        transformer_dim = feature_size
        self.max_prompt = max_prompt
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

        self.mask_downsample = nn.Conv3d(
            in_channels=2, out_channels=2, kernel_size=3, stride=2, padding=1
        )

        self.transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=transformer_dim,
            mlp_dim=512,
            num_heads=4,
        )
        # Positional encoding layer that transforms point coordinates into high-dimensional features
        # using random Fourier features. This helps the model understand spatial relationships
        # between points in the 3D volume by mapping coordinates to a higher-dimensional space.
        self.pe_layer = PositionEmbeddingRandom(transformer_dim // 2)
        
        # Point embeddings for foreground (label 1) and background (label 0)
        # These learnable embeddings encode semantic meaning for different point types
        # and are added to the positional encodings to create the final point representation
        self.point_embeddings = nn.ModuleList(
            [nn.Embedding(1, transformer_dim), nn.Embedding(1, transformer_dim)]
        )
        
        # Special embedding for ignored points (labeled as -1)
        # Used to mask out points that should not contribute to the segmentation
        self.not_a_point_embed = nn.Embedding(1, transformer_dim)
        
        # Embedding for special class points (labels 2-3)
        # Used for handling additional point categories beyond simple foreground/background
        self.special_class_embed = nn.Embedding(1, transformer_dim)
        
        # Learnable tokens that serve as queries for generating mask predictions
        # These tokens interact with image features through cross-attention in the transformer
        self.mask_tokens = nn.Embedding(1, transformer_dim)

        # Upsampling pathway to restore the spatial resolution of the features
        # Uses transposed convolution to increase resolution, followed by normalization,
        # non-linearity, and a final convolution for refinement
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose3d(
                transformer_dim,
                transformer_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.InstanceNorm3d(transformer_dim),  # Instance normalization for 3D data
            nn.GELU(),  # Non-linear activation function
            nn.Conv3d(
                transformer_dim, transformer_dim, kernel_size=3, stride=1, padding=1
            ),
        )

        # Hypernetwork MLP that generates weights for projecting features into mask space
        # Implements dynamic filter generation for mask prediction based on prompt embeddings
        self.output_hypernetworks_mlps = MLP(
            transformer_dim, transformer_dim, transformer_dim, 3
        )

        ## MultiMask output mechanism
        # Support for predicting multiple mask hypotheses for ambiguous prompts
        self.num_add_mask_tokens = num_add_mask_tokens
        # Separate hypernetworks for each additional mask prediction
        self.output_add_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim, 3)
                for i in range(self.num_add_mask_tokens)
            ]
        )
        # class embedding
        self.n_classes = n_classes
        self.last_supported = last_supported
        # Semantic class embeddings for handling different categories during segmentation
        # Enables the model to perform class-specific segmentation when class information is provided
        self.class_embeddings = nn.Embedding(n_classes, feature_size)
        
        # Special embedding for zero-shot classes (classes not seen during training)
        # Helps the model generalize to new categories not present in the training data
        self.zeroshot_embed = nn.Embedding(1, transformer_dim)
        
        # Embedding for classes that were part of the pre-training dataset
        # Allows the model to differentiate between pre-trained and novel classes
        self.supported_embed = nn.Embedding(1, transformer_dim)

    def forward(self, out, point_coords, point_labels, class_vector=None):
        """Forward pass for the 3D Point Mapping SAM model.
        
        Args:
            out (torch.Tensor): Input feature map from the backbone network.
            point_coords (torch.Tensor): Coordinates of the point prompts.
            point_labels (torch.Tensor): Labels of the point prompts.
            class_vector (torch.Tensor, optional): Class vector for semantic segmentation.
        
        Returns:
            torch.Tensor: Predicted segmentation masks.
        """
        # Feature downsampling to reduce computational complexity
        # Creates a lower-resolution representation of the input features
        out_low = self.feat_downsample(out)
        out_shape = out.shape[-3:]
        out = None  # Free memory
        torch.cuda.empty_cache()
        
        # Point prompt embedding process
        # 1. Shift points to center of voxel grid cells
        points = point_coords + 0.5  # Shift to center of pixel
        
        # 2. Apply positional encoding to point coordinates
        # Maps 3D coordinates to high-dimensional space using random Fourier features
        point_embedding = self.pe_layer.forward_with_coords(points, out_shape)
        
        # 3. Apply semantic meaning embeddings based on point labels
        # Different embeddings for ignored (-1), background (0), foreground (1) points
        # and special class points (2-3)
        point_embedding[point_labels == -1] = 0.0
        point_embedding[point_labels == -1] += self.not_a_point_embed.weight
        point_embedding[point_labels == 0] += self.point_embeddings[0].weight
        point_embedding[point_labels == 1] += self.point_embeddings[1].weight
        point_embedding[point_labels == 2] += (
            self.point_embeddings[0].weight + self.special_class_embed.weight
        )
        point_embedding[point_labels == 3] += (
            self.point_embeddings[1].weight + self.special_class_embed.weight
        )
        
        # Initialize mask tokens - these are learnable queries that will produce mask embeddings
        # through cross-attention with image features in the transformer
        output_tokens = self.mask_tokens.weight

        # Expand mask tokens to batch dimension to process all samples
        output_tokens = output_tokens.unsqueeze(0).expand(
            point_embedding.size(0), -1, -1
        )
        
        # Handle class information for semantic segmentation
        if class_vector is None:
            # If no class vector is provided, use default supported class embedding
            # This allows the model to operate in a class-agnostic mode
            tokens_all = torch.cat(
                (
                    output_tokens,
                    point_embedding,
                    self.supported_embed.weight.unsqueeze(0).expand(
                        point_embedding.size(0), -1, -1
                    ),
                ),
                dim=1,
            )
        else:
            # Process class-specific information by applying appropriate embeddings
            # Classes are split into "supported" (pre-trained) and "zero-shot" (novel) categories
            class_embeddings = []
            for i in class_vector:
                if i > self.last_supported:
                    # Zero-shot classes (novel classes not seen in training)
                    class_embeddings.append(self.zeroshot_embed.weight)
                else:
                    # Supported classes (classes seen during pre-training)
                    class_embeddings.append(self.supported_embed.weight)
            class_embeddings = torch.stack(class_embeddings)
            
            # Concatenate all token types for transformer processing
            tokens_all = torch.cat(
                (output_tokens, point_embedding, class_embeddings), dim=1
            )
            
        # Cross-attention transformer processing
        # Process tokens in batches to manage memory usage
        masks = []
        max_prompt = self.max_prompt
        for i in range(int(np.ceil(tokens_all.shape[0] / max_prompt))):
            # Clear variables from previous iterations to optimize memory usage
            src, upscaled_embedding, hyper_in = None, None, None
            torch.cuda.empty_cache()
            
            # Extract the current batch of tokens
            idx = (i * max_prompt, min((i + 1) * max_prompt, tokens_all.shape[0]))
            tokens = tokens_all[idx[0] : idx[1]]
            
            # Repeat image features for each token in the batch
            # This enables parallel processing of multiple prompts
            src = torch.repeat_interleave(out_low, tokens.shape[0], dim=0)
            
            # Apply positional encodings to the image features
            # This helps the transformer understand the spatial structure of the image
            pos_src = torch.repeat_interleave(
                self.pe_layer(out_low.shape[-3:]).unsqueeze(0), tokens.shape[0], dim=0
            )
            
            b, c, h, w, d = src.shape
            
            # Core transformer processing:
            # - Point prompts interact with image features via cross-attention
            # - Image features are updated based on prompt information
            # - Mask tokens extract relevant features for segmentation
            hs, src = self.transformer(src, pos_src, tokens)
            
            # Extract the updated mask tokens that now contain prompt-conditioned information
            mask_tokens_out = hs[:, :1, :]
            
            # Generate dynamic projection weights through the hypernetwork
            # These weights will be used to project features to mask predictions
            hyper_in = self.output_hypernetworks_mlps(mask_tokens_out)
            
            # Reshape the transformer-processed features back to spatial dimensions
            src = src.transpose(1, 2).view(b, c, h, w, d)
            
            # Upsample features to restore spatial resolution
            upscaled_embedding = self.output_upscaling(src)
            
            b, c, h, w, d = upscaled_embedding.shape
            
            # Generate mask predictions through matrix multiplication between
            # hypernetwork-generated weights and upscaled features
            # This implements an implicit decoder that can adapt to different prompt types
            masks.append(
                (hyper_in @ upscaled_embedding.view(b, c, h * w * d)).view(
                    b, -1, h, w, d
                )
            )
            
        # Combine all batch results into a single tensor of masks
        masks = torch.vstack(masks)
        return masks
