# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import monai
import numpy as np
import torch
import torch.nn as nn
from monai.utils import optional_import
from scripts.utils.trans_utils import convert_points_to_disc
from scripts.utils.trans_utils import get_largest_connected_component_mask as lcc
from scripts.utils.workflow_utils import sample_points_patch_val

rearrange, _ = optional_import("einops", name="rearrange")
NINF_VALUE = -9999
PINF_VALUE = 9999


class VISTA3D2(nn.Module):
    def __init__(self, image_encoder, class_head, point_head, text_head, feature_size):
        """Initialize the VISTA3D2 model.
        
        Args:
            image_encoder: Encoder network for processing input images
            class_head: Network head for class-based segmentation
            point_head: Network head for point-based segmentation
            text_head: Network head for text-based segmentation
            feature_size: Size of feature vectors produced by the encoder
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.class_head = class_head
        self.point_head = point_head
        self.text_head = text_head
        self.image_embeddings = None
        self.weight_mapper = nn.Sequential(
            nn.Linear(feature_size, 4 * feature_size),
            nn.GELU(),
            nn.InstanceNorm1d(4 * feature_size),
            nn.Linear(4 * feature_size, 1),
        )
        self.auto_freeze = False
        self.point_freeze = False

    def precompute_embedding(self, input_images):
        """Precompute image embeddings for later use.
        
        This method requires sliding window inference to handle large volumes.
        
        Args:
            input_images: Input image tensor
            
        Raises:
            NotImplementedError: This method needs to be implemented in subclasses
        """
        raise NotImplementedError

    def clear_cache(self):
        """Clear cached image embeddings to free memory."""
        self.image_embeddings = None

    def get_bs(self, class_vector, point_coords):
        """Get batch size from either class vector or point coordinates.
        
        Args:
            class_vector: Class vector tensor of shape [B, 1]
            point_coords: Point coordinates tensor of shape [B, N, 3]
            
        Returns:
            int: Batch size
            
        Raises:
            AssertionError: If both class_vector and point_coords are None
        """
        if class_vector is None:
            assert point_coords is not None, "prompt is required"
            return point_coords.shape[0]
        else:
            return class_vector.shape[0]

    def update_point_to_patch(self, patch_coords, point_coords, point_labels):
        """Update point coordinates with respect to patch coordinates.
        
        This method transforms global point coordinates to local patch coordinates
        and filters out points that fall outside the current patch.
        
        Args:
            patch_coords: Slice object representing patch coordinates
            point_coords: Tensor of shape [B, N, 3] containing point coordinates
            point_labels: Tensor of shape [B, N] containing point labels
            
        Returns:
            tuple: (updated_point_coords, updated_point_labels) - Points transformed to patch space
                  Returns (None, None) if no points fall within the patch
        """
        patch_ends = [
            patch_coords[-3].stop,
            patch_coords[-2].stop,
            patch_coords[-1].stop,
        ]
        patch_starts = [
            patch_coords[-3].start,
            patch_coords[-2].start,
            patch_coords[-1].start,
        ]
        # update point coords
        patch_starts = (
            torch.tensor(patch_starts, device=point_coords.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        patch_ends = (
            torch.tensor(patch_ends, device=point_coords.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # [1 N 1]
        indices = torch.logical_and(
            ((point_coords - patch_starts) > 0).all(2),
            ((patch_ends - point_coords) > 0).all(2),
        )
        # check if it's within patch coords
        point_coords = point_coords.clone() - patch_starts
        point_labels = point_labels.clone()
        if indices.any():
            point_labels[~indices] = -1
            point_coords[~indices] = 0
            # also remove padded points, mainly used for inference.
            not_pad_indices = (point_labels != -1).any(0)
            point_coords = point_coords[:, not_pad_indices]
            point_labels = point_labels[:, not_pad_indices]
        else:
            point_coords = None
            point_labels = None
        return point_coords, point_labels

    def connected_components_combine(
        self, logits, point_logits, point_coords, point_labels, mapping_index, thred=0.5
    ):
        """Combine auto-segmentation results with point click responses, or combine previous mask with point click responses.
        
        This method integrates point-based corrections with existing segmentation masks using connected component analysis.
        
        Args:
            logits: Tensor of shape [B, 1, H, W, D] - The existing segmentation logits (auto-segmentation or previous mask)
            point_logits: Tensor of shape [B', 1, H, W, D] - The point-based segmentation logits
            point_coords: List of tensors [B', N, 3] - Coordinates of clicked points
            point_labels: List of tensors [B', N] - Labels of clicked points (1=positive, 0=negative)
            mapping_index: Boolean tensor [B] - Indicates which batch elements have valid point clicks
            thred: float - Threshold value for sigmoid activation (default: 0.5)
            
        Returns:
            Updated logits tensor with point corrections applied
            
        Process:
            1. For regions with NaN values in logits, replace with point_logits
            2. For positive points within existing mask, add connected components containing those points
            3. For negative points, remove corresponding regions from the mask
            4. Connected component analysis ensures spatial coherence of corrections
        """
        logits = (
            logits.as_tensor() if isinstance(logits, monai.data.MetaTensor) else logits
        )
        _logits = logits[mapping_index]
        inside = []
        for i in range(_logits.shape[0]):
            inside.append(
                np.any(
                    [
                        _logits[
                            i,
                            0,
                            round(p[0].item()),
                            round(p[1].item()),
                            round(p[2].item()),
                        ].item()
                        > 0
                        for p in point_coords[i]
                    ]
                )
            )
        inside = torch.tensor(inside).to(logits.device)
        nan_mask = torch.isnan(_logits)
        _logits = torch.nan_to_num(_logits, nan=NINF_VALUE).sigmoid()
        pos_region = point_logits.sigmoid() > thred
        diff_pos = torch.logical_and(
            torch.logical_or(
                (_logits <= thred),
                inside.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
            ),
            pos_region,
        )
        diff_neg = torch.logical_and((_logits > thred), ~pos_region)
        cc = lcc(
            diff_pos, diff_neg, point_coords=point_coords, point_labels=point_labels
        )
        # cc is the region that can be updated by point_logits.
        cc = cc.to(logits.device)
        # Need to replace NaN with point_logits. diff_neg will never lie in nan_mask, only remove unconnected positive region.
        uc_pos_region = torch.logical_and(pos_region, ~cc)
        fill_mask = torch.logical_and(nan_mask, uc_pos_region)
        if fill_mask.any():
            # fill in the mean negative value
            point_logits[fill_mask] = -1
        # replace logits nan value and cc with point_logits
        cc = torch.logical_or(nan_mask, cc).to(logits.dtype)
        logits[mapping_index] *= 1 - cc
        logits[mapping_index] += cc * point_logits
        # debug_ccp(_logits, point_logits.sigmoid(), point_coords, point_labels, diff, cc, logits[mapping_index], np.random.randint(10000))
        return logits

    def gaussian_combine(
        self, logits, point_logits, point_coords, point_labels, mapping_index, radius
    ):
        """Combine point-based and auto-segmentation results using Gaussian weighting.
        
        This method blends the two segmentation results with a spatial Gaussian weight
        centered around the clicked points.
        
        Args:
            logits: Tensor of shape [B, 1, H, W, D] - The auto-segmentation logits
            point_logits: Tensor of shape [B', 1, H, W, D] - The point-based segmentation logits
            point_coords: List of tensors [B', N, 3] - Coordinates of clicked points
            point_labels: List of tensors [B', N] - Labels of clicked points
            mapping_index: Boolean tensor [B] - Indicates which batch elements have valid point clicks
            radius: float or None - Radius of Gaussian influence around points (if None, uses 1/5 of min dimension)
            
        Returns:
            Updated logits tensor with Gaussian-weighted combination of auto and point results
        """
        if radius is None:
            radius = min(point_logits.shape[-3:]) // 5  # empirical value 5
        weight = 1 - convert_points_to_disc(
            point_logits.shape[-3:], point_coords, point_labels, radius=radius
        ).sum(1, keepdims=True)
        weight[weight < 0] = 0
        logits = (
            logits.as_tensor() if isinstance(logits, monai.data.MetaTensor) else logits
        )
        logits[mapping_index] *= weight
        logits[mapping_index] += (1 - weight) * point_logits
        return logits

    def set_auto_grad(self, auto_freeze=False, point_freeze=False):
        """Control gradient flow by selectively freezing model components.
        
        This method allows freezing either the automatic segmentation branch or
        the point-based segmentation branch for targeted training.
        
        Args:
            auto_freeze: bool - Whether to freeze the automatic segmentation branch
            point_freeze: bool - Whether to freeze the point-based segmentation branch
        """
        if auto_freeze != self.auto_freeze:
            if hasattr(self.image_encoder, "set_auto_grad"):
                self.image_encoder.set_auto_grad(
                    auto_freeze=auto_freeze, point_freeze=point_freeze
                )
            else:
                for param in self.image_encoder.parameters():
                    param.requires_grad = (not auto_freeze) and (not point_freeze)
            for param in self.class_head.parameters():
                param.requires_grad = not auto_freeze
            self.auto_freeze = auto_freeze

        if point_freeze != self.point_freeze:
            if hasattr(self.image_encoder, "set_auto_grad"):
                self.image_encoder.set_auto_grad(
                    auto_freeze=auto_freeze, point_freeze=point_freeze
                )
            else:
                for param in self.image_encoder.parameters():
                    param.requires_grad = (not auto_freeze) and (not point_freeze)
            for param in self.point_head.parameters():
                param.requires_grad = not point_freeze
            self.point_freeze = point_freeze

    def forward(
        self,
        input_images,
        text_embeddings=None,
        point_coords=None,
        point_labels=None,
        class_vector=None,
        prompt_class=None,
        patch_coords=None,
        labels=None,
        label_set=None,
        prev_mask=None,
        radius=None,
        val_point_sampler=None,
        **kwargs,
    ):
        """Forward pass for VISTA3D model supporting multiple segmentation modes.
        
        This method handles class-based, point-based, and text-based segmentation,
        as well as combinations of these approaches. It supports both training and
        inference modes, including sliding window inference for large volumes.
        
        Args:
            input_images: Tensor [1, 1, H, W, D] - Input 3D image
            text_embeddings: Optional tensor - Text prompt embeddings for text-based segmentation
            point_coords: Optional tensor [B, N, 3] - Coordinates of clicked points
            point_labels: Optional tensor [B, N] - Labels of clicked points (-1=padding, 0=negative, 1=positive)
            class_vector: Optional tensor [B, 1] - Class indices for class-based segmentation
            prompt_class: Optional tensor [B, 1] - Class indices associated with point prompts
            patch_coords: Optional slice object - Coordinates of current patch in sliding window inference
            labels: Optional tensor [1, 1, H, W, D] - Ground truth labels for point-only evaluation
            label_set: Optional list - Label indices matching the indexes in labels
            prev_mask: Optional tensor [B, N, H, W, D] - Previous segmentation mask for refinement
            radius: Optional float - Radius for Gaussian combination of auto and point results
            val_point_sampler: Optional function - Function to sample points from labels for evaluation
            **kwargs: Additional keyword arguments
                - keep_cache: bool - Whether to cache image embeddings for future use
        
        Returns:
            Tensor [B, 1, H, W, D] - Segmentation logits
            
        Notes:
            - The method supports three main segmentation modes: class-based, point-based, and text-based
            - For class-based segmentation with point refinement, results are combined using either
              Gaussian weighting (during training) or connected components (during validation)
            - For point-only segmentation with a previous mask, connected components are used for refinement
            - Memory management is handled by clearing unused tensors and optionally caching embeddings
        """
        image_size = input_images.shape[-3:]
        device = input_images.device
        if point_coords is None and class_vector is None:
            return NINF_VALUE + torch.zeros([1, 1, *image_size], device=device)

        bs = self.get_bs(class_vector, point_coords)
        if patch_coords is not None:
            # if during validation and perform enable based point-validation.
            if labels is not None and label_set is not None:
                # if labels is not None, sample from labels for each patch.
                if val_point_sampler is None:
                    val_point_sampler = sample_points_patch_val
                point_coords, point_labels, prompt_class = val_point_sampler(
                    labels, patch_coords, label_set
                )
                if prompt_class[0].item() == 0:
                    point_labels[0] = -1
                labels, prev_mask = None, None
            elif point_coords is not None:
                # If not performing patch-based point only validation, use user provided click points for inference.
                # the point clicks is in original image space, convert it to current patch-coordinate space.
                point_coords, point_labels = self.update_point_to_patch(
                    patch_coords, point_coords, point_labels
                )

        if point_coords is not None and point_labels is not None:
            # remove points that used for padding purposes (point_label = -1)
            mapping_index = ((point_labels != -1).sum(1) > 0).to(torch.bool)
            if mapping_index.any():
                point_coords = point_coords[mapping_index]
                point_labels = point_labels[mapping_index]
                if prompt_class is not None:
                    prompt_class = prompt_class[mapping_index]
            else:
                if self.auto_freeze or (class_vector is None and patch_coords is None):
                    # if auto_freeze, point prompt must exist to allow loss backward
                    # in training, class_vector and point cannot both be None due to loss.backward()
                    mapping_index.fill_(True)
                else:
                    point_coords, point_labels = None, None

        if point_coords is None and class_vector is None and text_embeddings is None:
            return NINF_VALUE + torch.zeros([bs, 1, *image_size], device=device)

        if (
            self.image_embeddings is not None
            and kwargs.get("keep_cache", False)
            and class_vector is None
        ):
            out, out_auto = self.image_embeddings, None
        else:
            out, out_auto = self.image_encoder(
                input_images,
                with_point=point_coords is not None,
                with_label=class_vector is not None,
            )
        input_images = None

        # force releasing memories that set to None
        torch.cuda.empty_cache()
        if class_vector is not None:
            logits, _ = self.class_head(out_auto, class_vector)
            if point_coords is not None:
                point_logits = self.point_head(
                    out, point_coords, point_labels, class_vector=prompt_class
                )
                if patch_coords is None:
                    logits = self.gaussian_combine(
                        logits,
                        point_logits,
                        point_coords,
                        point_labels,
                        mapping_index,
                        radius,
                    )
                else:
                    # during validation use largest component
                    logits = self.connected_components_combine(
                        logits, point_logits, point_coords, point_labels, mapping_index
                    )
        elif text_embeddings is not None:
            logits = self.text_head(out, text_embeddings=text_embeddings)
        else:
            logits = NINF_VALUE + torch.zeros(
                [bs, 1, *image_size], device=device, dtype=out.dtype
            )
            logits[mapping_index] = self.point_head(
                out, point_coords, point_labels, class_vector=prompt_class
            )
            if prev_mask is not None and patch_coords is not None:
                logits = self.connected_components_combine(
                    prev_mask[patch_coords].transpose(1, 0).to(logits.device),
                    logits[mapping_index],
                    point_coords,
                    point_labels,
                    mapping_index,
                )

        if kwargs.get("keep_cache", False) and class_vector is None:
            self.image_embeddings = out.detach()
        return logits
