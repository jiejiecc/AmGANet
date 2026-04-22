import sys
import os
from typing import Dict, Tuple

# 添加 my_open_clip 所在的父目录到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import torch
import torch.nn as nn
from torch.nn import functional as F
# from my_open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer


class TextEncoder(nn.Module):
    def __init__(self, biomedclip_model):
        super().__init__()
        self.model = biomedclip_model
        self.dtype = biomedclip_model.text.transformer.dtype

    def forward(self, prompts, tokenized_prompts):
        x = self.model.encode_text(prompts, True, tokenized_prompts)
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, biomedclip_model, CTP, tokenizer, device):
        super().__init__()

        n_cls = len(classnames)
        n_ctx = 4
        ctx_init = "a photo of a"
        dtype = biomedclip_model.text.transformer.dtype
        ctx_dim = 768
        clip_imsize = 224
        cfg_imsize = 224
        self.tokenizer = tokenizer
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx == 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = self.tokenizer(ctx_init)
            with torch.no_grad():
                embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(self.tokenizer(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts])

        with torch.no_grad():
            embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = CTP

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                    suffix,
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        return prompts



from .prompt_templates import num_Covid19, num_busi_kvasir, location_busi_kvasir,location_Covid19





class VisualGatedAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.txt_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True)
            )
        self.out = nn.Linear(dim, dim)

    def forward(self, text_feats, img_feats):
        txt = self.txt_proj(text_feats).unsqueeze(0)      # (1,N,D)
        img = img_feats.unsqueeze(1)       # (B,1,D)
        gate = torch.sigmoid(self.out(torch.tanh(txt + img)))  # (B,N,D)
        return text_feats.unsqueeze(0) * gate


class FeatureAdapter(nn.Module):

    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.scale = nn.Parameter(torch.zeros(1))  # 

    def forward(self, x):
        delta = self.fc2(self.act(self.fc1(x)))
        return x + self.scale * delta


class CustomCLIP(nn.Module):
    """
    1) num branch: Retain softmax probabilities; Output hard num features + expected_count.
    2) location branch: Use sigmoid for multi-label classification; Output top-k multiple location tokens,
    and generate a mask using the prediction results from the quantity branch for downstream segmenters to use.
    """
    def __init__(self, cfg, task_name, biomedclip_model, tokenizer, device, max_location_tokens=3):
        super().__init__()
        self.task_name = task_name
        if task_name == "Covid19_X" or task_name == "Covid19_CT":
            self.prompt_learner_loc = PromptLearner(
                cfg, location_Covid19, biomedclip_model, CTP="end", tokenizer=tokenizer, device=device
            )
            self.prompt_learner_num = PromptLearner(
                cfg, num_Covid19, biomedclip_model, CTP="end", tokenizer=tokenizer, device=device
            )
        else:
            self.prompt_learner_loc = PromptLearner(
                cfg, location_busi_kvasir, biomedclip_model, CTP="end", tokenizer=tokenizer, device=device
            )
            self.prompt_learner_num = PromptLearner(
                cfg, num_busi_kvasir, biomedclip_model, CTP="end", tokenizer=tokenizer, device=device
            )

        self.image_encoder = biomedclip_model.visual
        self.text_encoder = TextEncoder(biomedclip_model)
        self.logit_scale = biomedclip_model.logit_scale
        self.dtype = biomedclip_model.text.transformer.dtype

        self.image_adapter = FeatureAdapter(dim=512, hidden_dim=512).to(device)
        self.text_adapter = VisualGatedAdapter(dim=512).to(device)

        # The quantity category is mapped to "target number prior".
        if task_name == "Covid19_X" or task_name == "Covid19_CT":
            self.register_buffer(
                "count_values",
                torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
            )

        else:
            self.register_buffer(
                "count_values",
                torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
            )   
        self.max_location_tokens = max_location_tokens

    def encode_texts(self, prompt_learner):
        prompts = prompt_learner()
        tokenized_prompts = prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        return text_features

    def _condition_text_features(self, raw_text_features, image_features):
        if self.text_adapter is not None:
            text_features = self.text_adapter(raw_text_features, image_features)
        else:
            batch_size = image_features.shape[0]
            text_features = raw_text_features.unsqueeze(0).expand(batch_size, -1, -1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def _compute_logits(self, image_features, text_features):
        logit_scale = self.logit_scale.exp()
        img_unsqueezed = image_features.unsqueeze(1)      # (B, 1, D)
        text_transposed = text_features.transpose(1, 2)  # (B, D, N)
        logits = logit_scale * torch.bmm(img_unsqueezed, text_transposed).squeeze(1)  # (B, N)
        return logits

    def _aggregate_num(self, logits_num, text_features_num):

        batch_size = logits_num.shape[0]
        batch_indices = torch.arange(batch_size, device=logits_num.device)

        num_probs = F.softmax(logits_num, dim=-1)  # (B, 3)
        num_indices = torch.argmax(num_probs, dim=-1)  # (B,)

        num_feat = text_features_num[batch_indices, num_indices, :]  # (B, D)
        expected_count = (num_probs * self.count_values.to(logits_num.device)).sum(dim=-1)  # (B,)

        # Use discrete prediction to determine the number of position tokens
        discrete_count = self.count_values.to(logits_num.device)[num_indices].long()
        if self.task_name == "Covid19_X" or self.task_name == "Covid19_CT":
            discrete_count = torch.clamp(discrete_count, min=1, max=self.max_location_tokens)
        else:
            discrete_count = torch.clamp(discrete_count, min=0, max=self.max_location_tokens)

        return {
            "logits": logits_num,
            "probs": num_probs,
            "pred_indices": num_indices,
            "pred_count": discrete_count,
            "expected_count": expected_count,
            "features": num_feat.unsqueeze(1),   # (B, 1, D)
        }

    def _aggregate_location(self, logits_loc, text_features_loc, pred_count):
        """
        Position Branch:
        - Use sigmoid to calculate multi-label probabilities
        - Select the top-k position tokens (k = max_location_tokens)
        - Then generate an effective token mask using pred_count for the segmenter to use
        """
        batch_size, num_locations = logits_loc.shape
        batch_indices = torch.arange(batch_size, device=logits_loc.device)

        loc_probs = torch.sigmoid(logits_loc)  # (B, N_loc)
        current_k = min(self.max_location_tokens, num_locations)
        topk_probs, topk_indices = torch.topk(loc_probs, k=current_k, dim=-1)  # (B, K)

        selected_loc_feats = text_features_loc[batch_indices.unsqueeze(1), topk_indices, :]  # (B, K, D)

        
        # pred_count: (B,), values in [1, K]
        token_positions = torch.arange(current_k, device=logits_loc.device).unsqueeze(0)  # (1, K)
        feature_mask = token_positions < pred_count.unsqueeze(1)  # (B, K), bool

        masked_loc_feats = selected_loc_feats * feature_mask.unsqueeze(-1).float()

        # pooled feature 
        denom = feature_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        pooled_loc_feat = masked_loc_feats.sum(dim=1) / denom  # (B, D)

        return {
            "logits": logits_loc,
            "probs": loc_probs,
            "topk_probs": topk_probs,
            "topk_indices": topk_indices,
            "feature_mask": feature_mask,
            "features": masked_loc_feats,         # (B, K, D)
            "pooled_feature": pooled_loc_feat.unsqueeze(1),  # (B, 1, D)
        }

    def forward(self, images):
        # 1. image encoding
        image_features = self.image_encoder(images.type(self.dtype))
        if self.image_adapter is not None:
            image_features = self.image_adapter(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 2. num branch
        raw_num_text_features = self.encode_texts(self.prompt_learner_num)         # (N_num, D)
        text_features_num = self._condition_text_features(raw_num_text_features, image_features)  # (B, N_num, D)
        logits_num = self._compute_logits(image_features, text_features_num)       # (B, N_num)
        num_results = self._aggregate_num(logits_num, text_features_num)

        # 3. location branch
        raw_loc_text_features = self.encode_texts(self.prompt_learner_loc)         # (N_loc, D)
        text_features_loc = self._condition_text_features(raw_loc_text_features, image_features)  # (B, N_loc, D)
        logits_loc = self._compute_logits(image_features, text_features_loc)       # (B, N_loc)
        loc_results = self._aggregate_location(
            logits_loc=logits_loc,
            text_features_loc=text_features_loc,
            pred_count=num_results["pred_count"],
        )

        return {
            "num": num_results,
            "location": loc_results,
        }
