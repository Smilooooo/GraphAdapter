"""
GraphCLIP v2 Trainer - Multi-backbone support

This trainer extends GraphCLIP_v1 to support multiple CLIP-like foundation models:
- OpenAI CLIP (RN50, ViT-B/16, etc.)
- BiomedCLIP
- PLIP
- QuiltNet
- CONCH

Usage:
    Set in config:
        MODEL:
          BACKBONE:
            NAME: "biomedclip"  # or "plip", "quiltnet", etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import math

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.metrics import compute_accuracy

from clip import clip

from .model_registry import load_clip_model, get_feature_dim, MODEL_CONFIGS
from .text_encoders import create_text_encoder, UnifiedTextEncoder
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT


# =============================================================================
# Templates
# =============================================================================

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    # Pathology datasets
    "LungHist700": "a histopathological image of {}.",
    "BRACS": "a histopathological image of {}.",
    "ICIAR2018": "a histopathological image of {}.",
}


# =============================================================================
# Graph Utilities (unchanged from v1)
# =============================================================================

def graph_norm_ours(A, batch=False, self_loop=True, symmetric=True):
    """Graph normalization: D^(-1/2) * A * D^(-1/2) or D^(-1) * A"""
    d = A.sum(-1)
    if symmetric:
        d = torch.pow(d, -0.5)
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A).bmm(D)
        else:
            D = torch.diag(d)
            norm_A = D.mm(A).mm(D)
    else:
        d = torch.pow(d, -1)
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A)
        else:
            D = torch.diag(d)
            norm_A = D.mm(A)
    return norm_A


def cal_similarity(x, p=2, dim=1):
    """Calculate similarity matrix."""
    x = F.normalize(x, p=p, dim=dim)
    return torch.mm(x, x.transpose(0, 1))


def cal_edge_emb(x, p=2, dim=1):
    """Calculate edge embeddings via similarity."""
    x = F.normalize(x, p=p, dim=dim)
    x = x.transpose(1, 2)
    A = torch.bmm(x, x.transpose(1, 2))
    return A


# =============================================================================
# Graph Convolution (modified to use configurable hidden_dim)
# =============================================================================

class GraphConvolution(nn.Module):
    def __init__(self, hidden_dim, name=None, device=None, class_num=None, 
                 sparse_inputs=False, act=nn.Tanh, bias=True, dropout=0.0):
        super().__init__()
        self.act = nn.Tanh()
        self.device = device
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        self.bias = bias
        self.hidden_dim = hidden_dim  # Now uses passed value, not hardcoded
        self.class_num = class_num
        self.gcn_weights = nn.Parameter(torch.ones(self.hidden_dim, self.hidden_dim))
        if self.bias:
            self.gcn_bias = nn.Parameter(torch.zeros(class_num, self.hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.gcn_weights.size(1))
        self.gcn_weights.data.uniform_(-stdv, stdv)

    def forward(self, feat, adj):
        x = feat
        node_size = adj.size()[1]
        adj = torch.clip(adj, min=0.0)
        I = torch.eye(node_size, device='cuda').unsqueeze(dim=0).to(self.device)
        adj = adj + I
        adj = graph_norm_ours(adj, batch=True, self_loop=True, symmetric=True)
        x = x.transpose(1, 2)
        pre_sup = torch.matmul(x, self.gcn_weights)
        output = torch.matmul(adj, pre_sup)

        if self.bias:
            output += self.gcn_bias.unsqueeze(1)
        if self.act is not None:
            return self.act(output[:, 0, :])
        else:
            return output[:, 0, :]


# =============================================================================
# Graph Learner (now with configurable feature_dim)
# =============================================================================

class GraphLearner(nn.Module):
    def __init__(self, cfg, classnames, feature_dim, base_text_features, base_img_features, dtype):
        super().__init__()
        self.device = dtype
        self.alpha = 0.1
        print(">> DCT scale factor: ", self.alpha)
        self.register_buffer("base_text_features", base_text_features)
        self.register_buffer("base_img_features", base_img_features)
        self.alpha_it = 0.7
        self.beta_it = cfg.TRAINER.GRAPHADAPTER.BETA
        self.node_num = 1
        self.hidden_dim = feature_dim  # Use model's feature dimension
        
        self.GCN_tt = GraphConvolution(
            self.hidden_dim, name='metagraph', device=self.device, 
            class_num=base_text_features.size()[0]
        )
        self.GCN_it = GraphConvolution(
            self.hidden_dim, name='metagraph', device=self.device, 
            class_num=base_text_features.size()[0]
        )

    def reset_parameters(self):
        for i in range(self.node_num):
            stdv = 1. / math.sqrt(self.graph_node[i].size(0))
            self.graph_node[i].data.uniform_(-stdv, stdv)

    def forward(self, img_feature):
        with torch.no_grad():
            node_cluster_t = self.base_text_features.view(
                1, self.base_text_features.size()[0]//4, 4, self.base_text_features.size()[1]
            )
            node_cluster_i = self.base_img_features.view(
                1, self.base_img_features.size()[0]//4, 4, self.base_img_features.size()[1]
            )
           
        graph_o_t_all = []
            
        for index in range(4):
            with torch.no_grad():
                inputs_text = self.base_text_features.unsqueeze(dim=1)
                inputs_img = img_feature.unsqueeze(dim=1)
                node_cluster_tt = node_cluster_t[:, :, index, :].repeat(inputs_text.size()[0], 1, 1)
                node_cluster_it = node_cluster_i[:, :, index, :].repeat(inputs_text.size()[0], 1, 1)
                feat_tt = torch.cat([inputs_text, node_cluster_tt], dim=1)
                feat_it = torch.cat([inputs_text, node_cluster_it], dim=1)
                feat_tt = feat_tt.transpose(1, 2).detach()
                feat_it = feat_it.transpose(1, 2).detach()
                edge_tt = cal_edge_emb(feat_tt).detach()
                edge_it = cal_edge_emb(feat_it).detach()
            
            graph_o_tt = self.GCN_tt(feat_tt, edge_tt)
            graph_o_it = self.GCN_it(feat_it, edge_it)
            graph_o_t = (graph_o_tt) * self.alpha_it + (1 - self.alpha_it) * graph_o_it
            graph_o_t_all.append(graph_o_t)
        
        graph_o_t = torch.stack(graph_o_t_all, dim=0).mean(dim=0)
    
        return self.beta_it * self.base_text_features + (1 - self.beta_it) * graph_o_t.squeeze(), img_feature


# =============================================================================
# Feature Extraction Functions
# =============================================================================

def _get_base_image_features(cfg, classnames, clip_model, img_encoder, train_loader_x, feature_dim):
    """Extract base image features from training data."""
    device = next(img_encoder.parameters()).device
    dtype = clip_model.dtype
    
    if dtype == torch.float16:
        img_encoder = img_encoder.cuda()
    
    with torch.no_grad():
        img_feature = []
        labels = []
        for epch in range(10):
            for batch_idx, batch in enumerate(train_loader_x):
                image = batch["img"]
                label = batch["label"]
                image = image.cuda()
                label = label.cuda()
                image_features = img_encoder(image.type(dtype)).detach()
                img_feature.append(image_features)
                labels.append(label)
        
        img_feature_list = torch.cat(img_feature, dim=0)
        label_list = torch.cat(labels, dim=0)
        sorted_labels, indices = torch.sort(label_list)
        label_len = len(sorted_labels) // (sorted_labels[-1] + 1)
        img_feature_list_all = torch.index_select(img_feature_list, 0, indices)
        b, c = img_feature_list_all.size()
        img_feature_list_all = img_feature_list_all.view(b // label_len, label_len, -1).mean(dim=1)
        img_encoder = img_encoder.to(device)

    return img_feature_list_all.to(device)


def _get_base_text_features(cfg, classnames, clip_model, text_encoder, framework):
    """Extract base text features using prompts."""
    device = next(text_encoder.parameters()).device
    dtype = clip_model.dtype
    
    if dtype == torch.float16:
        text_encoder = text_encoder.cuda()
    
    dataset = cfg.DATASET.NAME

    if dataset == "ImageNet":
        TEMPLATES = IMAGENET_TEMPLATES_SELECT
    else:
        TEMPLATES = []
    
    if dataset in CUSTOM_TEMPLATES:
        TEMPLATES += [CUSTOM_TEMPLATES[dataset]]
    else:
        TEMPLATES += ["a photo of a {}."]

    with torch.no_grad():
        text_embeddings = []
        for text in classnames:
            prompts = [template.format(text) for template in TEMPLATES]
            
            # Tokenize based on framework
            if framework == "transformers":
                # For transformers, encode directly
                if hasattr(text_encoder, 'encode_text_direct'):
                    features = text_encoder.encode_text_direct(prompts, device)
                    text_embeddings.append(features.mean(0, keepdim=True))
                    continue
                else:
                    from transformers import AutoTokenizer
                    # Fallback
                    tokens = clip.tokenize(prompts).to(device)
            elif framework in ("open_clip", "conch"):
                import open_clip
                tokens = open_clip.tokenize(prompts).to(device)
            else:
                tokens = clip.tokenize(prompts).to(device)
            
            embeddings = clip_model.token_embedding(tokens).type(dtype)
            if dtype == torch.float16:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))
            else:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))
    
    text_embeddings = torch.stack(text_embeddings).mean(1)
    text_encoder = text_encoder.to(device)
    return text_embeddings.to(device)


# =============================================================================
# Custom CLIP Model
# =============================================================================

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, train_loader_x, framework, feature_dim):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.framework = framework
        self.feature_dim = feature_dim
        
        # Create text encoder using unified interface
        text_encoder = create_text_encoder(
            clip_model._model if hasattr(clip_model, '_model') else clip_model,
            framework,
            feature_dim,
            tokenizer=clip_model._tokenizer if hasattr(clip_model, '_tokenizer') else None
        )
        img_encoder = self.image_encoder
        
        # Get base features
        base_text_features = _get_base_text_features(
            cfg, classnames, clip_model, text_encoder, framework
        )
        base_img_features = _get_base_image_features(
            cfg, classnames, clip_model, img_encoder, train_loader_x, feature_dim
        )

        self.graph_learner = GraphLearner(
            cfg, classnames, feature_dim, base_text_features, base_img_features, self.dtype
        )

    def forward(self, image):
        try:
            image_features = self.image_encoder(image.type(self.dtype)).detach()
        except:
            image_features = self.image_encoder(image.float()).detach()

        text_features, image_features = self.graph_learner(image_features)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


# =============================================================================
# Trainer
# =============================================================================

@TRAINER_REGISTRY.register()
class GraphCLIP_v2(TrainerX):
    """
    GraphCLIP v2 Trainer with multi-backbone support.
    
    Supports:
        - OpenAI CLIP: RN50, RN101, ViT-B/32, ViT-B/16
        - BiomedCLIP: biomedclip
        - PLIP: plip
        - QuiltNet: quiltnet
        - CONCH: conch
    """
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        backbone_name = cfg.MODEL.BACKBONE.NAME

        # Determine framework
        if backbone_name in MODEL_CONFIGS:
            framework = MODEL_CONFIGS[backbone_name]["framework"]
            feature_dim = MODEL_CONFIGS[backbone_name]["feature_dim"]
        else:
            framework = getattr(cfg.MODEL.BACKBONE, "FRAMEWORK", "openai_clip")
            feature_dim = getattr(cfg.MODEL.BACKBONE, "FEATURE_DIM", 512)

        print(f"Loading model (backbone: {backbone_name})")
        print(f"  Framework: {framework}")
        print(f"  Feature dim: {feature_dim}")
        
        # Load model using registry
        if framework == "openai_clip":
            # Use original loading for OpenAI CLIP (for compatibility)
            from clip import clip as clip_module
            url = clip_module._MODELS[backbone_name]
            model_path = clip_module._download(url)
            try:
                clip_model = torch.jit.load(model_path, map_location="cpu").eval()
                state_dict = None
            except RuntimeError:
                state_dict = torch.load(model_path, map_location="cpu")
            clip_model = clip_module.build_model(state_dict or clip_model.state_dict())
        else:
            # Use the model registry for other models
            clip_model = load_clip_model(cfg, device="cpu")
        
        clip_model.to(self.device)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(
            cfg, classnames, clip_model, self.train_loader_x, framework, feature_dim
        ).cuda()

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "graph_learner" not in name:
                param.requires_grad_(False)

        for param in self.model.graph_learner.parameters():
            param.requires_grad_(True)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.graph_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model.float()
        
        self.optim = build_optimizer(model=self.model.graph_learner, optim_cfg=cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("graph_learner", self.model.graph_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC
    
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
