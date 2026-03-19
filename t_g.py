import os
import sys
import yaml
import glob
import time
import random
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Subset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader as GeoDataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from SPMM.SPMM_models import GINet
from utils import molT5_encoder
from dataset import smi_txt_dataset1

class Config:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.text_model_path = 'D:\project\FALD-Mol\molt5\models\snapshots\9e82786751c0ea421b9252f632e763934dfbe8c2'
        self.graph_config_path = "D:\project\FALD-Mol\MTSSMol\test_finetune.yaml"
        self.graph_model_path = 'D:\project\FALD-Mol\pretrain_spmm\checkpoint_epoch=5-v2.ckpt'
        self.data_paths = [
            'D:\project\FALD-Mol\data\chebi_20\train_parsed.txt',
            'D:\project\FALD-Mol\data\PCdes\train_parsed.txt'
        ]
        self.results_dir = "contrast_text_graph"

        self.text_embed_dim = 1024
        self.graph_embed_dim = 512
        self.projection_dim = 1024
        self.temperature = 0.07
        self.description_length = 256
        self.cross_attn_num_heads = 8
        self.cross_attn_dropout = 0.2
        self.projection_dropout = 0.2

        self.global_batch_size = 64
        self.learning_rate = 1e-5
        self.weight_decay = 3e-4
        self.epochs = 60
        self.log_every = 50
        self.ckpt_every = 5000
        self.num_workers = 4
        self.global_seed = 42
        self.min_learning_rate = 1e-6
        self.scheduler_T0 = 12000
        self.scheduler_Tmult = 2
        self.use_num_gpus = 4
        self.val_split_ratio = 0.15
        self.early_stop_patience = 15
        self.early_stop_min_delta = 0.001


class TextGraphCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.text2graph_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.graph2text_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.text_refine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_feat: torch.Tensor, graph_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        text_attn, _ = self.text2graph_attn(
            query=text_feat,
            key=graph_feat,
            value=graph_feat
        )
        graph_attn, _ = self.graph2text_attn(
            query=graph_feat,
            key=text_feat,
            value=text_feat
        )
        text_fused = self.layer_norm(self.dropout(text_attn) + text_feat)
        text_fused = self.text_refine(text_fused)
        graph_fused = self.layer_norm(self.dropout(graph_attn) + graph_feat)
        return text_fused, graph_fused


def load_text_encoder(config):
    text_tokenizer = T5Tokenizer.from_pretrained(
        config.text_model_path,
        model_max_length=config.description_length,
        padding_side='right'
    )
    text_encoder = T5ForConditionalGeneration.from_pretrained(config.text_model_path)
    del text_encoder.decoder

    for name, param in text_encoder.named_parameters():
        if any(f"encoder.block.{i}" in name for i in range(21, 27)):
            param.requires_grad = True
        else:
            param.requires_grad = False

    for module in text_encoder.encoder.block[21:27]:
        for submodule in module.modules():
            if isinstance(submodule, nn.Linear):
                nn.init.xavier_uniform_(submodule.weight, gain=0.6)
                if submodule.bias is not None:
                    nn.init.constant_(submodule.bias, 0.001)

    return text_encoder.to(config.device), text_tokenizer


def load_graph_encoder(config):
    graph_config = yaml.load(open(config.graph_config_path, "r"), Loader=yaml.FullLoader)
    graph_config['dataset']['task'] = 'regression'

    model = GINet(task=graph_config['dataset']['task'], **graph_config["model"])

    checkpoint = torch.load(config.graph_model_path, map_location='cpu')
    state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("out_lin", "pred_head") if "out_lin" in key else key
        if new_key in model.state_dict() and value.shape == model.state_dict()[new_key].shape:
            new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, strict=False)

    for param in model.parameters():
        param.requires_grad = False
    return model.to(config.device)


class ProjectionHead(nn.Module):
    def __init__(self, text_input_dim, graph_input_dim, output_dim=1024, dropout=0.2):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim)
        )
        for m in self.text_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.7)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.001)
        for m in self.graph_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.7)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.001)

    def forward_text(self, x):
        return self.text_proj(x)

    def forward_graph(self, x):
        return self.graph_proj(x)


class TextGraphContrastiveLearner(nn.Module):
    def __init__(self, text_encoder, text_tokenizer, graph_model, config):
        super().__init__()
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        self.graph_model = graph_model
        self.config = config

        self.proj_head = ProjectionHead(
            text_input_dim=config.text_embed_dim,
            graph_input_dim=config.graph_embed_dim,
            output_dim=config.projection_dim,
            dropout=config.projection_dropout
        )
        self.text_graph_attn = TextGraphCrossAttention(
            embed_dim=config.projection_dim,
            num_heads=config.cross_attn_num_heads,
            dropout=config.cross_attn_dropout
        )
        self.temperature = nn.Parameter(torch.tensor(config.temperature), requires_grad=True)
        nn.init.constant_(self.temperature, config.temperature)

        self.text_graph_fusion = nn.Sequential(
            nn.Linear(config.projection_dim, config.projection_dim),
            nn.GELU(),
            nn.LayerNorm(config.projection_dim),
            nn.Dropout(config.projection_dropout)
        )

    def get_text_embedding(self, descriptions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        processed_desc = []
        for d in descriptions:
            if random.random() < 0.97:
                processed_desc.append(d)
            else:
                synonyms = {
                    "six-membered ring": "6-membered ring",
                    "hydroxyl group": "-OH group",
                    "carboxyl group": "-COOH group",
                    "double bond": "C=C bond",
                    "polar solvent": "polar dissolvent"
                }
                augmented = d
                for k, v in synonyms.items():
                    if k in augmented:
                        augmented = augmented.replace(k, v)
                        break
                processed_desc.append(augmented)

        text_embed, pad_mask = molT5_encoder(
            processed_desc, self.text_encoder, self.text_tokenizer,
            self.config.description_length, self.config.device
        )
        return text_embed, pad_mask

    def get_graph_embedding(self, graph_data) -> torch.Tensor:
        with torch.no_grad():
            graph_embed, _ = self.graph_model(graph_data)
        graph_embed = graph_embed.unsqueeze(1).repeat(1, self.config.description_length, 1)
        return graph_embed

    def info_nce_loss(self, text_feat: torch.Tensor, graph_feat: torch.Tensor) -> torch.Tensor:
        text_avg = F.normalize(text_feat.mean(dim=1), dim=1)
        graph_avg = F.normalize(graph_feat.mean(dim=1), dim=1)
        sim_matrix = F.cosine_similarity(text_avg.unsqueeze(1), graph_avg.unsqueeze(0), dim=2)
        temperature = torch.clamp(self.temperature, min=0.05, max=0.2)
        sim_matrix = sim_matrix / temperature
        labels = torch.arange(text_avg.size(0), device=text_avg.device)
        return F.cross_entropy(sim_matrix, labels)

    def forward(self, batch: dict) -> tuple[torch.Tensor, float]:
        h_text, pad_mask = self.get_text_embedding(batch['descriptions'])
        h_graph = self.get_graph_embedding(batch['graph_data'])

        z_text = self.proj_head.forward_text(h_text)
        z_graph = self.proj_head.forward_graph(h_graph)
        z_text = F.normalize(z_text, p=2, dim=-1)
        z_graph = F.normalize(z_graph, p=2, dim=-1)

        z_text_fused, z_graph_fused = self.text_graph_attn(z_text, z_graph)

        z_text_final = self.text_graph_fusion(z_text_fused)
        z_text_final = F.normalize(z_text_final, p=2, dim=-1)

        contrast_loss = self.info_nce_loss(z_text_final, z_graph_fused)
        total_loss = contrast_loss

        text_graph_sim = F.cosine_similarity(
            z_text_final.mean(dim=1), z_graph_fused.mean(dim=1), dim=1
        ).mean().item()

        return total_loss, text_graph_sim

    def encode_text(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            h_text, pad_mask = self.get_text_embedding(texts)
            h_text_norm = F.normalize(h_text, p=2, dim=-1)

            z_text = self.proj_head.forward_text(h_text)
            z_text = F.normalize(z_text, p=2, dim=-1)

            z_graph_sim = z_text
            z_text_fused, _ = self.text_graph_attn(z_text, z_graph_sim)
            z_text_final = self.text_graph_fusion(z_text_fused)
            z_text_final = F.normalize(z_text_final, p=2, dim=-1)

            final_embeds = 0.9 * h_text_norm + 0.1 * z_text_final

        return final_embeds, pad_mask


def create_logger(logging_dir: str, rank: int) -> logging.Logger:
    logger = logging.getLogger(f"Rank_{rank}")
    logger.setLevel(logging.INFO)
    if rank == 0 and logging_dir is not None:
        file_handler = logging.FileHandler(f"{logging_dir}/train.log")
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('[\033[34m%(asctime)s\033[0m] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    else:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('[\033[34m%(asctime)s\033[0m] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        return self.early_stop


def split_dataset(dataset, val_ratio=0.15, seed=42):
    from sklearn.model_selection import train_test_split
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=val_ratio, random_state=seed, shuffle=True
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset


def train(rank: int, world_size: int, config: Config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    set_global_seed(config.global_seed + rank)

    exp_dir = None
    ckpt_dir = None
    logger = None
    if rank == 0:
        os.makedirs(config.results_dir, exist_ok=True)
        exp_idx = len(glob.glob(f"{config.results_dir}/*"))
        exp_dir = f"{config.results_dir}/{exp_idx:03d}-text_graph_implicit_align"
        ckpt_dir = f"{exp_dir}/checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        logger = create_logger(exp_dir, rank)
        logger.info(f"=== Experiment Started ===")
        logger.info(f"Experiment directory: {exp_dir}")
        logger.info(f"Key configurations:")
        logger.info(f"  - Unfreezed MolT5 layers: 6 (21-26)")
        logger.info(f"  - Fusion weight: 90% text + 10% graph")
        logger.info(f"  - LR: {config.learning_rate}, Weight decay: {config.weight_decay}")
        logger.info(f"Training with {world_size} GPUs")
    else:
        logger = create_logger(None, rank)

    dist.barrier()

    if rank != 0:
        exp_dirs = sorted(glob.glob(f"{config.results_dir}/*"))
        exp_dir = exp_dirs[-1] if exp_dirs else None
        ckpt_dir = f"{exp_dir}/checkpoints" if exp_dir else None
        logger.info(f"Rank {rank} loaded experiment directory: {exp_dir}")

    logger.info("Loading text encoder (MolT5) and graph encoder (SPMM)...")
    text_encoder, text_tokenizer = load_text_encoder(config)
    graph_model = load_graph_encoder(config)

    contrast_model = TextGraphContrastiveLearner(
        text_encoder=text_encoder,
        text_tokenizer=text_tokenizer,
        graph_model=graph_model,
        config=config
    ).to(rank)
    contrast_model = torch.nn.parallel.DistributedDataParallel(
        contrast_model,
        device_ids=[rank],
        find_unused_parameters=True
    )

    logger.info("Loading and splitting dataset...")
    full_dataset = smi_txt_dataset1(
        data_path=config.data_paths,
        shuffle=True,
        unconditional=False,
        raw_description=False
    )
    logger.info(f"Full dataset size: {len(full_dataset)}")

    if rank == 0:
        train_dataset, val_dataset = split_dataset(
            full_dataset, val_ratio=config.val_split_ratio, seed=config.global_seed
        )
        with open(f"{exp_dir}/train_val_indices.pkl", 'wb') as f:
            pickle.dump((train_dataset.indices, val_dataset.indices), f)
        logger.info(f"Train set: {len(train_dataset)}, Val set: {len(val_dataset)}")
    else:
        train_dataset = None
        val_dataset = None

    dist.barrier()

    with open(f"{exp_dir}/train_val_indices.pkl", 'rb') as f:
        train_indices, val_indices = pickle.load(f)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config.global_seed
    )
    local_batch_size = config.global_batch_size // world_size
    train_dataloader = GeoDataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        follow_batch=["x"]
    )

    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    val_dataloader = GeoDataLoader(
        val_dataset,
        batch_size=local_batch_size,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        follow_batch=["x"]
    )

    optimizer = AdamW(
        contrast_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=config.scheduler_T0, T_mult=config.scheduler_Tmult, eta_min=config.min_learning_rate
    )

    early_stopper = EarlyStopping(
        patience=config.early_stop_patience,
        min_delta=config.early_stop_min_delta
    ) if rank == 0 else None
    best_val_sim = 0.0 if rank == 0 else None

    logger.info("Starting implicit alignment training...")
    contrast_model.train()
    train_steps = 0
    log_steps = 0
    running_loss = 0.0
    start_time = time.time()

    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)
        logger.info(f"\n===== Epoch {epoch + 1}/{config.epochs} (Train) =====")
        for batch_data in train_dataloader:
            _, descriptions, _, _, graph_data = batch_data
            batch = {
                'descriptions': descriptions,
                'graph_data': graph_data.to(rank)
            }

            optimizer.zero_grad()
            total_loss, text_graph_sim = contrast_model(batch)
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(contrast_model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            running_loss += total_loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % config.log_every == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)

                avg_loss = torch.tensor(running_loss / log_steps, device=rank)
                avg_sim = torch.tensor(text_graph_sim, device=rank)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_sim, op=dist.ReduceOp.SUM)
                avg_loss /= world_size
                avg_sim /= world_size

                current_lr = optimizer.param_groups[0]['lr']
                if rank == 0:
                    logger.info(
                        f"Step {train_steps:07d} | Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.8f} | Steps/sec: {steps_per_sec:.2f} | "
                        f"Text-graph sim: {avg_sim:.4f} | Best val sim: {best_val_sim:.4f}"
                    )

                running_loss = 0.0
                log_steps = 0
                start_time = time.time()

            if train_steps % config.ckpt_every == 0 and rank == 0:
                ckpt_path = f"{ckpt_dir}/step_{train_steps:07d}.pt"
                torch.save({
                    "model": contrast_model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "step": train_steps,
                    "best_val_sim": best_val_sim
                }, ckpt_path)
                logger.info(f"Saved checkpoint to: {ckpt_path}")
            if train_steps % config.ckpt_every == 0:
                dist.barrier()

        logger.info(f"===== Epoch {epoch + 1}/{config.epochs} (Validation) =====")
        contrast_model.eval()
        val_loss_sum = 0.0
        val_sim_sum = 0.0
        val_batch_count = 0

        with torch.no_grad():
            for batch_data in val_dataloader:
                _, descriptions, _, _, graph_data = batch_data
                batch = {
                    'descriptions': descriptions,
                    'graph_data': graph_data.to(rank)
                }
                val_loss, val_sim = contrast_model(batch)
                val_loss_sum += val_loss.item()
                val_sim_sum += val_sim
                val_batch_count += 1

        avg_val_loss = torch.tensor(val_loss_sum / val_batch_count, device=rank)
        avg_val_sim = torch.tensor(val_sim_sum / val_batch_count, device=rank)
        dist.all_reduce(avg_val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_val_sim, op=dist.ReduceOp.SUM)
        avg_val_loss /= world_size
        avg_val_sim /= world_size

        if rank == 0:
            logger.info(
                f"Validation | Loss: {avg_val_loss:.4f} | "
                f"Text-graph sim: {avg_val_sim:.4f} | "
                f"Best sim: {best_val_sim:.4f} | "
                f"Early stop counter: {early_stopper.counter}/{early_stopper.patience}"
            )

            if avg_val_sim > best_val_sim + config.early_stop_min_delta:
                best_val_sim = avg_val_sim
                best_ckpt_path = f"{ckpt_dir}/best_val_model.pt"
                torch.save({
                    "model": contrast_model.module.state_dict(),
                    "epoch": epoch,
                    "step": train_steps,
                    "best_val_sim": best_val_sim,
                    "config": config.__dict__
                }, best_ckpt_path)
                logger.info(f"Updated best model (sim: {best_val_sim:.4f}) to: {best_ckpt_path}")

            if early_stopper(avg_val_sim):
                logger.info(f"Early stopping triggered! No improvement for {config.early_stop_patience} epochs")
                logger.info(f"Final best validation similarity: {best_val_sim:.4f}")
                dist.destroy_process_group()
                return

        contrast_model.train()

    if rank == 0:
        logger.info(f"=== Training finished ===")
        logger.info(f"Final best text-graph similarity: {best_val_sim:.4f}")
        logger.info(f"Model saved to: {ckpt_dir}")
    dist.destroy_process_group()


def encode_text_with_implicit_graph(ckpt_path: str, texts: list[str], use_gpu: bool = True) -> tuple[
    torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    config = Config()
    for key, val in checkpoint.get('config', {}).items():
        setattr(config, key, val)

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    config.device = device

    text_encoder, text_tokenizer = load_text_encoder(config)
    graph_model = load_graph_encoder(config)

    model = TextGraphContrastiveLearner(
        text_encoder=text_encoder,
        text_tokenizer=text_tokenizer,
        graph_model=graph_model,
        config=config
    ).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()

    with torch.no_grad():
        final_embeds, pad_mask = model.encode_text(texts)

    print(f"Inference done: embed shape {final_embeds.shape}, device {device}")
    return final_embeds, pad_mask


def main():
    config = Config()
    set_global_seed(config.global_seed)

    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    if available_gpus < config.use_num_gpus:
        raise ValueError(f"Need {config.use_num_gpus} GPUs, only {available_gpus} available")

    required_paths = [config.text_model_path, config.graph_model_path, config.graph_config_path]
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")

    world_size = config.use_num_gpus
    print(f"\n=== Starting training ===")
    print(f"Experiment: {config.results_dir}")
    print(f"Optimizations: MolT5 6 unfreezed layers + normalization + 90% text +10% graph fusion")
    print(f"Training with {world_size} GPUs...")
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.spawn(
        train, args=(world_size, config), nprocs=world_size, join=True
    )


if __name__ == "__main__":
    main()