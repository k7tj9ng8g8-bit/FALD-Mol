import random
import argparse
import logging
import os
from glob import glob
from time import time
from collections import OrderedDict
from copy import deepcopy
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from models import DiT_models
from diffusion import create_diffusion
from train_autoencoder import vae_with_property
from utils import  regexTokenizer, AE_SMILES_encoder
from dataset import smi_txt_dataset
from download import find_model
from t_g import Config as TextGraphConfig, TextGraphContrastiveLearner
from t_g import load_text_encoder, load_graph_encoder

class TextGraphEncoderSingleton:
    _instance = None
    _initialized = False

    def __new__(cls, ckpt_path, device):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, ckpt_path, device):
        if not TextGraphEncoderSingleton._initialized:
            self.config = TextGraphConfig()
            self._load_encoder(ckpt_path, device)
            TextGraphEncoderSingleton._initialized = True

    def _load_encoder(self, ckpt_path, device):
        self.text_encoder, self.text_tokenizer = load_text_encoder(self.config)
        self.graph_model = load_graph_encoder(self.config)
        self.model = TextGraphContrastiveLearner(
            text_encoder=self.text_encoder,
            text_tokenizer=self.text_tokenizer,
            graph_model=self.graph_model,
            config=self.config
        ).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.model.eval()

    def encode(self, texts):
        with torch.no_grad():
            embeds, pad_mask = self.model.encode_text(texts)
        return embeds, pad_mask

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    dist.destroy_process_group()

def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def main(args):
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '8000'

    assert torch.cuda.is_available(), "Training requires at least one GPU."

    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
    else:
        logger = create_logger(None)

    text_graph_encoder = TextGraphEncoderSingleton(
        ckpt_path=args.t_g_ckpt,
        device=torch.device(f"cuda:{device}")
    )

    latent_size = 127
    in_channels = 64
    cross_attn = 768
    if args.text_encoder_name == 'llama2':
        condition_dim = 4096
    elif args.text_encoder_name == 'molt5':
        condition_dim = 1024

    model = DiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=args.num_classes,
        cross_attn=cross_attn,
        condition_dim=condition_dim
    )

    if args.ckpt:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        msg = model.load_state_dict(state_dict, strict=True)

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    diffusion = create_diffusion(timestep_respacing="")

    ae_config = {
        'bert_config_decoder': './config_decoder.json',
        'bert_config_encoder': './config_encoder.json',
        'bert_config_property': 'D:\project\FALD-Mol\SPMM\config_bert_property.json',
        'embed_dim': 256,
        'property_width': 768,
        'schedular': {'warmup_epochs': 0},
        'mlm_probability': 0.15
    }
    tokenizer = regexTokenizer(vocab_path='./vocab_bpe_300_sc.txt', max_len=127)
    ae_model = vae_with_property(
        config=ae_config,
        no_train=True,
        tokenizer=tokenizer,
        use_linear=True,
        num_properties=53
    )

    if args.vae:
        checkpoint = torch.load(args.vae, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['state_dict']

        filtered_state_dict = {}
        for key, val in state_dict.items():
            if 'ema' not in key and 'queue' not in key:
                filtered_state_dict[key] = val

        msg = ae_model.load_state_dict(filtered_state_dict, strict=False)

    for param in ae_model.parameters():
        param.requires_grad = False
    if hasattr(ae_model, 'text_encoder'):
        del ae_model.text_encoder
    ae_model = ae_model.to(device)
    ae_model.eval()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    dataset = smi_txt_dataset([
        'D:\project\FALD-Mol\data\chebi_20\train_parsed.txt',
        'D:\project\FALD-Mol\data\PCdes\train_parsed.txt',
        'D:\project\FALD-Mol\data\SMILES_Text_final.txt',
        'D:\project\FALD-Mol\data\unlabeled_200k.txt',
    ], data_length=None, shuffle=True, unconditional=False, raw_description=True)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = GeoDataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        for x, y, prop in loader:
            with torch.no_grad():
                x = AE_SMILES_encoder(x, ae_model, prop)
                y = [d if random.random() < 0.95 else dataset.null_text for d in y]
                biot5_embed, pad_mask = text_graph_encoder.encode(y)
                y = biot5_embed.detach().to(device)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y.type(torch.float32), pad_mask=pad_mask.bool())
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            opt.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            update_ema(ema, model.module)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "epoch": epoch,
                        "step": train_steps
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="FALD")
    parser.add_argument("--text-encoder-name", type=str, default="molt5", choices=["molt5", "llama2"])
    parser.add_argument("--description-length", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=32 * 6)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, default="D:\project\FALD-Mol\Pretrain_autoencoder\epoch_epoch=09_loss_loss_mlm=0.1303.ckpt")
    parser.add_argument("--t_g_ckpt", type=str, default="D:\project\FALD-Mol\contrast_text_graph\have-pretrain\checkpoints\best_val_model.pt")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50000)
    args = parser.parse_args()

    main(args)