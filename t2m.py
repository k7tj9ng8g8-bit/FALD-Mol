import os
import pandas as pd
import torch
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from tqdm import tqdm
import argparse
from einops import repeat
from train_autoencoder import vae_with_property
from utils import AE_SMILES_decoder, regexTokenizer
import time
from dataset import smi_txt_dataset
from torch.utils.data import DataLoader
from metrics import molfinger_evaluate, mol_evaluate
from rdkit import Chem, logger
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
def main(args):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)
    device = torch.device("cuda:6")
    rank = 0
    seed = torch.seed()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, device={device}")

    if args.ckpt is None or args.t_g_ckpt is None:
        raise ValueError("Must specify DiT checkpoint (--ckpt) and text-graph checkpoint (--t_g_ckpt)")

    if rank == 0:
        print("Initializing text-graph cross-modal encoder...")
    text_graph_encoder = TextGraphEncoderSingleton(
        ckpt_path=args.t_g_ckpt,
        device=device
    )
    if rank == 0:
        print("Text-graph encoder initialization finished")

    latent_size = 127
    in_channels = 64
    cross_attn = 768
    condition_dim = 1024

    model = DiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        cross_attn=cross_attn,
        condition_dim=condition_dim,
    ).to(device)

    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    msg = model.load_state_dict(state_dict, strict=False)
    if rank == 0:
        print(f'DiT loaded from {ckpt_path}, msg: {msg}')
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))

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
    ).to(device)

    if args.vae:
        logger.info(f"Loading Attribute-Aware VAE checkpoint from {args.vae}")
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
        logger.info(f"VAE load msg - Missing keys: {msg.missing_keys[:5]}, Unexpected keys: {msg.unexpected_keys[:5]}")
    for param in ae_model.parameters():
        param.requires_grad = False
    del ae_model.text_encoder2
    ae_model.eval()

    if rank == 0:
        print(f'Total VAE parameters: {sum(p.numel() for p in ae_model.parameters())}')

    assert args.cfg_scale >= 1.0
    using_cfg = args.cfg_scale > 1.0

    prompt_null = "no description."
    biot5_embed_null, mask_null = text_graph_encoder.encode([prompt_null])
    biot5_embed_null = biot5_embed_null.to(device).to(torch.float32)
    mask_null = mask_null.to(device).bool()

    test_dataset = smi_txt_dataset(
        ['D:\project\FALD-Mol\data\chebi_20\test_parsed.txt'],
        data_length=None,
        shuffle=False,
        unconditional=False,
        raw_description=True,
    )
    if rank == 0:
        print(f'Test set size: {len(test_dataset)}')

    loader = DataLoader(
        test_dataset,
        batch_size=int(args.per_proc_batch_size),
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    st = time.time()
    loader = tqdm(loader, miniters=1) if rank == 0 else loader

    output_file = './raw_vs_generated_molecules.txt'
    if rank == 0:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Raw Molecule Generated Molecule\n")
        print(f"Results will be saved to: {output_file}")

    all_raw_smiles = []
    all_pred_smiles = []

    for x, y, _ in loader:
        raw_smiles = [s.replace('[CLS]', '').strip() for s in x]
        all_raw_smiles.extend(raw_smiles)
        z = torch.randn(len(x), model.in_channels, latent_size, 1, device=device)

        biot5_embed, pad_mask = text_graph_encoder.encode(y)
        y_cond = biot5_embed.to(device).type(torch.float32)
        pad_mask_cond = pad_mask.to(device).bool()
        y_null = repeat(biot5_embed_null, '1 L D -> B L D', B=len(x))
        pad_mask_null = repeat(mask_null, '1 L -> B L', B=len(x))

        if using_cfg:
            z = torch.cat([z, z], 0)
            y = torch.cat([y_cond, y_null], 0)
            pad_mask = torch.cat([pad_mask_cond, pad_mask_null], 0)
            model_kwargs = dict(y=y, pad_mask=pad_mask, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y_cond, pad_mask=pad_mask_cond)
            sample_fn = model.forward

        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)

        samples = samples.squeeze(-1).permute((0, 2, 1))
        generated_smiles = AE_SMILES_decoder(samples, ae_model, stochastic=False, k=1)
        all_pred_smiles.extend(generated_smiles)

        assert len(raw_smiles) == len(generated_smiles)
        if rank == 0:
            with open(output_file, 'a', encoding='utf-8') as f:
                for raw_smi, gen_smi in zip(raw_smiles, generated_smiles):
                    clean_raw = raw_smi.replace('\n', '').replace('\r', '').strip()
                    clean_gen = gen_smi.replace('\n', '').replace('\r', '').strip()
                    f.write(f"{clean_raw} {clean_gen}\n")

    all_pred_smiles = [
        Chem.MolToSmiles(Chem.MolFromSmiles(l), isomericSmiles=True, canonical=True)
        if Chem.MolFromSmiles(l) else l
        for l in all_pred_smiles
    ]
    all_raw_smiles = [
        Chem.MolToSmiles(Chem.MolFromSmiles(l), isomericSmiles=True, canonical=True)
        for l in all_raw_smiles
    ]

    standardized_output = './raw_vs_generated_molecules_standardized.txt'
    if rank == 0:
        with open(standardized_output, 'w', encoding='utf-8') as f:
            f.write("# Raw Molecule (Standardized) Generated Molecule (Standardized)\n")
            for raw_smi, gen_smi in zip(all_raw_smiles, all_pred_smiles):
                f.write(f"{raw_smi} {gen_smi}\n")
        print(f"Standardized results saved to: {standardized_output}")

    assert len(all_raw_smiles) == len(all_pred_smiles)
    print(f"\nClosed-loop generation finished: {len(all_raw_smiles)} samples")
    print(f"Raw and generated molecules saved to: {output_file}")
    print(f"Standardized version saved to: {standardized_output}")

    print("\n=== Starting Evaluation: Raw vs Generated Molecules ===")
    print("=" * 60)

    print("\n[1. Molecular Fingerprint Similarity]")
    validity, maccs_sim, rdk_sim, morgan_sim = molfinger_evaluate(
        targets=all_raw_smiles,
        preds=all_pred_smiles,
        verbose=True
    )

    print("\n[2. Comprehensive Metrics Evaluation]")
    bleu, exact_match, lev_dist, val_score, result_df = mol_evaluate(
        targets=all_raw_smiles,
        preds=all_pred_smiles,
        verbose=True
    )

    print("\n=== Saving Results ===")
    os.makedirs('./vae_comparison_results', exist_ok=True)

    result_df['raw_smiles'] = all_raw_smiles
    result_df['pred_smiles'] = all_pred_smiles
    result_df.to_csv('./vae_comparison_results/vae_comparison_details.csv', index=False, encoding='utf-8')
    print("Detailed comparison results saved to: vae_comparison_results/vae_comparison_details.csv")

    summary_metrics = {
        'Total Samples': len(all_raw_smiles),
        'Validity': validity,
        'MACCS Similarity': maccs_sim,
        'RDK Similarity': rdk_sim,
        'Morgan Similarity': morgan_sim,
        'BLEU Score': bleu,
        'Exact Match': exact_match,
        'Avg Levenshtein Distance': lev_dist,
    }
    summary_df = pd.DataFrame([summary_metrics])
    summary_df.to_csv('./vae_comparison_results/vae_comparison_summary.csv', index=False, encoding='utf-8')
    print("Summary metrics saved to: vae_comparison_results/vae_comparison_summary.csv")

    print("\n" + "=" * 60)
    print("=== Final Evaluation Report ===")
    print("=" * 60)
    for key, value in summary_metrics.items():
        if isinstance(value, float):
            print(f"{key:<22}: {value:.4f}")
        else:
            print(f"{key:<22}: {value}")
    print("=" * 60)

    if rank == 0:
        print(f'\nTotal time: {time.time() - st:.2f} seconds')
        print('All results saved successfully!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="FALD")
    parser.add_argument("--description-length", type=int, default=256)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--vae", type=str, default="D:\project\FALD-Mol\Pretrain_autoencoder\have-pretrain-18.ckpt")
    parser.add_argument("--t_g_ckpt", type=str, default="D:\project\FALD-Mol\contrast_text_graph\have-pretrain\checkpoints\best_val_model.pt")
    parser.add_argument("--per-proc-batch-size", type=int, default=64)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    main(args)