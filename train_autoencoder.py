from pytorch_lightning.callbacks import LearningRateMonitor
from xbert import BertConfig, BertForMaskedLM
from transformers import BertModel
import torch
from torch import nn
import torch.distributed
from scheduler import create_scheduler
import copy
from torch.utils.data import DataLoader
from dataset import SMILESDataset_pretrain
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import argparse
from pathlib import Path
from utils import regexTokenizer
import torch.nn.functional as F


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class vae_with_property(pl.LightningModule):
    def __init__(self,
                 cp=None,
                 spmm_ckpt=None,
                 config=None,
                 loader_len=0,
                 no_train=False,
                 tokenizer=None,
                 use_linear=True,
                 num_properties=5):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.config = config
        self.tokenizer = tokenizer
        self.training_step_outputs = []
        self.num_properties = num_properties

        self.text_encoder = BertForMaskedLM(config=BertConfig.from_json_file(config['bert_config_decoder']))
        bert_config2 = BertConfig.from_json_file(config['bert_config_encoder'])
        self.text_encoder2 = BertModel(config=bert_config2)

        if cp:
            checkpoint = torch.load(cp, map_location='cpu')
            state_dict = copy.deepcopy(checkpoint.get('model', checkpoint.get('state_dict')))
            new_state_dict = {}
            for key in state_dict:
                if 'text_encoder.' in key:
                    new_key = key.replace('text_encoder.', '')
                    new_state_dict[new_key] = state_dict[key]
            msg = self.text_encoder2.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded encoder checkpoint: {msg}")
        for param in self.text_encoder2.parameters():
            param.requires_grad = False

        self.use_linear = use_linear
        if use_linear:
            self.output_dim = 64
            final_dim = bert_config2.hidden_size
            self.encode_prefix = nn.Linear(final_dim, self.output_dim)
            self.decode_prefix = nn.Linear(self.output_dim, final_dim)
            self.fuse_proj = nn.Sequential(
                nn.Linear(2 * self.output_dim, self.output_dim),
                nn.LayerNorm(self.output_dim)
            )

        self.property_width = config['property_width']
        bert_config_property = BertConfig.from_json_file(config['bert_config_property'])
        self.property_encoder = BertForMaskedLM(config=bert_config_property).bert
        self.property_embed = nn.Linear(num_properties, self.property_width)
        self.property_cls = nn.Parameter(torch.randn(1, 1, self.property_width, device=self.device))
        self.property_proj = nn.Linear(self.property_width, self.output_dim)

        if spmm_ckpt:
            print(f"Loading pretrained SPMM from {spmm_ckpt}")
            spmm_checkpoint = torch.load(spmm_ckpt, map_location='cpu')
            spmm_state_dict = spmm_checkpoint.get('model', spmm_checkpoint.get('state_dict', {}))

            filtered_spmm = {}
            for k, v in spmm_state_dict.items():
                if 'property_embed' in k:
                    print(f"Skipping {k} (property count mismatch)")
                    continue
                if 'property_proj.weight' in k:
                    print(f"Adapting {k} from 256dim to 64dim")
                    filtered_spmm[k] = v[:self.output_dim, :]
                    continue
                if 'property_proj.bias' in k:
                    print(f"Adapting {k} from 256dim to 64dim")
                    filtered_spmm[k] = v[:self.output_dim]
                    continue
                if any(prefix in k for prefix in ['property_encoder.', 'property_cls']):
                    new_k = k.replace('model.', '') if k.startswith('model.') else k
                    filtered_spmm[k] = v

            msg = self.load_state_dict(filtered_spmm, strict=False)
            print(
                f"SPMM weights loaded. Missing keys: {msg.missing_keys[:5]}, Unexpected keys: {msg.unexpected_keys[:5]}")

        if not no_train:
            self.loader_len = loader_len
            self.warmup_steps = config['schedular']['warmup_epochs']
            self.mlm_probability = config['mlm_probability']

    def forward(self, text_input_ids, text_attention_mask, text_input_ids2, text_attention_mask2, properties):
        with torch.no_grad():
            text_embeds = self.text_encoder2(
                text_input_ids2,
                attention_mask=text_attention_mask2,
                return_dict=True
            ).last_hidden_state

        if self.use_linear:
            z_smiles = self.encode_prefix(text_embeds)
            z_smiles_norm = F.normalize(z_smiles, dim=-1)

        batch_size = properties.shape[0]
        properties = properties.squeeze(1)
        property_feature = self.property_embed(properties)
        property_feature = property_feature.unsqueeze(1)

        properties_input = torch.cat([
            self.property_cls.expand(batch_size, -1, -1),
            property_feature
        ], dim=1)
        prop_atts = torch.ones(properties_input.size()[:-1], dtype=torch.long, device=self.device)

        prop_embeds = self.property_encoder(
            inputs_embeds=properties_input,
            attention_mask=prop_atts,
            return_dict=True
        ).last_hidden_state
        z_prop = self.property_proj(prop_embeds[:, 0, :])
        z_prop_norm = F.normalize(z_prop, dim=-1)

        z_prop_broadcast = z_prop_norm.unsqueeze(1).repeat(1, z_smiles.shape[1], 1)
        z_concat = torch.cat([0.8*z_smiles_norm, 0.2*z_prop_broadcast], dim=-1)
        z_fused = self.fuse_proj(z_concat)

        if self.use_linear:
            fused_emb = self.decode_prefix(z_fused)

        input_ids = text_input_ids.clone()
        labels = input_ids.clone()[:, 1:].contiguous()
        mlm_output = self.text_encoder(
            input_ids,
            attention_mask=text_attention_mask,
            encoder_hidden_states=fused_emb,
            return_dict=True,
            is_decoder=True,
            return_logits=True,
        )[:, :-1, :]
        loss_mlm = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')(mlm_output.transpose(1, 2), labels)
        return loss_mlm

    def configure_optimizers(self):
        arg_opt = self.config['optimizer']
        optimizer = torch.optim.AdamW(self.parameters(), lr=arg_opt['lr'], weight_decay=arg_opt['weight_decay'])

        arg_sche = AttrDict(self.config['schedular'])
        scheduler, _ = create_scheduler(arg_sche, optimizer)
        print(f"Scheduler type: {type(scheduler)}")
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if self.global_rank == 0:
            optimizer = self.optimizers()[optimizer_idx]
            current_epoch = self.current_epoch
            print(f"LR updated to: {optimizer.param_groups[0]['lr']:.8f} (Epoch: {current_epoch})")

    def training_step(self, train_batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()

        text, properties = train_batch

        text_input_ids = self.tokenizer(text, truncation='longest').to(self.device)
        text_attention_mask = torch.where(text_input_ids == 0, 0, 1).to(self.device)
        text2 = text
        text2_input_ids = self.tokenizer(text2, truncation='longest').to(self.device)
        text2_attention_mask = torch.where(text2_input_ids == 0, 0, 1).to(self.device)
        properties = properties.to(self.device)

        loss_mlm = self(
            text_input_ids, text_attention_mask,
            text2_input_ids, text2_attention_mask,
            properties
        )
        loss = loss_mlm

        if loss != torch.tensor(0., device=self.device):
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            optimizer.step()
            scheduler.step(epoch=self.current_epoch)

        else:
            print('aaaaaaaaaaaa')

        self.log('lr', optimizer.param_groups[0]["lr"], prog_bar=True)
        self.log('loss_mlm', loss_mlm, prog_bar=True)

        self.training_step_outputs.append(torch.tensor([loss_mlm, ], device=self.device))
        return loss_mlm

    def on_train_epoch_end(self):
        tmp = torch.stack(self.training_step_outputs[-1000:]).mean(dim=0).tolist()
        if self.global_rank == 0:
            print(f'\n mean loss: {tmp[0]:.4f}')

        self.training_step_outputs.clear()


def main(args, config):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    if local_rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Creating dataset")
    dataset = SMILESDataset_pretrain(
        args.data_path,
        data_length=[0, 10000000],
        is_train=False,
        shuffle=True,
        norm_path=args.norm_path
    )
    print(f'Loaded {len(dataset)} samples, {dataset.num_properties} properties')

    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    tokenizer = regexTokenizer(vocab_path=args.vocab_filename, max_len=127)

    print("Creating model")
    model = vae_with_property(
        config=config,
        cp=args.enc_checkpoint,
        spmm_ckpt=args.spmm_checkpoint,
        tokenizer=tokenizer,
        use_linear=True,
        num_properties=dataset.num_properties
    )
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename='epoch_{epoch:02d}_loss_{loss_mlm:.4f}',
        save_top_k=-1,
        every_n_epochs=1,
        save_last=False,
        save_weights_only=False,
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[2, 3, 4, 5, 6, 7],
        precision='16-mixed',
        max_epochs=config['schedular']['epochs'],
        callbacks=[checkpoint_callback, lr_monitor],
        strategy=DDPStrategy(find_unused_parameters=True),
        limit_val_batches=0.,
        accumulate_grad_batches=1,
        log_every_n_steps=400,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    if args.checkpoint:
        print(f"Resuming training from checkpoint: {args.checkpoint}")
        trainer.fit(model, train_dataloaders=data_loader, ckpt_path=args.checkpoint)
    else:
        trainer.fit(model, train_dataloaders=data_loader, ckpt_path=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--enc_checkpoint',
                        default='D:\project\FALD-Mol\Pretrain_encoder\checkpoint_epoch=4.ckpt')
    parser.add_argument('--spmm_checkpoint',
                        default='D:\project\FALD-Mol\pretrain_spmm\checkpoint_epoch=5-v2.ckpt')
    parser.add_argument('--data_path', default='D:\project\FALD-Mol\data\pubchem_10m.txt')
    parser.add_argument('--norm_path', default='D:\project\FALD-Mol\normalize.pkl')
    parser.add_argument('--output_dir', default='D:\project\FALD-Mol\Pretrain_autoencoder')
    parser.add_argument('--vocab_filename', default='D:\project\FALD-Mol\vocab_bpe_300_sc.txt')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    args = parser.parse_args()

    pretrain_config = {
        'property_width': 768,
        'embed_dim': 256,
        'batch_size': 256,
        'temp': 0.07,
        'mlm_probability': 0.15,
        'momentum': 0.995,
        'alpha': 0.4,
        'bert_config_decoder': './config_decoder.json',
        'bert_config_encoder': './config_encoder.json',
        'bert_config_property': 'D:\project\FALD-Mol\config_bert_property.json',
        'schedular': {
            'sched': 'cosine',
            'lr': 1e-4 * 6,
            'epochs': 10,
            'min_lr': 1e-6 * 6,
            'decay_rate': 1,
            'warmup_lr': 1e-5 * 6,
            'warmup_epochs': 0,
            'cooldown_epochs': 2
        },
        'optimizer': {
            'opt': 'adamW',
            'lr': 1e-4 * 6,
            'weight_decay': 0.01,
            'betas': (0.9, 0.98)
        }
    }

    pl.seed_everything(args.seed, workers=True)
    main(args, pretrain_config)