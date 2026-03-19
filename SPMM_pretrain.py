import os
from utils import regexTokenizer
from torch.utils.data import DataLoader
from dataset import smi_txt_dataset
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import torch.distributed
from SPMM.SPMM_models import SPMM
import argparse
from pathlib import Path
from transformers import BertTokenizer
from torch_geometric.data import Batch
def collate_fn(batch):
    smiles, descs, props, prop_masks, graphs = zip(*batch)
    return (
        list(smiles),
        list(descs),
        torch.stack(props, dim=0),       # [B, 53]
        torch.stack(prop_masks, dim=0),  # [B, 53]
        Batch.from_data_list(graphs)     # PyG 的图 Batch
    )


os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
def main(args, config):
    ngpu=6
    # data
    print("Creating dataset")
    dataset = smi_txt_dataset(args.data_path, data_length=None, shuffle=True, unconditional=False, raw_description=True)
    print('#data:', len(dataset), torch.cuda.is_available())
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=8, shuffle=False, pin_memory=True, drop_last=True,collate_fn=collate_fn)
    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False)
    tokenizer = regexTokenizer(vocab_path=args.vocab_filename, max_len=127)
    # tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)

    # model
    model = SPMM(config=config, tokenizer=tokenizer, loader_len=len(data_loader) // torch.cuda.device_count(),cp=args.enc_checkpoint)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        _ = model.load_state_dict(checkpoint['state_dict'], strict=False)

    # training
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args.output_dir, filename='checkpoint_{epoch}',
                                                       #save_top_k=2, 
                                                       #monitor='epoch',
                                                       #every_n_epochs=1,
                                                       every_n_train_steps=10000,
                                                       )
    trainer = pl.Trainer(accelerator='gpu', devices=ngpu, precision='16-mixed', max_epochs=config['schedular']['epochs'],
                         callbacks=[checkpoint_callback], strategy=DDPStrategy(find_unused_parameters=True), limit_val_batches=0.)
    trainer.fit(model, data_loader, None, ckpt_path=args.checkpoint if args.checkpoint else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--enc_checkpoint', default='D:\project\FALD-Mol\Pretrain_encoder\checkpoint_epoch=4.ckpt')
    parser.add_argument('--data_path', default='D:\project\FALD-Mol\data\pubchem_10m.txt')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='D:\project\FALD-Mol\pretrain_spmm')
    parser.add_argument('--vocab_filename', default='D:\project\FALD-Mol\vocab_bpe_300_sc.txt')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    pretrain_config = {
        'property_width': 1024,
        'embed_dim': 256,
        'batch_size': 32,
        'temp': 0.07,
        'mlm_probability': 0.15,
        'queue_size': 36864, # 24576,
        'momentum': 0.995,
        'alpha': 0.4,
        'bert_config_text': 'D:\project\FALD-Mol\config_encoder.json',
        'bert_config_prop': 'D:\project\FALD-Mol\config_decoder.json',
        'bert_config_property': 'D:\project\FALD-Mol\SPMM\config_bert_property.json',
        'schedular': {'sched': 'cosine', 'lr': 5e-5, 'epochs': 6, 'min_lr': 1e-5,
                      'decay_rate': 1, 'warmup_lr': 5e-5, 'warmup_epochs': 20, 'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': 5e-5, 'weight_decay': 0.02}
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, pretrain_config)
