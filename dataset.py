import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
from torch_geometric.data import Data
from SPMM.calc_property import calculate_property
from utils import split_into_sentences
from rdkit import Chem, RDLogger
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from rdkit.Chem.rdchem import BondType as BT
import os

RDLogger.DisableLog('rdApp.*')

ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

class smi_txt_dataset(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False, unconditional=False, raw_description=False, norm_path='D:\project\FALD-Mol\SPMM\normalize.pkl'):
        self.data = []
        if isinstance(data_path, str):
            data_path = [data_path]

        for dp in data_path:
            with open(dp, 'r', encoding='utf-8') as r:
                lines = [l.strip() for l in r.readlines()][1 if dp.endswith('.csv') else 0:]
                self.data += lines

        with open(norm_path, 'rb') as f:
            norm = pickle.load(f)
        self.property_mean = torch.tensor(norm[0], dtype=torch.float32) if isinstance(norm[0], (list, np.ndarray)) else norm[0]
        self.property_std = torch.tensor(norm[1], dtype=torch.float32) if isinstance(norm[1], (list, np.ndarray)) else norm[1]

        self.num_properties = 53
        if self.property_mean.shape[0] != self.num_properties or self.property_std.shape[0] != self.num_properties:
            raise ValueError(
                f"Normalization parameter dimension mismatch! Required 53 for both, current mean: {self.property_mean.shape[0]}, std: {self.property_std.shape[0]}"
            )

        self.unconditional = unconditional
        self.raw_description = raw_description
        self.null_text = "no description."
        self.train = not raw_description

        filtered_data = []
        invalid_patterns = [r'^=', r'^\(', r'^\+', r'^\-', r'^@', r'^\[']
        for line in self.data:
            try:
                parts = line.split('\t')
                if len(parts) >= 3:
                    smiles = parts[1].strip()
                elif len(parts) == 2:
                    smiles = parts[0].strip()
                elif len(parts) == 1:
                    smiles = parts[0].strip()
                else:
                    continue

                if not smiles or '*' in smiles or len(smiles) > 120 or len(smiles) < 5:
                    continue

                if '.' in smiles:
                    smiles = max(smiles.split('.'), key=len)

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                canonical_smiles = Chem.MolToSmiles(
                    mol,
                    canonical=True,
                    isomericSmiles=False,
                    kekuleSmiles=False
                )

                mol_canon = Chem.MolFromSmiles(canonical_smiles)
                if mol_canon is None:
                    continue

                if len(parts) >= 3:
                    new_line = f"{parts[0]}\t{canonical_smiles}\t{parts[2]}"
                elif len(parts) == 2:
                    new_line = f"{canonical_smiles}\t{parts[1]}"
                else:
                    new_line = canonical_smiles

                filtered_data.append(new_line)
            except Exception as e:
                continue

        if data_length:
            filtered_data = filtered_data[:data_length]

        if shuffle and self.train:
            random.shuffle(filtered_data)

        self.data = filtered_data
        print(f"Dataset loaded successfully: valid samples = {len(self.data)}")

    def _calculate_normalized_props(self, smiles):
        try:
            raw_props = calculate_property(smiles)
        except Exception as e:
            raw_props = torch.zeros(self.num_properties, dtype=torch.float32)

        raw_props = torch.nan_to_num(raw_props, nan=0.0, posinf=0.0, neginf=0.0)
        std = torch.where(self.property_std == 0, torch.tensor(1e-8, dtype=torch.float32), self.property_std)
        normalized_props = (raw_props - self.property_mean) / std

        return normalized_props.unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            contents = self.data[index].split('\t')
            if len(contents) == 3:
                cid, smiles, description = contents
            elif len(contents) == 2:
                smiles, description = contents
            elif len(contents) == 1:
                smiles, description = contents[0], self.null_text
            else:
                raise ValueError(f"Invalid data format: {contents}")

            if self.unconditional:
                description = self.null_text

            if not self.raw_description and description != self.null_text:
                description = sentence_randomize(description, only_one=True)

            original_smiles = smiles
            if self.train and random.random() > 0.5:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        stereoisomers = list(EnumerateStereoisomers(mol))
                        if len(stereoisomers) >= 2:
                            mol_aug = random.choice(stereoisomers)
                            smiles = Chem.MolToSmiles(
                                mol_aug,
                                canonical=True,
                                isomericSmiles=False
                            )
                        elif len(stereoisomers) == 1:
                            smiles = Chem.MolToSmiles(
                                stereoisomers[0],
                                canonical=True,
                                isomericSmiles=False
                            )
                except Exception as e:
                    smiles = original_smiles

            properties = self._calculate_normalized_props(smiles)

            return '[CLS]' + smiles, description, properties
        except Exception as e:
            raise RuntimeError(f"Failed to process sample index={index}: {str(e)}") from e

class smi_txt_dataset1(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False, unconditional=False, raw_description=False):
        self.data = []
        if isinstance(data_path, str):
            data_path = [data_path]

        for dp in data_path:
            with open(dp, 'r') as r:
                self.data += [l.strip() for l in r.readlines()][1 if dp.endswith('.csv') else 0:]

        with open('D:\project\FALD-Mol\normalize.pkl', 'rb') as w:
            norm = pickle.load(w)
        self.property_mean, self.property_std = norm

        if shuffle:
            random.shuffle(self.data)

        if data_length:
            self.data = self.data[:data_length]
        self.unconditional = unconditional
        self.raw_description = raw_description
        self.null_text = "no description."
        filtered_data = []
        for line in self.data:
            parts = line.split('\t')
            if len(parts) >= 3:
                smiles = parts[1].strip()
            elif len(parts) == 2:
                smiles = parts[0].strip()
            elif len(parts) == 1:
                smiles = parts[0].strip()
            else:
                continue
            if smiles and '*' not in smiles:
                filtered_data.append(line)
        self.data = filtered_data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            contents = self.data[index].split('\t')
            if len(contents) == 3:
                cid, smiles, description = contents
            elif len(contents) == 2:
                smiles, description = contents
            elif len(contents) == 1:
                smiles, description = contents[0], self.null_text
            else:
                raise ValueError("Invalid data format in the dataset!")
            if self.unconditional:
                description = self.null_text

            if not self.raw_description and description != self.null_text:
                description = sentence_randomize(description, only_one=True)
            else:
                pass

            if '.' in smiles:
                smiles = max(smiles.split('.'), key=len)
            graph = smiles_to_graph(smiles)
            mol = Chem.MolFromSmiles(smiles)
            sc_list = list(EnumerateStereoisomers(mol))
            mol = random.choice(sc_list)
            smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

            properties = (calculate_property(smiles) - self.property_mean) / self.property_std

            prop_mask = torch.ones(53)
            for i in range(len(properties)):
                if properties[i] == 0:
                    prop_mask[i] = 0

            return '[CLS]' + smiles, description, properties, prop_mask, graph
        except Exception as e:
            print(e)
            print('aaa', self.data[index])
            raise NotImplementedError

class SMILESDataset_pretrain1(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        if data_length is not None:
            with open(data_path, 'r') as f:
                for _ in range(data_length[0]):
                    f.readline()
                lines = []
                for _ in range(data_length[1] - data_length[0]):
                    lines.append(f.readline())
        else:
            with open(data_path, 'r') as f:
                lines = f.readlines()
        self.data = [l.strip() for l in lines]
        with open('D:\project\FALD-Mol\normalize.pkl', 'rb') as w:
            norm = pickle.load(w)
        self.property_mean, self.property_std = norm

        if shuffle:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]), isomericSmiles=False, canonical=True)
        properties = (calculate_property(smiles) - self.property_mean) / self.property_std

        return '[CLS]' + smiles, properties

class SMILESDataset_pretrain(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False, is_train=True, norm_path='D:\project\FALD-Mol\SPMM\normalize.pkl'):
        if data_length is not None:
            with open(data_path, 'r', encoding='utf-8') as f:
                for _ in range(data_length[0]):
                    f.readline()
                lines = []
                for _ in range(data_length[1] - data_length[0]):
                    line = f.readline()
                    if line and line.strip():
                        lines.append(line.strip())
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

        if not lines:
            raise ValueError("Dataset is empty, please check the data path or file validity")
        self.data = lines

        if shuffle and is_train:
            random.shuffle(self.data)

        self.train = is_train

        with open(norm_path, 'rb') as f:
            norm = pickle.load(f)
        self.property_mean = torch.tensor(norm[0], dtype=torch.float32) if isinstance(norm[0], (list, np.ndarray)) else norm[0]
        self.property_std = torch.tensor(norm[1], dtype=torch.float32) if isinstance(norm[1], (list, np.ndarray)) else norm[1]

        self.num_properties = 53
        if self.property_mean.shape[0] != self.num_properties or self.property_std.shape[0] != self.num_properties:
            raise ValueError(
                f"Normalization parameter dimension mismatch! Required 53 for both, current mean: {self.property_mean.shape[0]}, std: {self.property_std.shape[0]}"
            )

    def __len__(self):
        return len(self.data)

    def _calculate_normalized_props(self, smiles):
        try:
            raw_props = calculate_property(smiles)
        except Exception as e:
            raw_props = torch.zeros(self.num_properties, dtype=torch.float32)

        raw_props = torch.nan_to_num(raw_props, nan=0.0, posinf=0.0, neginf=0.0)
        std = torch.where(self.property_std == 0, torch.tensor(1e-8, dtype=torch.float32), self.property_std)
        normalized_props = (raw_props - self.property_mean) / std

        return normalized_props.unsqueeze(0)

    def __getitem__(self, index):
        smiles = self.data[index]
        smiles_aug = smiles

        if self.train and random.random() > 0.5:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    stereoisomers = list(EnumerateStereoisomers(mol))
                    if len(stereoisomers) >= 2:
                        mol_raw, mol_aug = random.sample(stereoisomers, k=2)
                        smiles = Chem.MolToSmiles(mol_raw, canonical=True, isomericSmiles=True)
                        smiles_aug = Chem.MolToSmiles(mol_aug, canonical=True, isomericSmiles=True)
                    elif len(stereoisomers) == 1:
                        smiles = Chem.MolToSmiles(stereoisomers[0], canonical=True, isomericSmiles=True)
            except Exception as e:
                pass

        props = self._calculate_normalized_props(smiles)

        if self.train:
            return smiles, smiles_aug, props
        else:
            return  smiles, props

def sentence_randomize(description, only_one=False):
    desc = split_into_sentences(description)
    desc2, tmp = [], []
    for d in desc:
        if not d[0].isalpha():
            if tmp:
                tmp.append(d)
            else:
                if len(desc2) == 0:
                    desc2.append(d)
                    continue
                head = desc2.pop()
                tmp.append(head)
                tmp.append(d)
        else:
            if tmp:
                desc2.append(' '.join(tmp))
                tmp = []
            desc2.append(d)
    if tmp:
        desc2.append(' '.join(tmp))
    forced = random.randint(0, len(desc2) - 1)
    if only_one:
        desc2 = [desc2[forced]]
    else:
        desc2 = [d for i, d in enumerate(desc2) if random.random() < 0.5 or i == 0]
    description = ' '.join(desc2)
    return description

def smiles_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        mol = Chem.AddHs(mol)

        type_idx = []
        chirality_idx = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if atomic_num not in ATOM_LIST:
                raise ValueError(f"Unsupported atomic number: {atomic_num} in SMILES {smiles}")
            type_idx.append(ATOM_LIST.index(atomic_num))
            chirality = atom.GetChiralTag()
            if chirality not in CHIRALITY_LIST:
                raise ValueError(f"Unsupported chirality: {chirality} in SMILES {smiles}")
            chirality_idx.append(CHIRALITY_LIST.index(chirality))

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        row, col = [], []
        edge_feat = []
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            row.extend([start, end])
            col.extend([end, start])

            bond_type = bond.GetBondType()
            if bond_type not in BOND_LIST:
                raise ValueError(f"Unsupported bond type: {bond_type} in SMILES {smiles}")
            bond_dir = bond.GetBondDir()
            if bond_dir not in BONDDIR_LIST:
                raise ValueError(f"Unsupported bond direction: {bond_dir} in SMILES {smiles}")
            edge_feat.append([BOND_LIST.index(bond_type), BONDDIR_LIST.index(bond_dir)])
            edge_feat.append([BOND_LIST.index(bond_type), BONDDIR_LIST.index(bond_dir)])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(edge_feat, dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        raise