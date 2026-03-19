from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
from Levenshtein import distance as lev
import numpy as np

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
from fcd import get_fcd, load_ref_model, canonical_smiles
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def canonical_smiles(smiles_list):
    canonical = []
    invalid_count = 0
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canonical.append(Chem.MolToSmiles(mol, isomericSmiles=False))
            else:
                invalid_count += 1
                print(f"Invalid SMILES: {smi}")
        except Exception as e:
            invalid_count += 1
            print(f"Error processing SMILES {smi}: {e}")
    if invalid_count > 0:
        print(f"Warning: {invalid_count} out of {len(smiles_list)} SMILES were invalid or could not be processed.")
    return canonical

def molfinger_evaluate(targets, preds, morgan_r=2, verbose=False):
    outputs = []
    bad_mols = 0

    for i in range(len(targets)):
        try:
            gt_smi = targets[i]
            ot_smi = preds[i]
            gt_m = Chem.MolFromSmiles(gt_smi)
            ot_m = Chem.MolFromSmiles(ot_smi)

            if ot_m is None:
                raise ValueError('Bad SMILES')
            outputs.append(('test', gt_m, ot_m))
        except:
            bad_mols += 1

    validity_score = len(outputs) / (len(outputs) + bad_mols)
    if verbose:
        print('validity:', validity_score)

    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []

    enum_list = outputs

    for i, (desc, gt_m, ot_m) in enumerate(enum_list):

        # if i % 100 == 0 and verbose:
        #     print(i, 'processed.')

        MACCS_sims.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m), metric=DataStructs.TanimotoSimilarity))
        RDK_sims.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_m), Chem.RDKFingerprint(ot_m), metric=DataStructs.TanimotoSimilarity))
        morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_m, morgan_r), AllChem.GetMorganFingerprint(ot_m, morgan_r)))

    maccs_sims_score = np.mean(MACCS_sims)
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)
    if verbose:
        print('Average MACCS Similarity:', maccs_sims_score)
        print('Average RDK Similarity:', rdk_sims_score)
        print('Average Morgan Similarity:', morgan_sims_score)

    return validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score


def mol_evaluate(targets, preds, verbose=False):
    fcd_model = load_ref_model()
    # fcd_score = get_fcd([w for w in canonical_smiles(targets) if w is not None], [w for w in canonical_smiles(preds) if w is not None], fcd_model)
    # print("FCD", fcd_score)
    fcd_model = load_ref_model()
    valid_targets = [w for w in canonical_smiles(targets) if w is not None]
    valid_preds = [w for w in canonical_smiles(preds) if w is not None]
    if not valid_targets or not valid_preds:
        print("Warning: One or both of the SMILES lists are empty after validation.")
        fcd_score = float('nan')  # 或者其他默认值
    else:
        fcd_score = get_fcd(valid_targets, valid_preds, fcd_model)
    print("FCD", fcd_score)

    outputs = []

    for i in range(len(targets)):
        gt_smi = targets[i]
        ot_smi = preds[i]
        outputs.append((None, gt_smi, ot_smi))

    references = []
    hypotheses = []

    for i, (smi, gt, out) in enumerate(outputs):

        # if i % 100 == 0:
        #     if verbose:
        #         print(i, 'processed.')

        gt_tokens = [c for c in gt]

        out_tokens = [c for c in out]

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        # mscore = meteor_score([gt], out)
        # meteor_scores.append(mscore)

    # BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    if verbose:
        print('BLEU score:', bleu_score)


    # Meteor score
    # _meteor_score = np.mean(meteor_scores)
    # print('Average Meteor score:', _meteor_score)

    
    #preds = [Chem.MolToSmiles(Chem.MolFromSmiles(l), isomericSmiles=False, canonical=True) if Chem.MolFromSmiles(l) else l for l in preds]
    #targets = [Chem.MolToSmiles(Chem.MolFromSmiles(l), isomericSmiles=False, canonical=True) for l in targets]
    #for i in range(len(targets)):
    #    gt_smi = targets[i]
    #    ot_smi = preds[i]
    #    outputs.append((None, gt_smi, ot_smi))
    
    references = []
    hypotheses = []

    levs = []

    num_exact = 0

    bad_mols = 0

    result_dataframe = pd.DataFrame(outputs, columns=['summary', 'ground truth', 'isosmiles'])
    result_dataframe['exact'] = 0
    result_dataframe['valid'] = 0
    result_dataframe['lev'] = 1000
    # print(result_dataframe.head())
    i = 0
    for i, (smi, gt, out) in enumerate(outputs):

        hypotheses.append(out)
        references.append(gt)

        try:
            m_out = Chem.MolFromSmiles(out)
            m_gt = Chem.MolFromSmiles(gt)
            
            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt):
            #print(len(list(EnumerateStereoisomers(m_out))))
            #if Chem.MolToInchi(m_gt) in [Chem.MolToInchi(tmp) for tmp in list(EnumerateStereoisomers(m_out))]:
                num_exact += 1
                result_dataframe.at[i, 'exact'] = 1
            else:
                result_dataframe.at[i, 'exact'] = 0
            
            # if gt == out: num_exact += 1 #old version that didn't standardize strings
            result_dataframe.at[i, 'valid'] = 1
        except:
            bad_mols += 1
            result_dataframe.at[i, 'valid'] = 0

        levs.append(lev(out, gt))
        # result_dataframe.iloc[i]['lev'] = lev(out, gt)
        result_dataframe.at[i, 'lev'] = lev(out, gt)

    # Exact matching score
    exact_match_score = num_exact / (i + 1)
    if verbose:
        print('Exact Match:', exact_match_score)

    # Levenshtein score
    levenshtein_score = np.mean(levs)
    if verbose:
        print('Levenshtein:', levenshtein_score)

    validity_score = 1 - bad_mols / len(outputs)
    if verbose:
        print('validity:', validity_score)

    

    return bleu_score, exact_match_score, levenshtein_score, validity_score, result_dataframe


def read_smiles_from_file(filepath):
    gt_smis = []
    op_smis = []
    with open(filepath) as f:
        lines = f.readlines()
    for line in lines:
        if len(line) < 3:
            continue
        s0, s1 = line.split(' || ')
        s0, s1 = s0.strip(), s1.strip()
        gt_smis.append(s1)
        op_smis.append(s0)
    return gt_smis, op_smis

if __name__ == '__main__':
    target = ['c1ccccc1', 'c1ccccc1O', 'c1ccccc1F', 'c1ccccc1C']
    pred = ['c1cccnc1', 'c1cnccc1O', 'c1ccccc1F', 'c1ncccc1C']
    _ = mol_evaluate(target, pred, verbose=True)[-1]
    print('='*100)
    molfinger_evaluate(target, pred, verbose=True)
    # file_path = 'generated_molecules_t2m1.txt'
    # targets, preds = read_smiles_from_file(file_path)
    bleu_score, exact_match_score, levenshtein_score, validity_score, result_df = mol_evaluate(target, pred, verbose=True)
    print('=' * 100)
    print("BLEU Score:", bleu_score)
    print("Exact Match Score:", exact_match_score)
    print("Levenshtein Score:", levenshtein_score)
    print("Validity Score:", validity_score)
    print('=' * 100)
    print(result_df)