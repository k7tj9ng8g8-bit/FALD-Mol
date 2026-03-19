from absl import logging
import numpy as np
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
from rdkit import RDLogger
import re
import torch.nn.functional as F

RDLogger.DisableLog('rdApp.*')



@torch.no_grad()
def AE_SMILES_encoder_noprops(sm, ae_model):
    if sm[0][:5] == "[CLS]":    sm = [s[5:] for s in sm]
    text_input = ae_model.tokenizer(sm).to(ae_model.device)
    text_input_ids = text_input
    text_attention_mask = torch.where(text_input_ids == 0, 0, 1).to(text_input.device)
    if hasattr(ae_model.text_encoder2, 'bert'):
        output = ae_model.text_encoder2.bert(text_input_ids, attention_mask=text_attention_mask, return_dict=True, mode='text').last_hidden_state
    else:
        output = ae_model.text_encoder2(text_input_ids, attention_mask=text_attention_mask, return_dict=True).last_hidden_state

    if hasattr(ae_model, 'encode_prefix'):
        output = ae_model.encode_prefix(output)
        if ae_model.output_dim*2 == output.size(-1):
            mean, logvar = torch.chunk(output, 2, dim=-1)
            logvar = torch.clamp(logvar, -30.0, 20.0)
            std = torch.exp(0.5 * logvar)
            output = mean + std * torch.randn_like(mean)
    return output




def AE_SMILES_encoder(sm, ae_model, properties=None, deterministic=False):


    if sm[0][:5] == "[CLS]":    sm = [s[5:] for s in sm]


    text_input_ids = ae_model.tokenizer(sm, truncation='max_len').to(ae_model.device)

    text_attention_mask = torch.where(
        text_input_ids == ae_model.tokenizer.pad_token_id,
        torch.tensor(0, dtype=torch.long, device=ae_model.device),
        torch.tensor(1, dtype=torch.long, device=ae_model.device)
    )


    with torch.no_grad():
        text_embeds = ae_model.text_encoder2(
            text_input_ids,
            attention_mask=text_attention_mask,
            return_dict=True
        ).last_hidden_state  # [B, 127, bert_hidden_dim]


    z_smiles = ae_model.encode_prefix(text_embeds)  # [B, 127, 64]
    z_smiles_norm = F.normalize(z_smiles, dim=-1)


    if properties is not None:

        properties = properties.to(ae_model.device).float()
        properties = properties.squeeze(1)  # [B, N]


        property_feature = ae_model.property_embed(properties)  # [B, 768]
        property_feature = property_feature.unsqueeze(1)  # [B, 1, 768]


        batch_size = properties.shape[0]
        properties_input = torch.cat([
            ae_model.property_cls.expand(batch_size, -1, -1),  # [B, 1, 768]
            property_feature
        ], dim=1)  # [B, 2, 768]

        # 步骤3：属性编码器提取特征
        prop_atts = torch.ones(
            properties_input.size()[:-1], dtype=torch.long, device=ae_model.device
        )  # [B, 2]
        prop_embeds = ae_model.property_encoder(
            inputs_embeds=properties_input,
            attention_mask=prop_atts,
            return_dict=True
        ).last_hidden_state  # [B, 2, 768]


        z_prop = ae_model.property_proj(prop_embeds[:, 0, :])  # [B, 64]
        z_prop_norm = F.normalize(z_prop, dim=-1)


        z_prop_broadcast = z_prop_norm.unsqueeze(1).repeat(1, z_smiles.shape[1], 1)  # [B, 127, 64]
        z_concat = torch.cat([0.8 * z_smiles_norm, 0.2 * z_prop_broadcast], dim=-1)  # [B, 127, 128]
        z_fused = ae_model.fuse_proj(z_concat)  # [B, 127, 64]
    else:
        z_fused = z_smiles_norm


    latent = z_fused.permute(0, 2, 1).unsqueeze(-1)  # [B, 64, 127, 1]

    return latent


@torch.no_grad()
def generate(model, image_embeds, text, stochastic=True, k=None):
    """
    适配VAE解码器的单token生成函数：与VAE-validator的自回归逻辑对齐
    """
    # 生成文本attention_mask（与VAE-validator一致）
    text_atts = torch.where(
        text == model.tokenizer.pad_token_id,
        torch.tensor(0, dtype=torch.long, device=model.device),
        torch.tensor(1, dtype=torch.long, device=model.device)
    )

    # 潜特征映射（与VAE-validator.generate_smiles一致）
    if hasattr(model, 'decode_prefix'):
        image_embeds = model.decode_prefix(image_embeds)  # [B, seq_len, 768]

    # 调用VAE的text_encoder生成logits（与VAE-validator完全一致）
    token_output = model.text_encoder(
        text,
        attention_mask=text_atts,
        encoder_hidden_states=image_embeds[:, :text.shape[1], :],  # 严格对齐文本长度
        return_dict=True,
        is_decoder=True,
        return_logits=True,
    )[:, -1, :]  # 取最后一个token的logits

    # 采样逻辑（保持原有，但确保与VAE-validator的贪心解码兼容）
    if k is not None:
        p = F.softmax(token_output, dim=-1)
        if stochastic:
            output = torch.multinomial(p, num_samples=k, replacement=False)
            return torch.log(torch.stack([p[i][output[i]] for i in range(output.size(0))])), output
        else:
            values, indices = torch.topk(p, k=k, dim=-1)
            return torch.log(values), indices
    else:
        if stochastic:
            p = F.softmax(token_output, dim=-1)
            m = Categorical(p)
            token_id = m.sample().unsqueeze(1)  # [B, 1]
        else:
            token_id = torch.argmax(token_output, dim=-1).unsqueeze(1)  # 贪心解码（与VAE-validator一致）
        return token_id


@torch.no_grad()
def AE_SMILES_decoder(pv, model, stochastic=False, k=2, max_length=150):
    """
    适配最终VAE的解码函数：修改核心逻辑以匹配VAE-validator.generate_smiles
    1. 修复终止条件：遇到[SEP]立即停止
    2. 增强SMILES清理：移除空格和空字符
    3. 对齐解码后处理：与tokenizer.decode逻辑一致
    """
    model.eval()
    candidate = []
    device = model.device
    tokenizer = model.tokenizer
    sep_id = tokenizer.sep_token_id  # 结束符ID（与VAE-validator一致）

    # 单路径生成（k=1，贪心解码，与VAE-validator保持一致）
    if k == 1:
        # 初始化输入：[CLS] token（与VAE-validator相同的起始符）
        text_input = torch.tensor([tokenizer.cls_token_id]).expand(pv.size(0), 1).to(device)

        for _ in range(max_length):
            # 生成下一个token（使用贪心解码）
            output = generate(model, pv, text_input, stochastic=False)  # [B, 1]
            text_input = torch.cat([text_input, output], dim=-1)

            # 关键修复：遇到[SEP]立即终止当前样本生成（而非等待所有样本）
            # 遍历每个样本，检查是否生成结束符
            for i in range(text_input.size(0)):
                if output[i] == sep_id:
                    # 对已生成结束符的样本，后续不再处理（用PAD填充）
                    text_input[i, text_input.shape[1]:] = tokenizer.pad_token_id

            # 若所有样本都生成了[SEP]，提前终止循环
            if (output == sep_id).all():
                break

        # 解码与清理（完全对齐VAE-validator的后处理）
        for i in range(text_input.size(0)):
            # 1. 解码token序列（使用模型自带tokenizer，确保与训练一致）
            smiles = tokenizer.decode(text_input[i])
            # 2. 处理解码结果（兼容list或单字符串输出）
            if isinstance(smiles, list):
                smiles = smiles[0] if smiles else ""
            # 3. 清理：移除空格、空字符（与VAE-validator完全一致）
            smiles = smiles.strip().replace(' ', '')
            candidate.append(smiles)

    # 多路径生成（k>1，保持原有逻辑但增强清理）
    else:
        for prop_embeds in pv:
            prop_embeds = prop_embeds.unsqueeze(0)  # [1, 127, 64]
            product_input = torch.tensor([tokenizer.cls_token_id]).expand(1, 1).to(device)

            # 第一轮Top-k采样
            values, indices = generate(model, prop_embeds, product_input, stochastic=stochastic, k=k)
            product_input = torch.cat([
                torch.tensor([tokenizer.cls_token_id]).expand(k, 1).to(device),
                indices.squeeze(0).unsqueeze(-1)
            ], dim=-1)  # [k, 2]
            current_p = values.squeeze(0)
            final_output = []

            for _ in range(max_length - 1):
                values, indices = generate(model, prop_embeds, product_input, stochastic=stochastic, k=k)
                k2_p = current_p[:, None] + values  # [k, k]
                product_input_k2 = torch.cat([
                    product_input.unsqueeze(1).repeat(1, k, 1),
                    indices.unsqueeze(-1)
                ], dim=-1)  # [k, k, seq_len+1]

                # 检查结束符（与单路径逻辑一致）
                if sep_id in indices:
                    ends = (indices == sep_id).nonzero(as_tuple=False)
                    for e in ends:
                        p_val = k2_p[e[0], e[1]].cpu().item()
                        final_output.append((p_val, product_input_k2[e[0], e[1]]))
                        k2_p[e[0], e[1]] = -1e5
                    if len(final_output) >= k:
                        break

                # 筛选Top-k路径
                current_p, top_indices = torch.topk(k2_p.flatten(), k)
                next_indices = torch.from_numpy(np.array(np.unravel_index(top_indices.cpu().numpy(), k2_p.shape))).T
                product_input = torch.stack([product_input_k2[i[0], i[1]] for i in next_indices], dim=0)

            # 选择最佳路径并清理（与单路径清理逻辑一致）
            if final_output:
                final_output = sorted(final_output, key=lambda x: x[0], reverse=True)[:1]
                smiles = tokenizer.decode(final_output[0][1])[0] if isinstance(tokenizer.decode(final_output[0][1]),
                                                                               list) else tokenizer.decode(
                    final_output[0][1])
            else:
                smiles = tokenizer.decode(product_input[0])[0] if isinstance(tokenizer.decode(product_input[0]),
                                                                             list) else tokenizer.decode(
                    product_input[0])
            # 关键：增加清理步骤（与VAE-validator一致）
            smiles = smiles.strip().replace(' ', '')
            candidate.append(smiles)

    return candidate











@torch.no_grad()
def molT5_encoder(descriptions, molt5, molt5_tokenizer, description_length, device):
    tokenized = molt5_tokenizer(descriptions, padding='max_length', truncation=True, max_length=description_length, return_tensors="pt").to(device)
    encoder_outputs = molt5.encoder(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask, return_dict=True).last_hidden_state
    return encoder_outputs, tokenized.attention_mask


def get_validity(smiles):
    from rdkit import Chem
    v = []
    for l in smiles:
        try:
            if l == "":
                continue
            s = Chem.MolToSmiles(Chem.MolFromSmiles(l), isomericSmiles=False)
            v.append(s)
        except:
            continue
    u = list(set(v))
    if len(u) == 0:
        return 0., 0.
    return len(v) / len(smiles)


alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'


def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences


def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]  # center crop
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)  # resize the center crop from [crop, crop] to [width, height]

    return np.array(img).astype(np.uint8)


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def drawRoundRec(draw, color, x, y, w, h, r):
    drawObject = draw

    '''Rounds'''
    drawObject.ellipse((x, y, x + r, y + r), fill=color)
    drawObject.ellipse((x + w - r, y, x + w, y + r), fill=color)
    drawObject.ellipse((x, y + h - r, x + r, y + h), fill=color)
    drawObject.ellipse((x + w - r, y + h - r, x + w, y + h), fill=color)

    '''rec.s'''
    drawObject.rectangle((x + r / 2, y, x + w - (r / 2), y + h), fill=color)
    drawObject.rectangle((x, y + r / 2, x + w, y + h - (r / 2)), fill=color)

class regexTokenizer():
    def __init__(self,vocab_path='./vocab_bpe_300_sc.txt',max_len=127):
        with open(vocab_path,'r') as f:
            x = f.readlines()
            x = [xx.replace('##', '') for xx in x]
            x2 = x.copy()
        x2.sort(key=len, reverse=True)
        pattern = "("+"|".join(re.escape(token).strip()[:-1] for token in x2)+")"
        self.rg = re.compile(pattern)

        self.idtotok  = { cnt:i.strip() for cnt,i in enumerate(x)}
        self.vocab_size = len(self.idtotok) #SOS, EOS, pad
        self.toktoid = { v:k for k,v in self.idtotok.items()}
        self.max_len = max_len
        self.cls_token_id = self.toktoid['[CLS]']
        self.sep_token_id = self.toktoid['[SEP]']
        self.pad_token_id = self.toktoid['[PAD]']

    def decode_one(self, iter):
        if self.sep_token_id in iter:   iter = iter[:(iter == self.sep_token_id).nonzero(as_tuple=True)[0][0].item()]
        # return "".join([self.ind2Letter(i) for i in iter]).replace('[SOS]','').replace('[EOS]','').replace('[PAD]','')
        return "".join([self.idtotok[i.item()] for i in iter[1:]])

    def decode(self,ids:torch.tensor):
        if len(ids.shape)==1:
            return [self.decode_one(ids)]
        else:
            smiles  = []
            for i in ids:
                smiles.append(self.decode_one(i))
            return smiles
    def __len__(self):
        return self.vocab_size

    def __call__(self,smis:list, truncation='max_len'):
        tensors = []
        lengths = []
        if type(smis) is str:
            smis = [smis]
        for i in smis:
            length, tensor = self.encode_one(i)
            tensors.append(tensor)
            lengths.append(length)
        output = torch.concat(tensors,dim=0)
        if truncation == 'max_len':
            return output
        elif truncation == 'longest':
            return output[:, :max(lengths)]
        else:
            raise ValueError('truncation should be either max_len or longest')

    def encode_one(self, smi):
        smi = '[CLS]' + smi + '[SEP]'
        res = [self.toktoid[i] for i in self.rg.findall(smi)]
        token_length = len(res)
        if token_length < self.max_len:
            res += [self.pad_token_id]*(self.max_len-len(res))
        else:
            res = res[:self.max_len]
            # res[-1] = self.sep_token_id
        return token_length, torch.LongTensor([res])

class ExtraMolecularFeatures:
    def __init__(self, max_weight, atom_weights):
        self.weight = WeightFeature(max_weight=max_weight, atom_weights=atom_weights)

    def __call__(self, node_types):
        weight = self.weight(node_types)  # (bs, 1)
        return weight


class WeightFeature:
    def __init__(self, max_weight, atom_weights):
        self.max_weight = max_weight
        # self.atom_weight_list = torch.Tensor(list(atom_weights.values()))
        self.atom_weight_list = torch.Tensor(atom_weights)

    def __call__(self, node_types):
        X = torch.argmax(node_types, dim=-1)  # (bs, n)
        X_weights = self.atom_weight_list[X]  # (bs, n)
        return X_weights.sum(dim=-1).unsqueeze(-1).type_as(node_types) / self.max_weight  # (bs, 1)


class ExtraFeatures:
    def __init__(self, extra_features_type, max_n_nodes):
        self.max_n_nodes = max_n_nodes
        self.ncycles = NodeCycleFeatures()
        self.features_type = extra_features_type
        if extra_features_type in ['eigenvalues', 'all']:
            self.eigenfeatures = EigenFeatures(mode=extra_features_type)

    def __call__(self, E_t, node_mask):
        n = node_mask.sum(dim=1).unsqueeze(1) / self.max_n_nodes
        x_cycles, y_cycles = self.ncycles(E_t, node_mask)  # (bs, n_cycles)

        if self.features_type == 'cycles':
            E = E_t
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            return torch.hstack((n, y_cycles))

        elif self.features_type == 'eigenvalues':
            eigenfeatures = self.eigenfeatures(E_t, node_mask)
            E = E_t
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            n_components, batched_eigenvalues = eigenfeatures  # (bs, 1), (bs, 10)
            return torch.hstack((n, n_components, batched_eigenvalues))

        elif self.features_type == 'all':
            eigenfeatures = self.eigenfeatures(E_t, node_mask)
            E = E_t
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            n_components, batched_eigenvalues, nonlcc_indicator, k_lowest_eigvec = eigenfeatures  # (bs, 1), (bs, 10),
            # (bs, n, 1), (bs, n, 2)
            if n_components.dim() == 1:
                n_components = n_components.unsqueeze(1)  # 将1维张量扩展为 [batch_size, 1]
            if y_cycles.dim() == 1:
                y_cycles = y_cycles.unsqueeze(1)  # 将1维张量扩展为 [batch_size, 1]
            if batched_eigenvalues.dim() == 1:
                batched_eigenvalues = batched_eigenvalues.unsqueeze(1)  # 将1维张量扩展为 [batch_size, 1]
            return torch.hstack((n, y_cycles, n_components, batched_eigenvalues))

        else:
            raise ValueError(f"Features type {self.features_type} not implemented")


class NodeCycleFeatures:
    def __init__(self):
        self.kcycles = KNodeCycles()

    def __call__(self, E_t, node_mask):
        adj_matrix = E_t[..., 1:].sum(dim=-1).float()

        x_cycles, y_cycles = self.kcycles.k_cycles(adj_matrix=adj_matrix)  # (bs, n_cycles)
        x_cycles = x_cycles.type_as(adj_matrix) * node_mask.unsqueeze(-1)
        # Avoid large values when the graph is dense
        x_cycles = x_cycles / 10
        y_cycles = y_cycles / 10
        x_cycles[x_cycles > 1] = 1
        y_cycles[y_cycles > 1] = 1
        return x_cycles, y_cycles


class EigenFeatures:
    """
    Code taken from : https://github.com/Saro00/DGN/blob/master/models/pytorch/eigen_agg.py
    """

    def __init__(self, mode):
        """ mode: 'eigenvalues' or 'all' """
        self.mode = mode

    def __call__(self, E_t, mask):
        A = E_t[..., 1:].sum(dim=-1).float() * mask.unsqueeze(1) * mask.unsqueeze(2)
        L = compute_laplacian(A, normalize=False)
        mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
        mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
        L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

        if self.mode == 'eigenvalues':
            eigvals = torch.linalg.eigvalsh(L)  # bs, n
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)

            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)
            return n_connected_comp.type_as(A), batch_eigenvalues.type_as(A)

        elif self.mode == 'all':
            eigvals, eigvectors = torch.linalg.eigh(L)
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)
            eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)
            # Retrieve eigenvalues features
            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)

            # Retrieve eigenvectors features
            nonlcc_indicator, k_lowest_eigenvector = get_eigenvectors_features(vectors=eigvectors,
                                                                               node_mask=mask,
                                                                               n_connected=n_connected_comp)
            return n_connected_comp, batch_eigenvalues, nonlcc_indicator, k_lowest_eigenvector
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")


def compute_laplacian(adjacency, normalize: bool):
    """
    adjacency : batched adjacency matrix (bs, n, n)
    normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    diag = torch.sum(adjacency, dim=-1)  # (bs, n)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)  # Degree matrix      # (bs, n, n)
    combinatorial = D - adjacency  # (bs, n, n)

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2

    diag0 = diag.clone()
    diag[diag == 0] = 1e-12

    diag_norm = 1 / torch.sqrt(diag)  # (bs, n)
    D_norm = torch.diag_embed(diag_norm)  # (bs, n, n)
    L = torch.eye(n).unsqueeze(0) - D_norm @ adjacency @ D_norm
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2


def get_eigenvalues_features(eigenvalues, k=5):
    """
    values : eigenvalues -- (bs, n)
    node_mask: (bs, n)
    k: num of non zero eigenvalues to keep
    """
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 1e-5).sum(dim=-1)
    assert (n_connected_components > 0).all(), (n_connected_components, ev)

    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack((eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues)))
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(0) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev


def get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    """
    vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
    returns:
        not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
        k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
    Warning: this function does not exactly return what is desired, the lcc might not be exactly the returned vector.
    """
    bs, n = vectors.size(0), vectors.size(1)

    # Create an indicator for the nodes outside the largest connected components
    k0 = min(n, 5)
    first_evs = vectors[:, :, :k0]  # bs, n, k0
    quantized = torch.round(first_evs * 1000) / 1000  # bs, n, k0
    random_mask = (50 * torch.ones((bs, n, k0)).type_as(vectors)) * (~node_mask.unsqueeze(-1))  # bs, n, k0
    min_batched = torch.min(quantized + random_mask, dim=1).values.unsqueeze(1)  # bs, 1, k0
    max_batched = torch.max(quantized - random_mask, dim=1).values.unsqueeze(1)  # bs, 1, k0
    nonzero_mask = quantized.abs() >= 1e-5
    is_min = (quantized == min_batched) * nonzero_mask * node_mask.unsqueeze(2)  # bs, n, k0
    is_max = (quantized == max_batched) * nonzero_mask * node_mask.unsqueeze(2)  # bs, n, k0
    is_other = (quantized != min_batched) * (quantized != max_batched) * nonzero_mask * node_mask.unsqueeze(2)

    all_masks = torch.cat((is_min.unsqueeze(-1), is_max.unsqueeze(-1), is_other.unsqueeze(-1)), dim=3)  # bs, n, k0, 3
    all_masks = all_masks.flatten(start_dim=-2)  # bs, n, k0 x 3
    counts = torch.sum(all_masks, dim=1)  # bs, k0 x 3

    argmax_counts = torch.argmax(counts, dim=1)  # bs
    lcc_indicator = all_masks[torch.arange(bs), :, argmax_counts]  # bs, n
    not_lcc_indicator = ((~lcc_indicator).float() * node_mask).unsqueeze(2)

    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = torch.cat((vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2)  # bs, n , n + to_extend
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(0) + n_connected.unsqueeze(2)  # bs, 1, k
    indices = indices.expand(-1, n, -1)  # bs, n, k
    first_k_ev = torch.gather(vectors, dim=2, index=indices)  # bs, n, k
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev


def batch_trace(X):
    """
    Expect a matrix of shape B N N, returns the trace in shape B
    :param X:
    :return:
    """
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace


def batch_diagonal(X):
    """
    Extracts the diagonal from the last two dims of a tensor
    :param X:
    :return:
    """
    return torch.diagonal(X, dim1=-2, dim2=-1)


class KNodeCycles:
    """ Builds cycle counts for each node in a graph.
    """

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.float()
        self.d = self.adj_matrix.sum(dim=-1)
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.float()
        self.k6_matrix = self.k5_matrix @ self.adj_matrix.float()

    def k3_cycle(self):
        """ tr(A ** 3). """
        c3 = batch_diagonal(self.k3_matrix)
        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = diag_a4 - self.d * (self.d - 1) - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(-1).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5_matrix)
        triangles = batch_diagonal(self.k3_matrix)

        c5 = diag_a5 - 2 * triangles * self.d - (self.adj_matrix @ triangles.unsqueeze(-1)).sum(dim=-1) + triangles
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(-1).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(self.k3_matrix ** 2)
        term3_t = torch.sum(self.adj_matrix * self.k2_matrix.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)
        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (term_1_t - 3 * term_2_t + 9 * term3_t - 6 * term_4_t + 6 * term_5_t - 4 * term_6_t + 4 * term_7_t +
                3 * term8_t - 12 * term9_t + 4 * term10_t)
        return None, (c6_t / 12).unsqueeze(-1).float()

    def k_cycles(self, adj_matrix, verbose=False):
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k3x, k3y = self.k3_cycle()
        assert (k3x >= -0.1).all()

        k4x, k4y = self.k4_cycle()
        assert (k4x >= -0.1).all()

        k5x, k5y = self.k5_cycle()
        assert (k5x >= -0.1).all(), k5x

        _, k6y = self.k6_cycle()
        assert (k6y >= -0.1).all()

        kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)
        kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)
        return kcyclesx, kcyclesy