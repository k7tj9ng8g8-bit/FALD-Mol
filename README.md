# FALD-Mol
This package contains deep learning models and related scripts to run FALD-Diff.  
Due to the large size of the model parameters and datasets, we have hosted them at this [location](https://pan.baidu.com/s/149O8FDSvoM3Wj6AHyRREPw?pwd=7mcg).

## Abstract
Text-guided molecule generation offers an intuitive and flexible paradigm for controllable drug design.
However, existing methods frequently suffer from semantic ambiguity in natural language descriptions
and inadequate fusion of molecular structural and physicochemical information. To address these lim
itations, we propose FALD-Mol, a multimodal latent diffusion framework for accurate and controllable
molecule generation under text guidance. Specifically, a SMILES-property joint encoding variational
autoencoder is designed, which integrates molecular structural representations with physicochemical
property features into a unified latent space. This design inherently ensures chemical validity while
enforcing property consistency throughout the generation process. To further alleviate the uncertainty
and vagueness of textual descriptions for molecules, a cross-modal alignment mechanism that aligns tex
tual embeddings with two-dimensional molecular structural features, enabling precise and semantically
grounded conditional guidance. Leveraging the aligned text and property conditions, a latent-space dif
fusion model generates novel molecular representations. Extensive experiments on the ChEBI-20 and
PubChem datasets demonstrate that FALD-Mol consistently outperforms state-of-the-art text-guided
molecule generation models across multiple evaluation metrics. Complementary ablation studies, prop
erty matching analyses, and customized text generation experiments further validate the effectiveness,
controllability, and generalization capability of the proposed framework, underscoring its strong potential
for practical applications in drug discovery and molecular design. 


## Train CPL-Diff
Create conda environment using requirements.yaml.
The training code for FALD-Diff is available at train.py.
The validation code is provided in t2m.py.

