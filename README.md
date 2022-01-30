# Low-resource-Machine-Translation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FSdk0dYF13gYgiBFm_LhdIydEm9n-mfW?usp=sharing)

This repository contains the code for the project relative to the course `Deep Natural Language Processing`. The goal of the project is to replicate the experiments performed by [Dabre et al.](https://aclanthology.org/D19-1146.pdf) on low-resource machine translation. In particular, starting from a machine translation model pretrained on a large dataset, we finetune it on a low-resource language. 

## Implementation details
The initial model chosen for the task is MarianMT, a transformer-based model pretrained on a large English-Chinese corpus. The model is finetuned on three low-resource languages from the [ALT dataset](https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/) (Vietnamese, Indonesian and Filipino). The finetuning is performed using the [Huggingface 🤗 Transformers library](https://huggingface.co/docs/transformers/index).

