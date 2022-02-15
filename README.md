# Low-resource-Machine-Translation

This repository contains the code for the project relative to the course `Deep Natural Language Processing`. The goal of the project is to replicate the experiments performed by [Dabre et al.](https://aclanthology.org/D19-1146.pdf) on low-resource machine translation. In particular, starting from a machine translation model pretrained on a large dataset, we finetune it on a low-resource language. Then, two extensions are implemented:
* The same approach is tested on translation from Vietnamese to English and, then, from English to the other low-resource languages
* The same approach is tested on a different dataset and a different language pair

## Implementation details

### Multilingual finetuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FSdk0dYF13gYgiBFm_LhdIydEm9n-mfW?usp=sharing)

The initial model chosen for the task is MarianMT, a transformer-based model pretrained on a large English-Chinese corpus. The model is finetuned on three low-resource languages from the [ALT dataset](https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/) (Vietnamese, Indonesian and Filipino). The finetuning is performed using the [Huggingface 🤗 Transformers library](https://huggingface.co/docs/transformers/index). The code for model finetuning and evaluation is available in the [finetuning_en_target](finetuning_en_target.ipynb) notebook.

### Changing direction of translation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/177UvaF0oq9p28fAZpD9vgbMKIkDOFrzK?usp=sharing)

For this task, the initial model is MarianMT pretrained on a Chinese-English corpus. The model is finetuned on the Vietnamese-Chinese task, then the English sentences are translated to another low-resource language using the models finetuned in the previous part. The results are assessed by computing the BLEU score. The code for Vietnamese-English finetuning is available in the [finetuning_vi_en](finetuning_vi_en.ipynb) notebook, whereas the code for translation among low-resource languages using pretrained models is available in the [translate_vi_target](translate_vi_target.ipynb) notebook.

### Testing on a different dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nuyhqmoJMN13Yxoe_IOoN8dL9NUbzRBm?usp=sharing)

In this task, the approach is experimented on the [WikiMatrix](https://github.com/facebookresearch/LASER/tree/main/tasks/WikiMatrix) dataset, which consists on many parallel sentences mined from Wikipedia using a distance metric to predict alignments. The selected language pair is English-Kazakh because it contains the same number of samples as those in the previous sections. The starting model is MarianMT pretrained on English-Turkish, and results are evaluated using the BLEU score. The code for model finetuning is available in the [finetuning_en_kazakh](finetuning_en_kazakh.ipynb) notebook.