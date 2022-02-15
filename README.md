# Low-resource-Machine-Translation

This repository contains the code for the project relative to the course `Deep Natural Language Processing`. The goal of the project is to replicate the experiments performed by [Dabre et al.](https://aclanthology.org/D19-1146.pdf) on low-resource machine translation. In particular, starting from a machine translation model pretrained on a large dataset, we finetune it on a low-resource language. Then, two extensions are implemented:
* The same approach is tested on translation from Vietnamese to English and, then, from English to the other low-resource languages
* The same approach is tested on a different dataset and a different language pair

## Implementation details

### Libraries detail
```
python 3.7.12
transformers 4.16.2
datasets 1.18.3
metrics 0.3.3
sacrebleu 2.0.0
torch 1.10.0 + cu111
```

### Multilingual finetuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FSdk0dYF13gYgiBFm_LhdIydEm9n-mfW?usp=sharing)

The initial model chosen for the task is [MarianMT](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh), a transformer-based model pretrained on a large English-Chinese corpus. The model is finetuned on four low-resource languages from the [ALT dataset](https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/) (Vietnamese, Indonesian, Khmer, and Filipino). The finetuning is performed using the [Huggingface 🤗 Transformers library](https://huggingface.co/docs/transformers/index) and relies on [trainer API](https://huggingface.co/docs/transformers/training). The code for model finetuning is available in the [finetuning_en_target](finetuning_en_target.ipynb) notebook.

### Changing direction of translation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/177UvaF0oq9p28fAZpD9vgbMKIkDOFrzK?usp=sharing)

For this task, the initial model is [MarianMT pretrained on a Chinese-English](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en) corpus. The model is finetuned on the Vietnamese-Chinese task, then the English sentences are translated to another low-resource language using the models finetuned in the previous part. The results are assessed by computing the BLEU score. The code for Vietnamese-English finetuning is available in the [finetuning_vi_en](finetuning_vi_en.ipynb) notebook, whereas the code to translate between two low-resource languages using pretrained models is available in the [translate_vi_target](translate_vi_target.ipynb) notebook.


### Testing on a different dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nuyhqmoJMN13Yxoe_IOoN8dL9NUbzRBm?usp=sharing)

In this task, the approach is experimented on the [WikiMatrix](https://github.com/facebookresearch/LASER/tree/main/tasks/WikiMatrix) dataset, which consists on many parallel sentences mined from Wikipedia using a distance metric to predict alignments. The selected language pair is English-Kazakh because it contains the same number of samples as those in the previous sections. The starting model is [MarianMT pretrained on English-Turkish](https://huggingface.co/Helsinki-NLP/opus-tatoeba-en-tr), and results are evaluated using the BLEU score. The code for model finetuning is available in the [finetuning_en_kazakh](finetuning_en_kazakh.ipynb) notebook.

## Model usage
Some of the models finetuned within this project are available on the [Huggingface hub](https://huggingface.co/CLAck), so they can be downloaded and used. An example of usage is provided in the following.
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Download the pretrained model for English-Vietnamese available on the hub
model = AutoModelForSeq2SeqLM.from_pretrained("CLAck/en-vi")
tokenizer = AutoTokenizer.from_pretrained("CLAck/en-vi")

input_sentence = "The cat sat on the mat"
translated = model.generate(**tokenizer(input_sentence, return_tensors="pt", padding=True))
output_sentence = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

```

