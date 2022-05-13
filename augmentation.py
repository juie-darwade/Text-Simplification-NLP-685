# -*- coding: utf-8 -*-
"""augmentation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1x3CBiKc25z0KJAqG5EnDztpV3zOLEAF-
"""

import sys
sys.path.append('/content/drive/MyDrive/685')
!cd '/content/drive/MyDrive/685'

!pip install googletrans==4.0.0-rc1
!pip install transformers
!pip install tokenizer

import googletrans
from googletrans import Translator
translator = Translator()

NUM_EXAMPLES = 2000
train_examples = []
train_targets = []
src_file_path = '/content/drive/MyDrive/685/dataset/src_train.txt'
tgt_file_path = '/content/drive/MyDrive/685/dataset/tgt_train.txt'

with open(src_file_path, 'r', encoding="utf8") as f:
    sents = [next(f) for x in range(NUM_EXAMPLES)]
    for s in sents:
        train_examples.append(s.strip())

with open(tgt_file_path, 'r', encoding="utf8") as f:
    sents = [next(f) for x in range(NUM_EXAMPLES)]
    for s in sents:
        train_targets.append(s.strip())

train_examples_augmented = []

for example, tgt in zip(train_examples, train_targets):
    train_examples_augmented.append((example,tgt))
    lang_one = ['af','vi','ko','mr']
    lang_two = ['vi','nl','fr','ro']
    for i in range(len(lang_one)):
      translated = translator.translate(tgt,src='en', dest=lang_one[i])
      paraphrase = translator.translate(translated.text, src=lang_one[i], dest=lang_two[i])
      paraphrase = translator.translate(paraphrase.text, src=lang_two[i], dest='en')
      train_examples_augmented.append((example, paraphrase.text))

output_dir = "/content/drive/MyDrive/685/dataset/bt"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
