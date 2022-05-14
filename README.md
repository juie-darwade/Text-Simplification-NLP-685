# Text-Simplification-NLP-685

We explored the task of Sentence Simplification using Phrase-BERT and GPT-2 tranformer pairing. We also analysed how backtranslation influenced the text simplification progress in a positive effect.
Refer to the requirements.txt for compatible library versions.


Model training:
python3 -u "Project/run.py" train --base_path "Project/" --src_train "dataset/src_train.txt" --src_valid "dataset/src_valid.txt" --tgt_train "dataset/tgt_train.txt" --tgt_valid "dataset/tgt_valid.txt" --seed 540





