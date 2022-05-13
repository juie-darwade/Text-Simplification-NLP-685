# Text-Simplification-NLP-685

We explored the task of Sentence Simplification using Phrase-BERT and GPT-2 tranformer pairing. We also analysed how backtranslation influenced the text simplification progress in a positive effect.
Refer to the requirements.txt for compatible library versions.


Model training:
python3 -u "Project/run.py" train --base_path "Project/" --src_train "dataset/src_train.txt" --src_valid "dataset/src_valid.txt" --tgt_train "dataset/tgt_train.txt" --tgt_valid "dataset/tgt_valid.txt" --seed 540

Model decoding:
python3 -u "Project/run.py" decode --base_path "Project/" --src_file "dataset/src_file.txt" --output "dataset/decoded.txt" --best_model "checkpoint/model_ckpt.pt"

Model testing:
python3 -u "Project/run.py" test --base_path "Project/" --src_file "dataset/src_file.txt" --output "dataset/decoded.txt" --best_model "checkpoint/model_ckpt.pt"





