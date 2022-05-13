import click
import pickle
import warnings
warnings.filterwarnings("ignore")
import re
import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, Subset
from data import WikiDataset
from tokenizer import Tokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from transformers import EncoderDecoderModel, BertConfig, EncoderDecoderConfig, GPT2Tokenizer
from datasets import list_datasets, list_metrics, load_dataset, load_metric
import time
import tqdm
import logging
import gc
import shutil
import sari

TRAIN_BATCH_SIZE = 4
N_EPOCH = 1
max_token_len = 512
LOG_EVERY = 1000

logging.basicConfig(filename="log_file.log", level=logging.INFO, 
                format="%(asctime)s:%(levelname)s: %(message)s")
CONTEXT_SETTINGS = dict(help_option_names = ['-h', '--help'])

model = EncoderDecoderModel.from_encoder_decoder_pretrained('whaleloops/phrase-bert', 'gpt2')
model.decoder.config.use_cache = False
tokenizer = Tokenizer(max_token_len)
model.config.decoder_start_token_id = tokenizer.gpt2_tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.gpt2_tokenizer.eos_token_id
model.config.max_length = max_token_len
model.config.no_repeat_ngram_size = 3
model.config.pad_token_id = tokenizer.gpt2_tokenizer.eos_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} as device")
model.to(device)

def collate_fn(batch):
    data_list, label_list, ref_list = [], [], []
    for _data, _label, _ref in batch:
        data_list.append(_data)
        label_list.append(_label)
        ref_list.append(_ref)
    return data_list, label_list, ref_list

def compute_bleu_score(logits, labels):
    refs = Tokenizer.get_sent_tokens(labels)
    weights = (1.0/2.0, 1.0/2.0, )
    score = corpus_bleu(refs, logits.tolist(), smoothing_function=SmoothingFunction(epsilon=1e-10).method1, weights=weights)
    return score

def compute_sari(norm, pred_tensor, ref):
    pred = tokenizer.decode_sent_tokens(pred_tensor)
    score = 0
    for step, item in enumerate(ref):
        score += sari.SARIsent(norm[step], pred[step], item)
    return score/TRAIN_BATCH_SIZE

def evaluate(data_loader, e_loss):
    was_training = model.training
    model.eval()
    eval_loss = e_loss
    bleu_score = 0
    sari_score = 0
    softmax = nn.LogSoftmax(dim = -1)
    
    
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            src_tensors, src_attn_tensors, tgt_tensors, tgt_attn_tensors, labels = tokenizer.encode_batch(batch)
            loss, logits = model(input_ids = src_tensors.to(device), 
                            decoder_input_ids = tgt_tensors.to(device),
                            attention_mask = src_attn_tensors.to(device),
                            decoder_attention_mask = tgt_attn_tensors.to(device),
                            labels = labels.to(device))[:2]
            outputs = softmax(logits)
            score = compute_bleu_score(torch.argmax(outputs, dim=-1), batch[1])
            s_score = compute_sari(batch[0], torch.argmax(outputs, dim=-1), batch[1])
            if step == 0:
                eval_loss = loss.item()
                bleu_score = score
                sari_score = s_score
            else:
                eval_loss = (1/2.0)*(eval_loss + loss.item())
                bleu_score = (1/2.0)* (bleu_score+score)
                sari_score = (1/2.0)* (sari_score+s_score)  
    if was_training:
        model.train()
    
    return eval_loss, bleu_score, sari_score

def load_checkpt(checkpt_path, optimizer=None):
    checkpoint = torch.load(checkpt_path)
    if device == "cpu":
        model.load_state_dict(checkpoint["model_state_dict"], map_location=torch.device("cpu"))
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"], map_location=torch.device("cpu"))
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    eval_loss = checkpoint["eval_loss"]
    epoch = checkpoint["epoch"]

    return optimizer, eval_loss, epoch

def save_model_checkpt(state, is_best, check_pt_path, best_model_path):
    f_path = check_pt_path
    torch.save(state, f_path)

    print("Best Torch model saved")
    '''if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)'''

@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version = '1.0.0')
def task():
    ''' This is the documentation of the main file. This is the reference for executing this file.'''
    pass


@task.command()
@click.option("--base_path", default="./", help="Base path to the project destination")
@click.option('--src_train', default="dataset/src_train.txt", help="train source file path")
@click.option('--tgt_train', default="dataset/tgt_train.txt", help="train target file path")
@click.option('--src_valid', default="dataset/src_valid.txt", help="validation source file path")
@click.option('--tgt_valid', default="dataset/tgt_valid.txt", help="validation target file path")
# @click.option('--ref_valid', default="dataset/ref_valid.pkl", help="validation reference file path")
@click.option('--best_model', default="best_model/model.pt", help="best model file path")
@click.option('--checkpoint_path', default="checkpoint/model_ckpt.pt", help=" model check point files path")
@click.option('--seed', default=123, help="manual seed value (default=123)")
def train(**kwargs):
    print("Loading datasets...")
    train_dataset = WikiDataset(kwargs['base_path']+kwargs['src_train'], kwargs['base_path']+kwargs['tgt_train'])
    valid_dataset = WikiDataset(kwargs['base_path']+kwargs['src_valid'], kwargs['base_path']+kwargs['tgt_valid'], ref=False)
    print("Dataset loaded successfully")
    
    train_dl = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    valid_dl = DataLoader(valid_dataset, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
#     train_dl = Subset(train_dataset, np.arange(100))
#     sample_sampler = RandomSampler(train_dl)
#     train_dl = DataLoader(train_dl, sampler=sample_sampler, collate_fn=collate_fn,batch_size=4)
#     valid_dl = Subset(valid_dataset, np.arange(20))

#     sample_sampler = RandomSampler(valid_dl)
#     valid_dl = DataLoader(valid_dl, sampler=sample_sampler, batch_size=4,collate_fn=collate_fn)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=3e-5)
    start_epoch=0
    eval_loss=float('inf')

    '''if os.path.exists(kwargs['base_path']+kwargs["checkpoint_path"]):
        optimizer, eval_loss, start_epoch = load_checkpt(kwargs['base_path']+kwargs["checkpoint_path"], optimizer)
        print(f"Loading model from checkpoint with start epoch: {start_epoch} and loss: {eval_loss}")
        logging.info(f"Model loaded from saved checkpoint with start epoch: {start_epoch} and loss: {eval_loss}")'''
    
    train_model(start_epoch, eval_loss, (train_dl, valid_dl), optimizer, kwargs['base_path']+kwargs["checkpoint_path"], kwargs['base_path']+kwargs["best_model"])

@task.command()
# @click.option("--base_path", default="./", help="Base path to the project destination")
# @click.option('--src_test', default="dataset/src_test.txt", help="test source file path")
# @click.option('--tgt_test', default="dataset/tgt_test.txt", help="test target file path")
def test():
    print("Testing Model module executing...")
    logging.info(f"Test module invoked.")
    #_, _, _ = load_checkpt(kwargs['base_path']+kwargs["best_model"])
    checkpoint = torch.load('checkpoint/model_ckpt.pt')
    print(f"Model loaded.")
    model.eval()
    #load test_dataset for ASSET
    
    dataset = load_dataset('asset', split='validation[:20%]', 'test[:10%]')
    
    test_dl = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate_fn)
    test_start_time = time.time()
    test_loss, bleu_score, sari_score = evaluate(test_dl, 0)
    test_loss = test_loss/TRAIN_BATCH_SIZE
    print(f'Avg. eval loss: {test_loss:.5f} | blue score: {bleu_score} | sari score: {sari_score} | time elapsed: {time.time() - test_start_time}')
    # logging.info(f'Avg. eval loss: {test_loss:.5f} | blue score: {bleu_score} | sari score: {sari_score} | time elapsed: {time.time() - test_start_time}')
    print("Test Complete!")


# @task.command()
# @click.option("--base_path", default="/home/jupyter/Project/", help="Base path to the project destination")
# @click.option('--src_file', default="dataset/src_file.txt", help="test source file path")
# @click.option('--output', default="outputs/decoded.txt", help="file path to save predictions")
def decode_sentences():
    print("Decoding sentences module executing...", flush = True)
    logging.info(f"Decode module invoked.")
    model.eval()
    softmax = nn.LogSoftmax(dim = -1)
#     dataset = WikiDataset(kwargs['base_path']+kwargs['src_file'])
#     print("Dataset src", type(dataset.src))
#     dataset_loader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate_fn, shuffle=False)
    
    f = open("/home/jupyter/Project/dataset/src_file.txt", "r")
    lines = f.readlines()
    
    with torch.no_grad():
        for line in lines:
            src_tensors, src_attn_tensors, tgt_tensors, tgt_attn_tensors, labels = tokenizer.encode_batch(line)
            loss, logits = model(input_ids = src_tensors.to(device), 
                            decoder_input_ids = tgt_tensors.to(device),
                            attention_mask = src_attn_tensors.to(device),
                            decoder_attention_mask = tgt_attn_tensors.to(device),
                            labels = labels.to(device))[:2]
            outputs = softmax(logits)

            pred_tensor = torch.argmax(outputs, dim=-1)
            print(pred_tensor)
            pred = tokenizer.decode_sent_tokens(pred_tensor)
            print("Expected", line)
            print("Pred",pred)
            

    
#         for step, batch in enumerate(data_loader):
#             src_tensors, src_attn_tensors, tgt_tensors, tgt_attn_tensors, labels = tokenizer.encode_batch(batch)
#             loss, logits = model(input_ids = src_tensors.to(device), 
#                             decoder_input_ids = tgt_tensors.to(device),
#                             attention_mask = src_attn_tensors.to(device),
#                             decoder_attention_mask = tgt_attn_tensors.to(device),
#                             labels = labels.to(device))[:2]
#             outputs = softmax(logits)
#             pred_tensor = torch.argmax(outputs, dim=-1)
#             pred = tokenizer.decode_sent_tokens(pred_tensor)
#             print("Expected", batch[1])
#             print("Pred",pred)
    
#     for sent in sent_tensors:
#         with torch.no_grad():
#             predicted = model.generate(sent[0].to(device), attention_mask=sent[1].to(device), decoder_start_token_id=model.config.decoder.decoder_start_token_id)
#             predicted_list.append(predicted.squeeze())
#     output = tokenizer.decode_sent_tokens(predicted_list)
#     with open("/home/jupyter/Project/outputs/baseline_predict.txt", "w") as f:
#         for sent in output:
#             sent_new = re.sub('[^A-Za-z0-9]+',' ',sent).strip()
#             print(sent_new)
#             f.write(sent_new + "\n")
#     print("Output file saved successfully.")
    

@task.command()
@click.option("--base_path", default="./", help="Base path to the project destination")
@click.option('--src_file', default="dataset/src_file_backup.txt", help="test source file path")
@click.option('--output', default="outputs/decoded.txt", help="file path to save predictions")
def decode(**kwargs):
    print("Decoding sentences module executing...", flush = True)
    logging.info(f"Decode module invoked.")
    print(f"Model loaded.")
    checkpoint = torch.load('checkpoint/model_ckpt.pt')
    model.load_state_dict(checkpoint,strict=False)
    model.eval()
    dataset = WikiDataset(kwargs['base_path']+kwargs['src_file'])
    print("Dataset src", type(dataset.src))
    predicted_list = []
    sent_tensors = tokenizer.encode_sent(dataset.src)
    print("Decoding Sentence")
    
    for sent in sent_tensors:
        with torch.no_grad():
            predicted = model.generate(sent[0].to(device), attention_mask=sent[1].to(device), decoder_start_token_id=model.config.decoder.decoder_start_token_id)
            predicted_list.append(predicted.squeeze())
    output = tokenizer.decode_sent_tokens(predicted_list)
    with open("/home/jupyter/Project/outputs/baseline_predict.txt", "w") as f:
        for sent in output:
            sent_new = re.sub('[^A-Za-z0-9]+',' ',sent).strip()
            print(sent_new)
            f.write(sent_new + "\n")
    print("Output file saved successfully.")

def train_model(start_epoch, eval_loss, loaders, optimizer, check_pt_path, best_model_path):
    best_eval_loss = eval_loss
    print("Model training started...")
    for epoch in range(0, N_EPOCH):
        print(f"Epoch {epoch} running...")
        epoch_start_time = time.time()
        epoch_train_loss = 0
        epoch_eval_loss = 0
        model.train()
        for step, batch in enumerate(loaders[0]): #train dataset (src, tgt)
            src_tensors, src_attn_tensors, tgt_tensors, tgt_attn_tensors, labels = tokenizer.encode_batch(batch)
            optimizer.zero_grad()
            model.zero_grad()
            loss = model(input_ids = src_tensors.to(device), 
                            decoder_input_ids = tgt_tensors.to(device),
                            attention_mask = src_attn_tensors.to(device),
                            decoder_attention_mask = tgt_attn_tensors.to(device),
                            labels = labels.to(device))[0]
            if step == 0:
                epoch_train_loss = loss.item()
            else:
                epoch_train_loss = (1/2.0)*(epoch_train_loss + loss.item())

            loss.backward()
            optimizer.step()

            if (step+1) % LOG_EVERY == 0:
                print(f'Epoch: {epoch} | iter: {step+1} | avg. train loss: {epoch_train_loss} | time elapsed: {time.time() - epoch_start_time}')
                logging.info(f'Epoch: {epoch} | iter: {step+1} | avg. train loss: {epoch_train_loss} | time elapsed: {time.time() - epoch_start_time}')

        eval_start_time = time.time()
        epoch_eval_loss, bleu_score, sari_score = evaluate(loaders[1], epoch_eval_loss) #(loaders[1] = valid_dl [src, tgt])
        epoch_eval_loss = epoch_eval_loss/TRAIN_BATCH_SIZE
        print(f'Completed Epoch: {epoch} | avg. eval loss: {epoch_eval_loss:.5f} | blue score: {bleu_score} | Sari score: {sari_score} | time elapsed: {time.time() - eval_start_time}')
        logging.info(f'Completed Epoch: {epoch} | avg. eval loss: {epoch_eval_loss:.5f} | blue score: {bleu_score}| Sari score: {sari_score} | time elapsed: {time.time() - eval_start_time}')

        check_pt = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'eval_loss': epoch_eval_loss,
            'sari_score': sari_score,
            'bleu_score': bleu_score
        }
        check_pt_time = time.time()
        print("Saving Checkpoint.......")
        if epoch_eval_loss < best_eval_loss:
            print("New best model found")
            logging.info(f"New best model found")
            best_eval_loss = epoch_eval_loss
            save_model_checkpt(check_pt, True, check_pt_path, best_model_path)

        print(f"Checkpoint saved successfully with time: {time.time() - check_pt_time}")
        logging.info(f"Checkpoint saved successfully with time: {time.time() - check_pt_time}")

        gc.collect()
        torch.cuda.empty_cache()
        
        print("Completed task")
        decode_sentences()

if __name__ == "__main__":
    
    task()
