from model.seq2seq import Seq2SeqTransformer
from utils.DataLoader import get_dataloader
from utils.train_evaluate import train_epoch, evaluate
import torch
import torch.nn as nn
from timeit import default_timer as timer
from utils.metrics import get_blue4_score, get_meteor_score
from sklearn.model_selection import ParameterGrid
from utils.predict import predict
from tqdm import tqdm
from rouge import Rouge
import numpy as np
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_IDX = 1


def train(embedding_size=256, num_layers=8, lr=0.0001, i=5):
    BATCH_SIZE = 64
    train_dataloader, val_dataloader, vocab, val_src, val_tgt_ = get_dataloader("./data/train.csv", batch_size=BATCH_SIZE, val_size=0.2)
    torch.manual_seed(42)
    SRC_VOCAB_SIZE = len(vocab)
    TGT_VOCAB_SIZE = len(vocab)
    # VOCAB_SIZE = len(vocab)
    EMB_SIZE = embedding_size
    NHEAD = 8
    FFN_HID_DIM = 4*embedding_size
    
    NUM_ENCODER_LAYERS = num_layers
    NUM_DECODER_LAYERS = num_layers
    EPOCHS = 30

    NUM_BLUE = 100
    MAX_PATIENCE = 3

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                    NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    # optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)

    val_loss_list = []
    min_val_loss = float("inf")
    best_epoch = 0
    patience = 0

    val_tgt_ = [i for i in val_tgt_]

    print("device: ", DEVICE)
    for epoch in range(1, EPOCHS+1):
        patience += 1
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, train_dataloader, loss_fn)
        end_time = timer()
        val_loss = evaluate(transformer, val_dataloader, loss_fn)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
        if min_val_loss - val_loss > 1e-2:
            best_epoch = epoch
            min_val_loss = val_loss
            patience = 0
            torch.save(transformer.state_dict(), f'best_model_{i}.pth')
        
        val_loss_list.append(val_loss)
        if patience > MAX_PATIENCE:
            print("Early stopping")
            print(f"Best epoch: {best_epoch}")
            print(f"Best val loss: {min_val_loss}")
            break
    # BLUE4
    # transformer.load_state_dict(torch.load('best_model_2.pth'))
    predict_sentence_list = [predict(transformer, i, vocab) for i in tqdm(val_src[:NUM_BLUE])]
    predict_sentence_list = [i.strip().split(" ") for i in predict_sentence_list]

    blue4_score = get_blue4_score(predict_sentence_list, val_tgt_[:NUM_BLUE])
    print(len(predict_sentence_list))
    print("blue4 score: ", blue4_score)

    # ROUGE
    rouge_score = Rouge().get_scores([" ".join(i) for i in predict_sentence_list], [" ".join(i) for i in val_tgt_[:NUM_BLUE]])
    rouge_1 = [score['rouge-1']['f'] for score in rouge_score]
    rouge_2 = [score['rouge-2']['f'] for score in rouge_score]
    rouge_l = [score['rouge-l']['f'] for score in rouge_score]

    mean_rouge_1 = np.mean(rouge_1)
    mean_rouge_2 = np.mean(rouge_2) 
    mean_rouge_l = np.mean(rouge_l)

    print("ROUGE-1 平均分数:", mean_rouge_1)
    print("ROUGE-2 平均分数:", mean_rouge_2)
    print("ROUGE-L 平均分数:", mean_rouge_l)

    # METEOR
    meteor_score_ = get_meteor_score(predict_sentence_list, val_tgt_[:NUM_BLUE])
    print("meteor score: ", meteor_score_)

    return (blue4_score, mean_rouge_1, mean_rouge_2, mean_rouge_l, meteor_score_)

if __name__ == '__main__':
    embedding_size = [128, 256, 512]
    num_layers = [2, 4, 6, 8]
    lr = [0.00001, 0.0001, 0.001, 0.01]

    param_grid = {'embedding_size': embedding_size, 
                'num_layers': num_layers,
                'lr': lr}

    grid = ParameterGrid(param_grid)

    i = 5    
    with open("results.csv", "a") as f:
        for params in grid:
            print(params)
            print("----------------------------------------------------")
            blue4_score, mean_rouge_1, mean_rouge_2, mean_rouge_l, meteor_score_ = train(params['embedding_size'], params['num_layers'], params['lr'], i=5)
            l = [params['embedding_size'], params['num_layers'], params['lr'], f'best_model_{i}.pth', blue4_score, mean_rouge_1, mean_rouge_2, mean_rouge_l, meteor_score_]
            f.write(",".join(map(str, l)) + "\n")
            i += 1
