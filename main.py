from bidict import bidict
import fire
from tqdm import tqdm

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as spstats
import sklearn as sk
import sklearn.metrics as skmetrics

import torch
import torch.nn.functional as F
import torch.optim as optim

import models

FLOAT_DTYPE = np.float32
LONG_DTYPE = np.int64

P_MIN = 0.0001
P_MAX = 0.9999

H_MIN = 15/60/24 # 15 Minutes
H_MAX = 9*30 # 9 Months


def preprocess_df(df):
    df = df.drop(columns=["timestamp","lexeme_string", "learning_language", "ui_language", "session_seen", "session_correct"])
    uid_dict = bidict()
    for i,uid in enumerate(df["user_id"].unique()):
        uid_dict[uid] = i

    lid_dict = bidict()
    for i,lid in enumerate(df["lexeme_id"].unique()):
        lid_dict[lid] = i

    df["user_id"] = np.array(list(map(lambda x: uid_dict[x], df["user_id"])), dtype=np.int64)
    df["lexeme_id"] = np.array(list(map(lambda x: lid_dict[x], df["lexeme_id"])), dtype=np.int64)
    df["delta"] = df["delta"] / (60*60*24)
    df["seen"] = np.sqrt(df["history_seen"])
    df["correct"] = np.sqrt(df["history_correct"])
    df["wrong"] = np.sqrt(df["history_seen"]-df["history_correct"])
    
    df = df.drop(columns=["history_seen","history_correct"])
    return df, uid_dict, lid_dict

def prepare_batch(df, indexes, include_delta=False):
    bdf = df.iloc[indexes]
    
    feat_list = ["seen","correct","wrong"] + (["delta"] if include_delta else [])
    x = bdf[feat_list].to_numpy(dtype=FLOAT_DTYPE)
    
    l = bdf["lexeme_id"].to_numpy(dtype=LONG_DTYPE)
    
    t = bdf["delta"].to_numpy(dtype=FLOAT_DTYPE)[:,np.newaxis]
    
    p = np.clip(bdf["p_recall"].to_numpy(dtype=FLOAT_DTYPE),P_MIN,P_MAX)[:,np.newaxis]
    
    h = np.clip(-t/np.log2(p),H_MIN,H_MAX)
    
    return x, l, t, p, h

model_dict = {
    "hlr": models.HalfLifeRegression,
    "lr": models.LogisticRegression
}

def to_cuda(x):
    return x.cuda()

def main(
        model:str="hlr",
        learning_rate:float = 0.001,
        hl_loss:float = 0.01,
        l2_loss:float = 0.1,
        batch_size:int = 1024,
        num_epochs:int = 10,
        test_split:float = 0.1,
        use_cuda:bool = True,
        supress_tqdm:bool = False,
        ):
    torch.set_printoptions(profile="full")
    logfile = open("log.txt","w")
    use_cuda = use_cuda and torch.cuda.is_available()
    
    df = pd.read_csv("learning_traces.13m.csv")
    df, uid_dict, lid_dict = preprocess_df(df)

    include_delta = model!="hlr"
    num_features = 4 if include_delta else 3
    num_lexemes = 1 + df["lexeme_id"].max()
    
    reg = model_dict[model](num_features=num_features, num_lexemes=num_lexemes, h_min=H_MIN, h_max=H_MAX, p_min=P_MIN, p_max=P_MAX)
    
    if use_cuda:
        reg = reg.cuda()
    
    opt = optim.Adagrad(reg.parameters(),lr=learning_rate, weight_decay=l2_loss)
    
    n = len(df)
    
    df_train, df_test = df[:int(-test_split*n)], df[int(-test_split*n):]
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()
    
    n_train, n_test = map(len,(df_train, df_test))
    
    for e in tqdm(range(num_epochs), total=num_epochs, desc="Epoch ", disable=supress_tqdm):
        #permutation = np.arange(n_train)
        permutation = np.random.permutation(n_train)
        for b in tqdm(range(0,n_train,batch_size), total=n_train/batch_size, desc="Batch ", disable=supress_tqdm):
            opt.zero_grad()
            idx = permutation[b:b+batch_size]
            
            x, l, t, p, h = prepare_batch(df_train, idx, include_delta)
            
            x, l, t, p, h = map(torch.tensor,(x, l, t, p, h))
            if use_cuda:
                x, l, t, p, h = map(to_cuda,(x, l, t, p, h))
            
            pred_p, pred_h = reg(x,l,t)
            
            loss = F.mse_loss(input=pred_p,target=p)
            if hl_loss>0:
                loss += hl_loss*F.mse_loss(input=pred_h,target=h)
            
            loss.backward()
            opt.step()
        #end for
    #end for
    
    reg = reg.cpu()
    
    errors = []
    pred_hs = []
    test_hs = []
    pred_ps = []
    test_ps = []
    permutation = np.arange(n_test)
    logfile.write("TESTING")
    with torch.no_grad():
        for i in tqdm(range(0,n_test,batch_size), total=n_test/batch_size, desc="Test ", disable=supress_tqdm):
            idx = permutation[i:i+batch_size]
            x, l, t, p, h = prepare_batch(df_test, idx, include_delta)
            test_hs.append(h)
            test_ps.append(p[:,0])
            
            # The reason I write these to the log file is that some of these values give nan outputs to the embedding layers
            logfile.write(str(i)+"\n")
            logfile.write(str(x.tolist())+"\n")
            logfile.write(str(l.tolist())+"\n")
            x, l, t, p, h = map(torch.tensor,(x, l, t, p, h))
            pred_p, pred_h = reg(x,l,t)
            
            err = F.l1_loss(input=pred_p,target=p,reduction="none")
            errors.append(err.squeeze().cpu().detach().numpy())
            pred_hs.append(pred_h.cpu().detach().numpy())
            pred_ps.append(pred_p.cpu().detach().numpy()[:,0])
        #end for
    #end no_grad
    errors,pred_hs,test_hs,pred_ps,test_ps = map(np.concatenate,(errors,pred_hs,test_hs,pred_ps,test_ps))
    print("MAE:", np.mean(errors))
    print("AUC:", skmetrics.roc_auc_score(y_true=np.rint(test_ps),y_score=pred_ps))
    print("WRS:", spstats.ranksums(test_ps,pred_ps))
    print("Cor(h):", spstats.spearmanr(pred_hs,test_hs))
#end main

if __name__=="__main__":
    fire.Fire(main)
