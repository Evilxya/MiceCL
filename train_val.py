import time
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import overall_performance
import numpy as np
from transformers import AutoModel
import random
from data_loader import dataset, collate_fn
from torch.utils.data import DataLoader, SequentialSampler

def cal_order(args, train_data, model):
    model.eval()
    train_set = dataset(train_data)
    eval_sampler = SequentialSampler(train_set)
    eval_dataloader = DataLoader(
        train_set,
        sampler=eval_sampler,
        batch_size=args.TRAIN.train_batch_size,
        collate_fn=collate_fn,
    )
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    difficulty = []
    for eval_batch in tqdm(eval_dataloader, desc="Evaluating Difficulties", position=0, leave=True):
        eval_batch = tuple(t.cuda() for t in eval_batch)
        ids_ls, segs_ls, att_mask_ls, ids_rs, att_mask_rs, segs_rs, labels = eval_batch
        # compute logits
        out = model(ids_ls, segs_ls, att_mask_ls, ids_rs, att_mask_rs, segs_rs)
        loss = loss_fn(out, labels)
        difficulty += list(loss.detach().cpu().numpy())

    data = []
    for i, diff in enumerate(difficulty):
        ids_l, segs_l, att_mask_l, ids_r, att_mask_r, segs_r, label = train_data[i]
        data.append([diff, i, label])
    data = sorted(data)
    l = len(data)
    #order = [data[i][1] for i in range(l)]
    order = []
    order0, order1 = [], []
    for i in range(l):
        if data[i][2] == 0:
            order0.append(data[i][1])
        else:
            order1.append(data[i][1])
    l0 = len(order0)
    l1 = len(order1)
    t = l0 // l1
    t = max(1, t)
    print(f't {t}')
    for i in range(l1):
        if (i+1)*t < len(order0):
            for j in range(t):
                order.append(order0[i*t+j])
        order.append(order1[i])
    return order

def cal_steps(order, epochs, batch_size):
    starting_percent = 0.5
    speed = float(1.5)*(1-starting_percent)/epochs
    steps = int(starting_percent*len(order)+batch_size-1) // batch_size
    for epoch in range(epochs):
        percent = starting_percent + speed*epoch
        percent = min(1.0, percent)
        nums = int(percent*len(order))
        steps += (nums+batch_size-1) // batch_size
    return steps
    


def ajust_order(order, epoch, epochs):
    starting_percent = 0.5
    speed = float(1.5)*(1-starting_percent)/epochs
    percent = min(1.0, starting_percent+speed*epoch)
    l = len(order)
    order = order[:int(l*percent)]
    return order

def train(epoch, model, loss_fn, optimizer, train_loader, scheduler=None):
    epoch_start_time = time.time()
    model.train()
    tr_loss = 0  # training loss in current epoch

    # ! training
    for step, batch in enumerate(tqdm(train_loader, desc='Iteration')):
        # unpack batch data
        batch = tuple(t.cuda() for t in batch)
        ids_ls, segs_ls, att_mask_ls, ids_rs, att_mask_rs, segs_rs, labels = batch
        # compute logits
        out = model(ids_ls, segs_ls, att_mask_ls, ids_rs, att_mask_rs, segs_rs)
        # compute loss
        loss = loss_fn(out, labels)
        tr_loss += loss.item()

        # back propagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()  # adjusting learning rate
        optimizer.zero_grad()

    timing = time.time() - epoch_start_time
    cur_lr = optimizer.param_groups[0]["lr"]
    print(f"Timing: {timing}, Epoch: {epoch + 1}, training loss: {tr_loss}, current learning rate {cur_lr}")


def val(model, val_loader):
    # make sure to open the eval mode.
    model.eval()

    # prepare loss function
    loss_fn = nn.CrossEntropyLoss()

    val_loss = 0
    val_preds = []
    val_labels = []
    for batch in tqdm(val_loader, desc='Test'):
        # unpack batch data
        batch = tuple(t.cuda() for t in batch)
        ids_ls, segs_ls, att_mask_ls, ids_rs, att_mask_rs, segs_rs, labels = batch

        with torch.no_grad():
            # compute logits
            out = model(ids_ls, segs_ls, att_mask_ls, ids_rs, att_mask_rs, segs_rs)
            # get the prediction labels
            preds = torch.max(out.data, 1)[1].cpu().numpy().tolist()  # prediction labels [1, batch_size]
            # compute loss
            loss = loss_fn(out, labels)
            val_loss += loss.item()

            labels = labels.cpu().numpy().tolist()  # ground truth labels [1, batch_size]
            val_labels.extend(labels)
            val_preds.extend(preds)
    print(f"val loss: {val_loss}")

    # get overall performance
    val_acc, val_prec, val_recall, val_f1 = overall_performance(val_labels, val_preds)
    return val_acc, val_prec, val_recall, val_f1


def set_random_seeds(seed):
    """
    set random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_plm(args):
    # loading Pretrained Model
    plm = AutoModel.from_pretrained(args.DATA.plm)
    if args.DATA.use_context:
        config = plm.config
        config.type_vocab_size = 4
        plm.embeddings.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        plm._init_weights(plm.embeddings.token_type_embeddings)

    return plm