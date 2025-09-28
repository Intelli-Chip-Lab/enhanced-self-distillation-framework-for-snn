# -*- coding: utf-8 -*-
import sys
import math
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import time
import torch
from model import *
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from util import Logger, Bar, AverageMeter, accuracy, load_dataset, warp_decay, split_params, init_config
from model.layer import *
import numpy as np
def train(train_ldr, optimizer, model, evaluator, args):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    bar = Bar('Processing', max=len(train_ldr))
    device = next(model.parameters()).device

    for idx, (ptns, labels) in enumerate(train_ldr):
        ptns, labels = ptns.to(device), labels.to(device)
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        with torch.no_grad():
            model.set_spike_prop_state()
            model.reset_model()
            in_data, _ = torch.broadcast_tensors(ptns, torch.zeros((args.T,) + ptns.shape))
            in_data = in_data.reshape(-1, *in_data.shape[2:])
            out_spikes = model(in_data)
        model.set_rate_prop_state()
        in_data, _ = torch.broadcast_tensors(ptns, torch.zeros((args.T,) + ptns.shape))
        in_rate = in_data.mean(dim=0).detach()
        out_rate = model(in_rate)
        avg_fr = out_rate[-1]
        hard_loss = 0
        soft_loss = 0
        avg_temp = torch.stack(out_rate)
        teach_avg = make_teacher(avg_temp, labels)
        for t in range(len(out_rate)):
            hard_loss += evaluator(out_rate[t], labels)
        for t in range(len(out_rate)):
            soft_loss += esd_loss(out_rate[t], teach_avg.detach(), 3)
        loss = (hard_loss * args.alp + soft_loss * args.beta)
        loss2 = loss / model.time_step
        loss2.backward()
        optimizer.step()
        prec1, prec5 = accuracy(avg_fr.data, labels.data, topk=(1, 5))
        losses.update(loss.data.item(), ptns.size(0))
        top1.update(prec1.item(), ptns.size(0))
        top5.update(prec5.item(), ptns.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        bar.suffix = (
            '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total} | ETA: {eta} | '
            'Loss: {loss:.4f} | top1: {top1:.4f} | top5: {top5:.4f}'
        ).format(
            batch=idx + 1,
            size=len(train_ldr),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()

    return top1.avg, losses.avg

def test(val_ldr, model, evaluator, args):

    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    bar = Bar('Processing', max=len(val_ldr))

    with torch.no_grad():
        for idx, (ptns, labels_batch) in enumerate(val_ldr):
            ptns, labels_batch = ptns.to(next(model.parameters()).device), labels_batch.to(next(model.parameters()).device)
            model.reset_model()
            model.set_spike_prop_state()
            in_data, _ = torch.broadcast_tensors(ptns, torch.zeros((args.T,) + ptns.shape))
            in_data = in_data.reshape(-1, *in_data.shape[2:])
            out_spikes = model(in_data)
            avg_fr = out_spikes.reshape(args.T, -1, out_spikes.shape[1]).mean(0)
            loss = evaluator(avg_fr, labels_batch)
            prec1, prec5 = accuracy(avg_fr.data, labels_batch.data, topk=(1, 5))
            losses.update(loss.data.item(), ptns.size(0))
            top1.update(prec1.item(), ptns.size(0))
            top5.update(prec5.item(), ptns.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=idx + 1,
                size=len(val_ldr),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        bar.finish()
        return top1.avg, losses.avg

def main():
    # set device, data type
    device, dtype = torch.device(args.device if torch.cuda.is_available() else "cpu"), torch.float

    log = Logger(args, args.log_path)
    log.info_args(args)
    writer = SummaryWriter(args.log_path)

    train_data, val_data, num_class = load_dataset(args.dataset, args.data_path, cutout=args.cutout,
                                                   auto_aug=args.auto_aug)
    train_ldr = DataLoader(dataset=train_data, batch_size=args.train_batch_size, shuffle=True,
                           pin_memory=True, num_workers=args.num_workers)
    val_ldr = DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False,
                         pin_memory=True, num_workers=args.num_workers)

    kwargs_spikes = {'v_reset': args.v_reset, 'thresh': args.thresh, 'decay': warp_decay(args.decay),
                     'detach_reset': args.detach_reset}
    model = eval(args.arch + f'(num_classes={num_class}, **kwargs_spikes)')
    model.wrap_model(time_step=args.T)
    model.to(device, dtype)
    params = split_params(model)
    params = [
        {'params': params[1], 'weight_decay': args.wd},
        {'params': params[2], 'weight_decay': 0}
    ]

    if args.optim.lower() == 'sgdm':
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9)
    elif args.optim.lower() == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, amsgrad=False)
    else:
        raise NotImplementedError()

    evaluator = torch.nn.CrossEntropyLoss()
    start_epoch = 0
    best_epoch = 0
    best_acc = 0.0
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state['best_net'])
        log.info('Load checkpoint from epoch {}'.format(start_epoch))
        log.info('Test the checkpoint: {}'.format(test(val_ldr, model,evaluator, args=args)))
    args.start_epoch = start_epoch
    if args.scheduler.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.num_epoch)
    else:
        raise NotImplementedError()

    for epoch in range(start_epoch, args.num_epoch):
        train_acc, train_loss = train(train_ldr, optimizer, model, evaluator, args=args)
        if args.scheduler != 'None':
            scheduler.step()
        val_acc, val_loss = test(val_ldr, model, evaluator, args=args)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            state = {
                'best_acc': best_acc,
                'best_epoch': epoch,
                'best_net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, os.path.join(args.log_path, 'model_weights.pth'))
        log.info(
            'Epoch %03d: train loss %.5f, test loss %.5f, train acc %.5f,test acc %.5f, Saved custom_model..  with best acc %.5f in the epoch %03d\n'
            % (
                epoch, train_loss, val_loss, train_acc, val_acc, best_acc, best_epoch)
        )
        # record in tensorboard
        writer.add_scalars('Loss', {'val': val_loss, 'train': train_loss},
                           epoch + 1)
        writer.add_scalars('Acc', {'val': val_acc, 'train': train_acc},
                           epoch + 1)

    log.info(
        'Finish training: the best validation accuracy of SNN is {} in epoch {}. \n The relate checkpoint path: {}'.format(
            best_acc, best_epoch, os.path.join(args.log_path, 'model_weights.pth')))


if __name__ == '__main__':
    from config.config import args
    init_config(args)
    main()
