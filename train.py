import argparse
import logging
import math
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy, Logger

logger = logging.getLogger(__name__)
best_acc = 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=7./16., last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))
    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def test(args, test_loader, model, epoch):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets, reduction='sum')
            test_loss += loss.item()
            _, pred = outputs.max(1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
    acc = 100. * correct / total
    print(f"Test Epoch {epoch}: Loss {test_loss/total:.4f}, Acc {acc:.2f}%")
    return acc


def train(args, labeled_loader, unlabeled_loader, test_loader, model, optimizer, ema_model, scheduler):
    global best_acc
    model.train()
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            prog_bar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])
        else:
            prog_bar = range(args.eval_step)
        end = time.time()
        for batch_idx in prog_bar:
            # fetch labeled batch
            try:
                inputs_x, targets_x = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                inputs_x, targets_x = next(labeled_iter)
            # fetch unlabeled batch
            try:
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            data_time.update(time.time() - end)
            batch_size = inputs_x.size(0)
            # interleave for batchnorm consistency
            inputs = torch.cat([inputs_x, inputs_u_w, inputs_u_s], dim=0)
            inputs = interleave(inputs, 2 * args.mu + 1).to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 2 * args.mu + 1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            # supervised loss
            Lx = F.cross_entropy(logits_x, targets_x.to(args.device), reduction='mean')
            # unsupervised loss
            with torch.no_grad():
                pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()
            Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
            loss = Lx + args.lambda_u * Lu
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.use_ema:
                ema_model.update(model)
            scheduler.step()
            losses.update(loss.item(), batch_size)
            mask_probs.update(mask.mean().item(), batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                prog_bar.set_description(f"Loss {losses.avg:.4f}, Mask {mask_probs.avg:.4f}")
        # evaluation
        acc = test(args, test_loader, model, epoch)
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if args.local_rank in [-1, 0]:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, args.out)
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar('Test/Accuracy', acc, epoch)


def main():
    parser = argparse.ArgumentParser(description='FixMatch with Imbalanced Data Handling')
    parser.add_argument('--gpu-id', default=0, type=int)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'stl10', 'smallimagenet'])
    parser.add_argument('--num-labeled', type=int, default=4000)
    parser.add_argument('--arch', default='wideresnet', choices=['wideresnet', 'resnet'])
    parser.add_argument('--total-steps', default=2**20, type=int)
    parser.add_argument('--eval-step', default=1024, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--lr', default=0.03, type=float)
    parser.add_argument('--warmup', default=0, type=float)
    parser.add_argument('--wdecay', default=5e-4, type=float)
    parser.add_argument('--nesterov', action='store_true', default=True)
    parser.add_argument('--use-ema', action='store_true', default=True)
    parser.add_argument('--ema-decay', default=0.999, type=float)
    parser.add_argument('--mu', default=7, type=int)
    parser.add_argument('--lambda-u', dest='lambda_u', default=1.0, type=float)
    parser.add_argument('--T', default=1.0, type=float)
    parser.add_argument('--threshold', default=0.95, type=float)
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--opt_level', type=str, default='O1')
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    # imbalanced settings
    parser.add_argument('--num-max', default=500, type=int)
    parser.add_argument('--imb-ratio-label', default=1, type=int)
    parser.add_argument('--imb-ratio-unlabel', default=1, type=int)
    parser.add_argument('--flag-reverse-LT', default=0, type=int)

    args = parser.parse_args()
    global best_acc

    # device setup
    if args.local_rank == -1:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
    args.device = device

    logging.basicConfig(level=logging.INFO)
    logger.warning(f"Device: {args.device}, World size: {args.world_size}")

    if args.seed is not None:
        set_seed(args.seed)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    # dataset and model settings
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.model_depth = 28; args.model_width = 2
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.model_depth = 28; args.model_width = 8
    # add stl10/smallimagenet config as needed

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](args, './data')
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else torch.utils.data.distributed.DistributedSampler
    labeled_loader = DataLoader(labeled_dataset, sampler=train_sampler(labeled_dataset), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, sampler=train_sampler(unlabeled_dataset), batch_size=args.batch_size*args.mu, num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size, num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # model
    def create_model():
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            return models.build_wideresnet(depth=args.model_depth, widen_factor=args.model_width, dropout=0, num_classes=args.num_classes)
        else:
            import models.resnet_ori as models
            return models.ResNet50(num_classes=args.num_classes)
    model = create_model().to(args.device)

    # optimizer & scheduler
    no_decay = ['bias', 'bn']
    grouped = [
        {'params': [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = SGD(grouped, lr=args.lr, momentum=0.9, nesterov=args.nesterov)
    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)

    # EMA
    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)
    else:
        ema_model = None

    args.start_epoch = 0
    # resume
    if args.resume:
        assert os.path.isfile(args.resume)
        ckpt = torch.load(args.resume)
        args.start_epoch = ckpt['epoch']
        best_acc = ckpt['best_acc']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if args.use_ema:
            ema_model.ema.load_state_dict(ckpt['ema_state_dict'])

    # DDP
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # start training
    train(args, labeled_loader, unlabeled_loader, test_loader, model, optimizer, ema_model, scheduler)

if __name__ == '__main__':
    main()
