# This code is constructed based on Pytorch Implementation of FixMatch(https://github.com/kekmodel/FixMatch-pytorch)
import argparse
import logging
import math
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy
from utils import Logger
from progress.bar import Bar
import loss.semiConLoss as scl

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_b = 0

# 作用：该函数用于生成不平衡数据集的类分布
# 具体来说，它根据给定的最大样本数（max_num）、类别数（class_num）和不平衡程度（gamma）来生成每个类别的样本数。通过调整gamma值，可以控制类别分布的不平衡程度。
# 用途：主要用于模拟不平衡数据集，以便在训练时更好地模拟实际中的数据分布情况，尤其是在长尾数据中。
def make_imb_data(max_num, class_num, gamma, flag=1, flag_LT=0):
    mu = np.power(1 / gamma, 1 / (class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))

    if flag == 0 and flag_LT == 1:
        class_num_list = list(reversed(class_num_list))
    return list(class_num_list)

# 作用：该函数计算标签频率的调整系数（adjustments）。它首先计算标签频率并进行归一化，然后使用对数调整频率。这些调整系数可以在训练时用来加权损失函数，处理类别不平衡问题。
# 用途：为每个标签计算一个调整系数，以便在训练过程中根据类别的频率调整损失函数，减小类别不平衡带来的影响。
def compute_adjustment_list(label_list, tro, args):
    label_freq_array = np.array(label_list)
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(args.device)
    return adjustments

# 作用：该函数计算给定训练数据集（train_loader）中每个类别的频率，并将其归一化得到概率分布。这些概率值（py）将在伪标签生成和模型训练过程中用于加权。
# 用途：生成每个类别的概率分布（py），用于后续伪标签的生成，或者在半监督学习中调整无标签数据的伪标签。
def compute_py(train_loader, args):
    """compute the base probabilities"""
    label_freq = {}
    for i, (inputs, labell) in enumerate(train_loader):
        labell = labell.to(args.device)
        for j in labell:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    label_freq_array = torch.from_numpy(label_freq_array)
    label_freq_array = label_freq_array.to(args.device)
    return label_freq_array

# 作用：用于保存模型训练的中间状态（checkpoint），包括模型的参数、优化器状态、学习率调度器状态、当前最佳准确率等。如果当前模型表现最好，还会将其复制为model_best.pth.tar。
# 用途：在训练过程中定期保存模型的状态，以便后续恢复训练或者做模型选择。
def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar', epoch_p=1):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))

# 作用：设置随机种子，确保训练过程的可重复性。通过设置Python、NumPy、PyTorch的随机种子，以及设置cudnn的确定性模式，保证每次训练的结果一致。
# 用途：确保实验结果可复现，这对科研和调试尤为重要。
def set_seed(args):
    seed = args.seed
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 作用：该函数创建一个学习率调度器，使用余弦退火（Cosine Annealing）和预热（warm-up）策略。学习率在训练的初期逐渐增加，达到预设的最大值，然后开始逐渐下降，使用余弦衰减。
# 用途：优化训练过程中的学习率调整，以避免训练初期的学习率过大导致训练不稳定，且后期的学习率逐渐减小有助于模型更好地收敛。
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

# 作用：该函数根据py（类别概率分布）计算标签的调整系数。它通过对py进行对数变换得到调整系数，最终用于加权损失函数。
# 用途：根据标签的概率分布来调整训练中的损失，以便更好地应对类别不平衡问题。
def compute_adjustment_by_py(py, tro, args):
    adjustments = torch.log(py ** tro + 1e-12)
    adjustments = adjustments.to(args.device)
    return adjustments

# 作用：该函数通过对输入张量进行指数运算并归一化，使得输出符合概率分布形式。T是温度参数，控制输出的平滑度。该函数通常用于模型输出的概率进行处理，使其更加平滑或“尖锐”。
# 用途：用于在训练中生成平滑的伪标签或进行概率调整。
def sharp(a, T):
    a = a ** T
    a_sum = torch.sum(a, dim=1, keepdim=True)
    a = a / a_sum
    return a.detach()

# 作用：这是训练脚本的主函数，负责解析命令行参数、初始化模型、加载数据、设置优化器和学习率调度器等。它还包括了训练过程的准备工作，如分布式训练环境的初始化、日志记录器的创建、以及模型的定义。
# 用途：作为整个训练过程的入口，组织并调用了训练、测试等功能模块。
def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'stl10', 'smallimagenet'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnet'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=250000, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=500, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=1, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=0, type=int,
                        help="random seed")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    parser.add_argument('--num-max', default=500, type=int,
                        help='the max number of the labeled data')
    parser.add_argument('--num-max-u', default=4000, type=int,
                        help='the max number of the unlabeled data')
    parser.add_argument('--imb-ratio-label', default=1, type=int,
                        help='the imbalanced ratio of the labelled data')
    parser.add_argument('--imb-ratio-unlabel', default=1, type=int,
                        help='the imbalanced ratio of the unlabeled data')
    parser.add_argument('--flag-reverse-LT', default=0, type=int,
                        help='whether to reverse the distribution of the unlabeled data')
    parser.add_argument('--ema-mu', default=0.99, type=float,
                        help='mu when ema')

    parser.add_argument('--tau', default=2.0, type=float,
                        help='tau for head consistency')
    parser.add_argument('--est-epoch', default=5, type=int,
                        help='the start step to estimate the distribution')
    parser.add_argument('--img-size', default=32, type=int,
                        help='image size for small imagenet')
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='ema ratio for estimating distribution of the unlabeled data')
    parser.add_argument('--beta', default=0.5, type=float,
                        help='ema ratio for estimating distribution of the all data')
    parser.add_argument('--lambda1', default=0.7, type=float,
                        help='coefficient of final loss')
    parser.add_argument('--lambda2', default=1.0, type=float,
                        help='coefficient of final loss')

    args = parser.parse_args()
    global best_acc
    global best_acc_b

    # 作用：根据传入的参数（如args.arch）选择模型架构并创建模型实例。在代码中，它支持两种架构：wideresnet和resnet，并根据指定的深度和宽度创建模型。
    # 用途：根据用户指定的网络架构和参数创建模型实例，便于后续训练。
    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)

        elif args.arch == 'resnet':
            import models.resnet_ori as models
            model = models.ResNet50(num_classes=args.num_classes, rotation=True, classifier_bias=True)

        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters()) / 1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank},"
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}", )

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.dataset_name = 'cifar10'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.dataset_name = 'cifar100'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2

    elif args.dataset == 'stl10':
        args.num_classes = 10
        args.dataset_name = 'stl10'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2


    elif args.dataset == 'smallimagenet':
        args.num_classes = 127
        if args.img_size == 32:
            args.dataset_name = 'imagenet32'
        elif args.img_size == 64:
            args.dataset_name = 'imagenet64'

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, 'datasets/' + args.dataset_name)

    if args.local_rank == 0:
        torch.distributed.barrier()

    labeled_trainloader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    args.est_step = 0

    args.py_con = compute_py(labeled_trainloader, args)
    args.py_uni = torch.ones(args.num_classes) / args.num_classes
    # args.py_uni = args.py_uni.to(args.device)

    args.py_all = args.py_con
    args.py_unlabeled = args.py_uni

    class_list = []
    for i in range(args.num_classes):
        class_list.append(str(i))

    title = 'FixMatch-' + args.dataset
    args.logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
    args.logger.set_names(
        ['Top1_co acc', 'Top5_co acc', 'Best Top1_co acc', 'Top1_b acc', 'Top5_b acc', 'Best Top1_b acc'])

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.py_unlabeled = checkpoint['py_unlabeled']
        args.py_all = checkpoint['py_all']

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()

    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)

    args.logger.close()

# 作用：这是训练过程的核心函数，负责加载数据、计算损失、更新模型参数、计算伪标签等。每个epoch中，它会对有标签和无标签数据分别进行处理，计算各种损失（包括分类损失和对比损失），并更新模型。
# 用途：执行训练的主要步骤，包括训练数据的加载、损失计算、反向传播、梯度更新等。
def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    global best_acc
    global best_acc_b
    test_accs = []
    avg_time = []
    end = time.time()
    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
    logits_la_s = compute_adjustment_by_py(args.py_con, args.tau, args)
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    semiConLoss = scl.SemiConLoss(args.batch_size, args.batch_size, args.num_classes, args)
    semiConLoss2 = scl.softConLoss(args.batch_size, args.batch_size, args.num_classes, args)
    model.train()
    lbs = args.batch_size
    ubs = args.batch_size * args.mu
    py_labeled = args.py_con.to(args.device)
    py_unlabeled = args.py_uni.to(args.device)
    py_all = args.py_all.to(args.device)
    cut1 = lbs + 3 * ubs
    pro = ubs / (ubs + lbs)
    for epoch in range(args.start_epoch, args.epochs):
        print('current epoch: ', epoch + 1)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_con = AverageMeter()
        losses_cls = AverageMeter()
        losses_con2 = AverageMeter()

        bar = Bar('Training', max=args.eval_step)

        num_unlabeled = torch.ones(args.num_classes).to(args.device)
        num_all = torch.ones(args.num_classes).to(args.device)
        for batch_idx in range(args.eval_step):
            try:
                (inputs_x, inputs_x_s, inputs_x_s1), targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                (inputs_x, inputs_x_s, inputs_x_s1), targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s, inputs_u_s1), u_real = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s, inputs_u_s1), u_real = next(unlabeled_iter)
            u_real = u_real.to(args.device)
            mask_l = (u_real != -2).float().unsqueeze(1).to(args.device)
            data_time.update(time.time() - end)
            inputs = torch.cat([inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1, inputs_x_s, inputs_x_s1], dim=0).to(
                args.device)
            targets_x = targets_x.to(args.device)
            feat, feat_mlp, center_feat = model(inputs)
            # -----------------------------------------------------------------------------------------------------------
            logits = model.classify(feat[:cut1])
            logits_b = model.classify1(feat[:cut1])
            logits_x = logits[:lbs]
            logits_x_w, logits_x_s, logits_x_s1 = logits[lbs:].chunk(3)
            logits_x_b = logits_b[:lbs]
            # logits LA
            logits_x_b_w, logits_x_b_s, logits_x_b_s1 = logits_b[lbs:].chunk(3)
            del logits, logits_b

            targets_x = targets_x.to(torch.int64)

            l_u_s = F.cross_entropy(logits_x, targets_x, reduction='mean')
            l_b_s = F.cross_entropy(logits_x_b + logits_la_s, targets_x, reduction='mean')
            logits_la_u = (- compute_adjustment_by_py((1 - pro) * py_labeled + pro * py_all, 1.0, args) +
                           compute_adjustment_by_py(py_unlabeled, 1 + args.tau / 2, args))
            logits_co = 1 / 2 * (logits_x_w + logits_la_u) + 1 / 2 * logits_x_b_w
            energy = -torch.logsumexp((logits_co.detach()) / args.T, dim=1)
            pseudo_label_co = F.softmax((logits_co.detach()) / args.T, dim=1)
            pseudo_label_con = sharp(F.softmax((logits_co.detach()) / args.T, dim=1), 4.0)

            prob_co, targets_co = torch.max(pseudo_label_co, dim=-1)
            mask = prob_co.ge(args.threshold)
            mask = mask.float()

            targets_co = torch.cat([targets_co, targets_co], dim=0).to(args.device)
            logits_b_s = torch.cat([logits_x_b_s, logits_x_b_s1], dim=0).to(args.device)
            logits_la_u_b = compute_adjustment_by_py(py_all, args.tau, args)
            mask_twice = torch.cat([mask, mask], dim=0)
            l_u_b = (F.cross_entropy(logits_b_s + logits_la_u_b, targets_co,
                                     reduction='none') * mask_twice).mean()

            logits_u_s = torch.cat([logits_x_s, logits_x_s1], dim=0).to(args.device)
            l_u_u = (F.cross_entropy(logits_u_s, targets_co,
                                     reduction='none') * mask_twice).mean()

            loss_u = max(1.5, args.mu) * l_u_u + l_u_s
            loss_b = max(1.5, args.mu) * l_u_b + l_b_s
            loss_cls = loss_u + loss_b
            # ----------------------------------------------------------------------------------------------------------
            feat_mlp = feat_mlp[lbs:]
            f3, f4 = feat_mlp[ubs:3 * ubs, :].chunk(2)
            f1, f2 = feat_mlp[3 * ubs:, :].chunk(2)

            # ----------------------------------------------------------------------------------------------------------
            feat_mlp = torch.cat([center_feat, feat_mlp[3 * ubs:, :], feat_mlp[:3 * ubs, :]], dim=0)
            center_label = torch.ones(args.num_classes, args.num_classes).to(args.device)
            one_hot_targets = F.one_hot(targets_x, num_classes=args.num_classes)
            one_hot_targets = torch.cat([one_hot_targets, one_hot_targets], dim=0).to(args.device)
            label_contrac = torch.cat([center_label, one_hot_targets], dim=0).to(args.device)
            # la = compute_adjustment_by_py(py_all, 1.0, args)
            contrac_loss = semiConLoss(feat_mlp, label_contrac)

            # ----------------------------------------------------------------------------------------------------------
            maskcon = energy.le(-8.75)
            idx = torch.nonzero(maskcon).squeeze()
            f3 = torch.reshape(f3[idx, :], (-1, f1.shape[1]))
            f4 = torch.reshape(f4[idx, :], (-1, f1.shape[1]))
            pseudo_label_con = torch.reshape(pseudo_label_con[idx, :], (-1, args.num_classes))

            label_contrac = torch.cat([center_label, one_hot_targets, pseudo_label_con, pseudo_label_con], dim=0).to(
                args.device)
            feat_all = torch.cat([center_feat, f1, f2, f3, f4], dim=0)
            contrac_loss2 = semiConLoss2(label_contrac, feat_all, args.device)

            loss = args.lambda1 * loss_cls + args.lambda2 * contrac_loss + (1 - args.lambda1) * contrac_loss2

            loss.backward()
            losses.update(loss.item())
            losses_cls.update(loss_cls.item())
            losses_con.update(contrac_loss.item())
            losses_con2.update(contrac_loss2.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            mask = mask.unsqueeze(1).to(args.device)
            maskcon = maskcon.float().unsqueeze(1).to(args.device)
            num_all += torch.sum(pseudo_label_co * mask, dim=0)
            # num_unlabeled += torch.sum(pseudo_label_co * mask_l * mask, dim=0)
            num_unlabeled += torch.sum(pseudo_label_co * mask_l * maskcon, dim=0)
            batch_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({batch}/{size}) | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                         'Loss: {loss:.4f} | Loss_cls: {loss_cls:.4f} | Loss_con: {loss_con:.4f} | Loss_con2: {loss_con2:.4f}'.format(
                batch=batch_idx + 1,
                size=args.eval_step,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                loss_cls=losses_cls.avg,
                loss_con=losses_con.avg,
                loss_con2=losses_con2.avg,
            )
            bar.next()
        bar.finish()

        if epoch > args.est_epoch:
            py_unlabeled = args.alpha * py_unlabeled + (1 - args.alpha) * num_unlabeled / sum(num_unlabeled)
            py_all = args.beta * py_all + (1 - args.beta) * num_all / sum(num_all)
        print('\n')
        print(py_unlabeled)
        print(py_all)
        avg_time.append(batch_time.avg)

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model
        test_la = - compute_adjustment_by_py(1 / 2 * py_labeled + 1 / 2 * py_all, 1.0, args)
        if args.local_rank in [-1, 0]:

            test_loss, test_acc, test_top5_acc, test_acc_b, test_top5_acc_b = test(args, test_loader,
                                                                                                 test_model, epoch,
                                                                                                 test_la)
            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc_b, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc_b > best_acc_b

            best_acc = max(test_acc, best_acc)
            best_acc_b = max(test_acc_b, best_acc_b)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema

            if (epoch + 1) % 10 == 0 or (is_best and epoch > 250):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                    'acc': test_acc,
                    'best_acc': best_acc_b,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'py_unlabeled': py_unlabeled,
                    'py_all': py_all
                }, is_best, args.out, epoch_p=epoch + 1)

            test_accs.append(test_acc_b)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc_b))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

            args.logger.append([test_acc, test_top5_acc, best_acc, test_acc_b, test_top5_acc_b, best_acc_b])
    if args.local_rank in [-1, 0]:
        args.writer.close()

# 作用：该函数在每个epoch后进行模型的评估，计算并输出测试集上的损失、准确率（top-1和top-5）。它用于验证模型的性能，并帮助选择最佳模型。
# 用途：评估模型在测试集上的表现，计算标准的评估指标，如准确率，便于跟踪模型的训练效果。
def test(args, test_loader, model, epoch, la):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top1_b = AverageMeter()
    top5_b = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs_feat = model(inputs)
            outputs = model.classify(outputs_feat)
            outputs_b = model.classify1(outputs_feat)
            outputs_co = 1 / 2 * (outputs + la) + 1 / 2 * outputs_b
            loss = F.cross_entropy(outputs_b, targets)

            prec1_b, prec5_b = accuracy(outputs_b, targets, topk=(1, 5))
            prec1_co, prec5_co = accuracy(outputs_co, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1_co.item(), inputs.shape[0])
            top5.update(prec5_co.item(), inputs.shape[0])
            top1_b.update(prec1_b.item(), inputs.shape[0])
            top5_b.update(prec5_b.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

    logger.info("top-1 acc: {:.2f}".format(top1_b.avg))
    logger.info("top-5 acc: {:.2f}".format(top5_b.avg))

    return losses.avg, top1.avg, top5.avg, top1_b.avg, top5_b.avg


if __name__ == '__main__':
    main()
