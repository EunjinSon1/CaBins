import argparse
import os
import sys
import uuid
from datetime import datetime as dt

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import wandb
from tqdm import tqdm

import model_io
import utils

from loss import SILogLoss, BinsChamferLoss
from utils import RunningAverage

sys.path.append(sys.path[0])

from models.mainModel import *
from dataloader import DepthDataLoader

PROJECT = "CaBins"
logging = True


def is_rank_zero(args):
    return args.rank == 0

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def convert_models_to_fp32(l):
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()

    if isinstance(l, nn.MultiheadAttention):
        for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
            tensor = getattr(l, attr)
            if tensor is not None:
                tensor.data = tensor.data.float()

    for name in ["text_projection", "proj"]:
        if hasattr(l, name):
            attr = getattr(l, name)
            if attr is not None:
                attr.data = attr.data.float()


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    ###################################### Load model ##############################################

    clipModel, _ = load("RN101", device=args.gpu, jit=False)

    model = TotalModel(clipModel, args.gpu, args.max_depth, args.min_depth, args.n_bins)
    convert_models_to_fp32(model)
    ################################################################################################

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    args.multigpu = False
    if args.distributed:
        # Use DDP
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size / ngpus_per_node)

        args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print(args.gpu, args.rank, args.batch_size, args.workers)
        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                          find_unused_parameters=True)

    elif args.gpu is None:
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    args.epoch = 0
    args.last_epoch = -1
    train(model, args, epochs=args.epochs, lr=args.lr, device=args.gpu, root=args.root,
          experiment_name=args.name, optimizer_state_dict=None)



def train(model, args, epochs=10, experiment_name="DeepLab", lr=0.0001, root=".", device=None,
          optimizer_state_dict=None):
    global PROJECT
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ###################################### Logging setup #########################################
    print(f"Training {experiment_name}")

    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{args.bs}-tep{epochs}-lr{lr}-wd{args.wd}-{uuid.uuid4()}"
    name = f"{experiment_name}_{run_id}"
    should_write = ((not args.distributed) or args.rank == 0)
    should_log = should_write and logging
    if should_log:
        tags = args.tags.split(',') if args.tags != '' else None
        if args.dataset != 'nyu':
            PROJECT = PROJECT + f"-{args.dataset}"
        wandb.init(project=PROJECT, name=name, config=args, dir=args.root, tags=tags, notes=args.notes)


    ######################################Data loader######################################
    train_loader = DepthDataLoader(args, 'train').data
    test_loader = DepthDataLoader(args,'online_eval').data

    ###################################### losses ##############################################
    criterion_ueff = SILogLoss()
    criterion_bins = BinsChamferLoss() if args.chamfer else None

    #####################################train mode#####################################
    model.train()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("Total number of learning parameters: {}".format(num_params_update))

    ###################################### Optimizer ################################################
    if args.same_lr:
        print("Using same LR")
        params = model.parameters()
    else:
        print("Using diff LR")
        m = model.module if args.multigpu else model

        params=[
            {"params": m.clip_model.parameters(),"lr": 0.1*lr},
            {"params": m.context_decoder.parameters(), "lr": lr},
            {"params": m.decoder.parameters(),"lr": lr},
            {"params": m.computeAdaptiveBin.parameters(), "lr": lr},
            {"params": m.predict_D.parameters(), "lr": lr}
            ]

    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)


    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))
        del checkpoint


    ################################################################################################
    # some globals
    iters = len(train_loader)
    step = args.epoch * iters
    best_loss_abs = np.inf

    ###################################### Scheduler ###############################################
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader),
                                              cycle_momentum=True,
                                              base_momentum=0.85, max_momentum=0.95, last_epoch=args.last_epoch,
                                              div_factor=args.div_factor,
                                              final_div_factor=args.final_div_factor)
    if args.resume != '' and scheduler is not None:
        scheduler.step(args.epoch + 1)


    ################################################################################################

    for epoch in range(args.epoch, epochs):
        ################################# Train loop ##########################################################
        if should_log: wandb.log({"Epoch": epoch}, step=step)
        for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
                             total=len(train_loader)) if is_rank_zero(
                args) else enumerate(train_loader):

            optimizer.zero_grad()

            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue

            bin_edges, pred = model(img)

            mask = depth > args.min_depth
            l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)

            if args.w_chamfer > 0:
                l_chamfer = criterion_bins(bin_edges, depth)
            else:
                l_chamfer = torch.Tensor([0]).to(img.device)

            loss = l_dense + args.w_chamfer * l_chamfer

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional


            optimizer.step()

            if should_log and step % 5 == 0:
                wandb.log({f"Train/{criterion_ueff.name}": l_dense.item()}, step=step)
                wandb.log({f"Train/{criterion_bins.name}": l_chamfer.item()}, step=step)

            step += 1
            scheduler.step()


            ########################################################################################################

            if should_write and step % args.validate_every == 0:
                ################################# Validation loop ##################################################
                model.eval()
                metrics, val_si = validate(args, model, test_loader, criterion_ueff, epoch, epochs, device)

                # print("Validated: {}".format(metrics))
                if should_log:
                    wandb.log({
                        f"Test/{criterion_ueff.name}": val_si.get_value(),
                        # f"Test/{criterion_bins.name}": val_bins.get_value()
                    }, step=step)

                    wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=step)
                    model_io.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_{run_id}_latest.pt",
                                             root=os.path.join(root, "checkpoints"))

                if metrics['abs_rel'] < best_loss_abs and should_write:
                    model_io.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_{run_id}.pt",
                                             root=os.path.join(root, "checkpoints"))
                    best_loss_abs = metrics['abs_rel']

                model.train()
                #################################################################################################

    return model


def validate(args, model, test_loader, criterion_ueff, epoch, epochs, device='cpu'):
    with torch.no_grad():
        val_si = RunningAverage()
        # val_bins = RunningAverage()
        metrics = utils.RunningAverageDict()
        for batch in tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation") if is_rank_zero(
                args) else test_loader:
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue

            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            bins, pred = model(img)

            mask = depth > args.min_depth
            l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)
            val_si.append(l_dense.item())

            pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)

            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth_eval] = args.min_depth_eval
            pred[pred > args.max_depth_eval] = args.max_depth_eval
            pred[np.isinf(pred)] = args.max_depth_eval
            pred[np.isnan(pred)] = args.min_depth_eval

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                        int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)

            metrics.update(utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]))

        return metrics.get_value(), val_si


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Training script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--epochs', default=25, type=int, help='number of total epochs to run')
    parser.add_argument('--n-bins', '--n_bins', default=80, type=int,
                        help='number of bins/buckets to divide depth range into')
    parser.add_argument('--lr', '--learning-rate', default=0.00000357, type=float, help='max learning rate')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float, help='weight decay')
    parser.add_argument('--w_chamfer', '--w-chamfer', default=0.1, type=float, help="weight value for chamfer loss")
    parser.add_argument('--div-factor', '--div_factor', default=25, type=float, help="Initial div factor for lr")
    parser.add_argument('--final-div-factor', '--final_div_factor', default=100, type=float,
                        help="final div factor for lr")

    parser.add_argument('--bs', default=16, type=int, help='batch size')
    parser.add_argument('--validate-every', '--validate_every', default=1000, type=int, help='validation period')
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument("--name", default="UnetAdaptiveBins")
    parser.add_argument("--same-lr", '--same_lr', default=False, action="store_true",
                        help="Use same LR for all param groups")
    parser.add_argument("--distributed", default=True, action="store_true", help="Use DDP if set")
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")
    parser.add_argument("--resume", default='', type=str, help="Resume from checkpoint")

    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--tags", default='sweep', type=str, help="Wandb tags")

    parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")
    parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")

    parser.add_argument("--data_path", default='../dataset/nyu/sync/', type=str,
                        help="path to dataset")
    parser.add_argument("--gt_path", default='../dataset/nyu/sync/', type=str,
                        help="path to dataset")

    parser.add_argument('--filenames_file',
                        default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
                        type=str, help='path to the filenames text file')

    parser.add_argument('--input_height', type=int, help='input height', default=416)
    parser.add_argument('--input_width', type=int, help='input width', default=544)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=30)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)

    parser.add_argument('--do_random_rotate', default=True,
                        help='if set, will perform random rotation for augmentation',
                        action='store_true')
    parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--use_right', help='if set, will randomly use right images when train on KITTI',
                        action='store_true')

    parser.add_argument('--data_path_eval',
                        default="../dataset/nyu/official_splits/test/",
                        type=str, help='path to the data for online evaluation')
    parser.add_argument('--gt_path_eval', default="../dataset/nyu/official_splits/test/",
                        type=str, help='path to the groundtruth data for online evaluation')
    parser.add_argument('--filenames_file_eval',
                        default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
                        type=str, help='path to the filenames text file for online evaluation')

    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=30)
    parser.add_argument('--eigen_crop', default=True, help='if set, crops according to Eigen NIPS14',
                        action='store_true')
    parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')

    parser.add_argument('--checkpoint_path', type=str, help='path to a checkpoint to load', default='')



    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    args.batch_size = args.bs
    args.num_threads = args.workers
    args.mode = 'train'
    args.chamfer = args.w_chamfer > 0
    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root)

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
        nodes = node_str.split(',')

        args.world_size = len(nodes)
        args.rank = int(os.environ['SLURM_PROCID'])

    except KeyError as e:
        # We are NOT using SLURM
        args.world_size = 1
        args.rank = 0
        nodes = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method('forkserver')

        print(args.rank)
        port = np.random.randint(15000, 15025)
        args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print(args.dist_url)
        args.dist_backend = 'nccl'
        args.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)
