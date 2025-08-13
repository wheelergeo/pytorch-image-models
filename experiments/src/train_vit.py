import argparse
import timm
import torch
import os
import time
import sys
import json


from timm.data import create_loader, resolve_model_data_config
from timm.utils import CheckpointSaver, AverageMeter
from timm.utils import update_summary, is_primary, reduce_tensor, get_outdir, accuracy, distribute_bn, init_distributed_device
from timm.models import safe_model_name, resume_checkpoint
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.loss import LabelSmoothingCrossEntropy
from experiments.utils.datasets import ParquetImageDataset
from datetime import datetime
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel


def validate(
        args,
        epoch,
        model,
        loader,
        loss_fn,
        channels_last=False,
        prefetcher=True,
        device=torch.device('cuda'),
        model_dtype=None,
        world_size=1):
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()
    end = time.time()
    with torch.inference_mode():
        if is_primary(args):
            loader = tqdm(loader, file=sys.stdout)
            loader.desc = "[Valid epoch {}]".format(epoch)
            loader.set_postfix(
                loss="N/A",
                acc1="N/A",
                acc5="N/A",
                throughput="N/A"
            )

        for input, target in loader:
            if not prefetcher:
                input = input.to(device=device, dtype=model_dtype)
                target = target.to(device=device)
            if channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # # augmentation reduction
            # reduce_factor = args.tta
            # if reduce_factor > 1:
            #     output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
            #     target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = output.shape[0]
            if world_size > 1:
                batch_size = batch_size * world_size
                reduced_loss = reduce_tensor(loss.data, world_size)
                acc1 = reduce_tensor(acc1, world_size)
                acc5 = reduce_tensor(acc5, world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()
            
            losses_m.update(reduced_loss.item(), batch_size)
            top1_m.update(acc1.item(), batch_size)
            top5_m.update(acc5.item(), batch_size)

            if is_primary(args):
                loader.set_postfix(
                    loss=f"{losses_m.avg:>6.3f}",
                    acc1=f"{top1_m.avg:>7.3f}",
                    acc5=f"{top5_m.avg:>7.3f}",
                    throughput=f"{batch_size / (time.time() - end):.3f}"
                )

            end = time.time()

    return {'loss': losses_m.avg, 'top1': top1_m.avg, 'top5': top5_m.avg}


def train_one_epoch(
        args,
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        accum_steps=1,
        channels_last=False,
        prefetcher=True,
        device=torch.device('cuda'),
        lr_scheduler=None,
        saver=None,
        model_dtype=None,
        mixup_fn=None,
        precise_timing=False,
        recovery_interval=None,
        world_size=1):
    # 用于分布式训练，以便在梯度累计时，临时关闭分布式梯度同步，从而提升训练效率
    has_no_sync = hasattr(model, "no_sync")

    # 用于记录每次更新的时间
    update_time_m = AverageMeter()
    # 用于记录损失
    losses_m = AverageMeter()

    model.train()

    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    # 用于记录最后一个 batch_id，若已经执行到最后一个batch，无论是否满足 accum_steps 的整数倍都需要更新梯度
    last_batch_idx = len(loader) - 1
    # 用于记录最后几个不足 accum_steps 整数倍的首个 batch_id，用于确保最后一次梯度能进行更新
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0

    if is_primary(args):
        loader = tqdm(loader, file=sys.stdout)
        loader.desc = "[Train epoch {}]".format(epoch)
        loader.set_postfix(
            lr="N/A",
            loss="N/A",
            update="N/A",
            throughput="N/A"
        )
    
    for batch_idx, (input, target) in enumerate(loader):
        def _forward():
            output = model(input)
            _loss = loss_fn(output, target)
            if accum_steps > 1:
                _loss /= accum_steps
            return _loss

        def _backward(_loss):
            _loss.backward(create_graph=False)
            if need_update:
                optimizer.step()

        if not prefetcher:
            input, target = input.to(device=device, dtype=model_dtype), target.to(device=device)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        need_update = (batch_idx == last_batch_idx) or ((batch_idx + 1) % accum_steps == 0)
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        global_batch_size = batch_size = input.shape[0]
        global_batch_size *= world_size

        if has_no_sync and not need_update:
            # 在 with 内多线程不会触发梯度同步，确保仅在 need_update 时同步梯度
            with model.no_sync():
                loss = _forward()
                _backward(loss)
        else:
            loss = _forward()
            _backward(loss)

        losses_m.update(loss.item() * accum_steps, batch_size)
        update_sample_count += global_batch_size

        if not need_update:
            continue

        num_updates += 1
        optimizer.zero_grad()

        # calculate and update time
        if precise_timing:
            torch.cuda.synchronize()
        time_now = time.time()

        update_time_m.update(time_now - update_start_time)
        update_start_time = time_now

        lrl = [param_group['lr'] for param_group in optimizer.param_groups]
        lr = sum(lrl) / len(lrl)
        loss_avg, loss_now = losses_m.avg, losses_m.val
        if world_size > 1:
            loss_avg = reduce_tensor(loss.new([loss_avg]), world_size).item()
            loss_now = reduce_tensor(loss.new([loss_now]), world_size).item()

        if is_primary(args):
            loader.set_postfix(
                lr=f"{lr:.3e}",
                loss=f"{loss_avg:#.3g}",
                update=f"{update_time_m.val:.2f}s",
                throughput=f"{update_sample_count / update_time_m.val:>7.2f}im/s"
            )

        if saver is not None and recovery_interval and (
                (update_idx + 1) % recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            # 每次参数更新时更新一次学习率
            lr_scheduler.step_update(num_updates=num_updates)

        update_sample_count = 0
        # end for

    loss_avg = losses_m.avg
    if world_size > 1:
        loss_avg = torch.tensor([loss_avg], device=device, dtype=torch.float32)
        loss_avg = reduce_tensor(loss_avg, world_size).item()

    return {'loss': loss_avg}


def run(args: argparse.Namespace):
    device = init_distributed_device(args)
    print(f'Process {args.rank}, total {args.world_size}, device {args.device}') 

    # 1. create model
    model = timm.models.create_model(
        args.model,
        pretrained=False,
        in_chans=3,
        num_classes=200,
        drop_path_rate=args.drop_path_rate,
        drop_rate=args.drop_rate,
    ).to(device)
    with torch.no_grad():
        model.get_classifier().weight.mul_(args.head_init_scale)
        model.get_classifier().bias.mul_(args.head_init_scale)
    data_config = resolve_model_data_config(model)
    # optionally resume from a checkpoint
    if args.resume_path:
        optimizer = create_optimizer_v2(
            model,
            **optimizer_kwargs(cfg=args)
        )
        resume_epoch = resume_checkpoint(
            model,
            args.resume_path,
            optimizer=optimizer,
            log_info=is_primary(args)
        )
        resume_epoch = resume_epoch if resume_epoch is not None else 0
        args.lr = optimizer.param_groups[0]['lr']
    else:
        resume_epoch = 0
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[device])
            
    if is_primary(args):
        print(f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    # 2. create dataset
    train_dataset = ParquetImageDataset(parquet_files=args.train_files)
    val_dataset = ParquetImageDataset(parquet_files=args.val_files)

    # 3. create dataloader
    train_batch_size = min(args.train_batch_size, len(train_dataset))
    val_batch_size = min(args.val_batch_size, len(val_dataset))
    num_workers = min([os.cpu_count() // args.world_size, train_batch_size if train_batch_size > 1 else 0, 8])
    if is_primary(args):
        print('Using {} dataloader workers every process'.format(num_workers))
    train_loader = create_loader(
        train_dataset,
        input_size=data_config['input_size'],
        batch_size=train_batch_size,
        is_training=True,
        no_aug=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=num_workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        crop_mode=data_config['crop_mode'],
        pin_memory=True,
        device=device,
        use_prefetcher=args.prefetcher,
        use_multi_epochs_loader=False,
    )
    val_loader = create_loader(
        val_dataset,
        input_size=data_config['input_size'],
        batch_size=val_batch_size,
        is_training=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=num_workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        crop_mode=data_config['crop_mode'],
        pin_memory=True,
        device=device,
        use_prefetcher=args.prefetcher,
    )

    # 4. create loss function, optimizer
    train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing).to(device=device)
    validate_loss_fn = torch.nn.CrossEntropyLoss().to(device=device)
    if args.lr is None:
        on = args.opt.lower()
        global_batch_size = train_batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        batch_ratio = batch_ratio ** 0.5 if any([o in on for o in ('ada', 'lamb')]) else batch_ratio
        args.lr = args.lr_base * batch_ratio
        optimizer = create_optimizer_v2(
            model,
            **optimizer_kwargs(cfg=args)
        )
    elif args.lr is not None and args.weight_decay is not None and args.resume_path in (None, ''):
        # default lr = 3e-4
        optimizer = create_optimizer_v2(
            model,
            **optimizer_kwargs(cfg=args)
        )
    if is_primary(args):
        print(f'Learning rate: {args.lr:#.3g}')
        
    # 5. create saver
    eval_metric = 'top1' if val_loader is not None else 'loss'
    # decreasing_metric 确定是最小最优，还是最大最优
    decreasing_metric = eval_metric == 'loss'
    saver = None
    output_dir = None
    if is_primary(args):
        if args.resume_path:
            output_dir = os.path.dirname(args.resume_path)
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model)
            ])
            output_dir = get_outdir(args.output_dir, exp_name)
        saver = CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing_metric,
            max_history=100,
        )

    # 6. create scheduler
    # len(train_loader) 表示每个 epoch 包含的 patch 数
    # + args.grad_accum_steps - 1 属于向上取整操作，确保最后一个累计步不满也能参与梯度更新
    updates_per_epoch = (len(train_loader) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args, decreasing_metric=decreasing_metric),
        updates_per_epoch=updates_per_epoch,
    )
    if lr_scheduler is not None and resume_epoch > 0:
        # 每个 epoch 开始时更新一次学习率
        lr_scheduler.step_update(resume_epoch * updates_per_epoch)
    if is_primary(args):
        # 如果 warmup_prefix 为 True，则说明 warmup_epochs 是额外添加的，不计算在训练 epochs 中
        if args.warmup_prefix:
            sched_explain = f'(warmup({args.warmup_epochs}) + epochs({args.epochs}))'
        else:
            sched_explain = f'(warmup({args.warmup_epochs}) + epochs({args.epochs - args.warmup_epochs}))'
        print(f'Scheduled epochs: {num_epochs} {sched_explain}. '
              f'LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.')

    # 7. train
    results = []
    best_epoch = 0
    best_metric = None
    early_stop_cnt = 0
    try:
        for epoch in range(resume_epoch, num_epochs):
            stop_flag = torch.tensor([0], device=device)
            if hasattr(train_dataset, 'set_epoch'):
                train_dataset.set_epoch(epoch)
            elif args.distributed and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            train_output = train_one_epoch(
                args=args,
                epoch=epoch,
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_fn=train_loss_fn,
                accum_steps=args.grad_accum_steps,
                channels_last=args.channels_last,
                prefetcher=args.prefetcher,
                device=device,
                lr_scheduler=lr_scheduler,
                saver=saver,
                model_dtype=torch.float32,
                recovery_interval=None,
                world_size=args.world_size)

            val_output = None
            if val_loader is not None:
                val_output = validate(
                    args=args,
                    epoch=epoch,
                    model=model,
                    loader=val_loader,
                    loss_fn=validate_loss_fn,
                    channels_last=args.channels_last,
                    prefetcher=args.prefetcher,
                    device=device,
                    model_dtype=torch.float32,
                    world_size=args.world_size)
                
            if is_primary(args):
                latest_results = {
                    'epoch': epoch,
                    'train': train_output,
                }
                if val_output is not None:
                    latest_results['val'] = val_output
                results.append(latest_results)

            if output_dir is not None:
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                update_summary(
                    epoch,
                    train_output,
                    val_output,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                )

            if val_output is not None:
                latest_metric = val_output[eval_metric]
            else:
                latest_metric = train_output[eval_metric]

            if saver is not None:
                # save proper checkpoint with eval metric
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=latest_metric)
                # early stopping judgment
                if best_epoch == epoch:
                    early_stop_cnt = 0  # 有提升，重置计数器
                else:
                    early_stop_cnt += 1  # 没提升，计数器+1
                if early_stop_cnt >= args.early_stop_patience:
                    print(f"Early stopping at epoch {epoch} (no improvement in {args.early_stop_patience} epochs)")
                    stop_flag[0] = 1
                    if args.distributed:
                        torch.distributed.broadcast(stop_flag, src=args.rank)

            if stop_flag.item():
                if args.distributed:
                    torch.distributed.barrier()
                break
    except KeyboardInterrupt:
        pass

    # distributed
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    if best_metric is not None:
        print('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

    if is_primary(args):
        # for parsable results display, dump top-10 summaries to avoid excess console spam
        display_results = sorted(
            results,
            key=lambda x: x.get('validation', x.get('train')).get(eval_metric, 0),
            reverse=decreasing_metric,
        )
        print(f'--result\n{json.dumps(display_results[-10:], indent=4)}')


def defaultargs() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument('--device',
                             type=str,
                             default='cuda',
                             help='Device type to use for training')
    model_group.add_argument('--model',
                             type=str,
                             default='vit_tiny_patch16_224',
                             help='Name of model to train')
    model_group.add_argument('--drop-path-rate',
                             type=float,
                             default=0.2,
                             help='Stochastic depth rate')
    model_group.add_argument('--drop-rate',
                             type=float, 
                             default=0.2,
                             help='Dropout rate')
    model_group.add_argument('--head-init-scale',
                             type=float,
                             default=0.001,
                             help='Classifier head initialization scaling factor')
    model_group.add_argument('--resume-path',
                             type=str, 
                             default='',
                             help='Resume full model and optimizer state from checkpoint (default: none)')
    
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument('--train-files',
                            type=str,
                            nargs='+',
                            default=["./dataset/tiny-imagenet/data/train-00000-of-00001-1359597a978bc4fa.parquet"],
                            help='Train files paths')
    data_group.add_argument('--val-files',
                            type=str,
                            nargs='+',
                            default=["./dataset/tiny-imagenet/data/valid-00000-of-00001-70d52db3c749a935.parquet"],
                            help='Val files paths')
    data_group.add_argument('--train-batch-size',
                            type=int,
                            default=128,
                            help='Training batch size per device')
    data_group.add_argument('--val-batch-size',
                            type=int,
                            default=128,
                            help='Validation batch size per device')
    data_group.add_argument('--prefetcher',
                            default=True,
                            action='store_true',
                            help='enable fast prefetcher')
    
    optim_group = parser.add_argument_group('Optimizer Options')
    optim_group.add_argument('--label-smoothing',
                             type=float,
                             default=0.1,
                             help='Label smoothing factor')
    optim_group.add_argument('--opt',
                             default='adamw',
                             type=str,
                             help='Optimizer')
    optim_group.add_argument('--momentum',
                             type=float,
                             default=0.9,
                             help='Optimizer momentum')
    optim_group.add_argument('--weight-decay',
                             type=float,
                             default=0.05,
                             help='weight decay')
    optim_group.add_argument('--lr',
                             type=float,
                             default=5e-4,
                             help='Learning rate')
    optim_group.add_argument('--lr-base',
                             type=float,
                             default=5e-4,
                             help='base learning rate: lr = lr_base * global_batch_size / base_size')
    optim_group.add_argument('--lr-base-size',
                             type=int,
                             default=512,
                             help='base learning rate batch size (divisor, default: 256).')
    
    saver_group = parser.add_argument_group('Saver Options')
    saver_group.add_argument('--output-dir',
                             type=str,
                             default='./checkpoints/my-vit-tiny-patch16-224',
                             help='Output directory for checkpoints and summaries')
    
    schedule_group = parser.add_argument_group('Scheduler Options')
    schedule_group.add_argument('--sched',
                                 type=str,
                                 default='cosine',
                                 help='LR scheduler (default: "cosine"')
    schedule_group.add_argument('--sched-on-updates',
                                default=True,
                                action='store_true',
                                help='enable scheduler on updates')
    schedule_group.add_argument('--warmup-epochs',
                                type=int,
                                default=5,
                                help='Number of warmup epochs')
    schedule_group.add_argument('--warmup-prefix',
                                default=True,
                                action='store_true',
                                help='Whether to regard warmup epochs as extra epochs')
    
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument('--epochs',
                             type=int,
                             default=100,
                             help='Total number of epochs to train for')
    train_group.add_argument('--grad-accum-steps',
                             type=int,
                             default=1,
                             help='Number of gradient accumulation steps')
    train_group.add_argument('--channels-last',
                             default=True,
                             action='store_true',
                             help='Whether to use channels-last memory format')
    train_group.add_argument('--early-stop-patience',
                             type=int,
                             default=8,
                             help='Early stopping patience (epochs)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    run(defaultargs())