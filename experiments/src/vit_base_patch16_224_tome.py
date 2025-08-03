import timm
import argparse
import torch
import os
import sys
import time
import json


from tqdm import tqdm
from thop import profile, clever_format


@torch.no_grad()
def inference(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              device: str):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    batch_time = timm.utils.AverageMeter()
    losses = timm.utils.AverageMeter()
    top1 = timm.utils.AverageMeter()
    top5 = timm.utils.AverageMeter()
    model.eval()

    tqdm_loader = tqdm(data_loader, file=sys.stdout)
    for images, labels in tqdm_loader:
        images = images.to(device)
        labels = labels.to(device)

        # compute output and calculate loss and inference time
        torch.cuda.synchronize()
        end = time.time()
        output = model(images)
        loss = criterion(output, labels)
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)

        # measure accuracy and record loss
        batch_size = output.shape[0]
        acc1, acc5 = timm.utils.accuracy(output.detach(), labels, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        tqdm_loader.set_postfix(acc=f"{top1.avg:.2f}", loss=f"{losses.avg:.4f}")

    return batch_time, losses, top1, top5


def evaluate(args: argparse.Namespace):
    if args.tome_r:
        os.environ["TOME_R"] = str(args.tome_r)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. create model
    model = timm.create_model('vit_base_patch16_224.augreg_in1k', 
                              pretrained=False,
                              checkpoint_path=args.pretrained_model_path).to(device)
    data_config = timm.data.resolve_model_data_config(model)

    # 2. create dataset
    dataset = timm.data.create_dataset(
        name='',
        root=args.dataset_path,
        split='val',
    )
    total_images = len(dataset)

    # 3. create dataloader
    batch_size = min(args.batch_size, len(dataset))
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])
    if args.verbosity:
        print('Using {} dataloader workers every process'.format(num_workers))
    data_loader = timm.data.create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=batch_size,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=num_workers,
        crop_pct=data_config['crop_pct'],
        crop_mode=data_config['crop_mode'],
        pin_memory=True,
        device=device,
    )

    # 4. inference
    with torch.inference_mode():
        model.eval()
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        dummy_input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).to(device=device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops, params = clever_format([flops, params], "%.2f")
        for _ in range(5):
            _ = model(dummy_input)

        batch_time, losses, top1, top5 = inference(model, data_loader, device)
    
    # 5.print results
    throughput = total_images / batch_time.sum if batch_time.sum > 0 else 1
    if args.verbosity:
        print(f"\n--- r={args.tome_r} Performance Metrics ---")
        print(f"Device: {device}")
        print(f"FLOPs: {flops}, Params: {params}")
        print(f"Total images processed: {total_images}")
        print(f"Accuracy: {top1.avg:.2f}%")
        print(f"Total inference time: {batch_time.sum:.4f} seconds")
        print(f"Throughput: {throughput:.2f} im/s")
    
    # 6. save results to file
    if args.results_save_path:
        results = {
            "device": str(device),
            "flops": flops,
            "params": params,
            "total_images": total_images,
            "accuracy": top1.avg,
            "total_inference_time": batch_time.sum,
            "throughput": throughput  
        }
        
        with open(os.path.join(args.results_save_path, f"performance_tome_r-{args.tome_r}.json"), "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', 
                        type=str,
                        default='cuda:0',
                        help='Device to use for inference')
    parser.add_argument('--pretrained-model-path', 
                        type=str,
                        default='./checkpoints/vit-base-patch16-224-augreg-in1k/model.safetensors',
                        help='Path to the pretrained model')
    parser.add_argument('--batch-size', 
                        type=int,
                        default=4,
                        help='Batch size for dataloader')
    parser.add_argument('--dataset-path', 
                        type=str,
                        default='./dataset/imagenet1k',
                        help='Path to the dataset')
    parser.add_argument('--results-save-path',
                        type=str,
                        default='./workdir/performance',
                        help='Save results to JSON file')
    parser.add_argument('--tome-r',
                        type=int,
                        default=0,
                        help='Token merging\'s hyper-parameter')
    parser.add_argument('--verbosity', 
                        default=False,
                        action='store_true',
                        help='Enable verbose output')
    args = parser.parse_args()

    for i in range(0, 17):
        args.tome_r = i
        print(f"------------------tome_r: {i}------------------")
        evaluate(args)