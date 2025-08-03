import argparse
import utils


from src import vit_base_patch16_224_tome as vit_tome


if __name__ == '__main__':
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

    
    # for i in range(2, 17):
    #     print(f"------------------tome_r: {i}------------------")
    #     args.tome_r = i
    #     vit_tome.evaluate(args)
    utils.plot_performance()