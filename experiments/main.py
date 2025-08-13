from src.train_vit import run, defaultargs
# from src.val_vit_tome import evaluate, defaultargs
from utils.plot import plot_performance2, plot_performance, plot_from_csv


model = ['vit_tiny_patch16_224.augreg_in21k_ft_in1k',
         'deit_tiny_patch16_224.fb_in1k',

         'vit_small_patch16_224.augreg_in1k',
         'deit_small_patch16_224.fb_in1k',

         'vit_base_patch16_224.augreg_in1k',
         'deit_base_patch16_224.fb_in1k',

         'vit_large_patch16_224.augreg_in21k_ft_in1k',
         'deit3_large_patch16_224.fb_in22k_ft_in1k']

model_path = ['vit-tiny-patch16-224-augreg-in21k-ft-in1k',
              'deit-tiny-patch16-224-fb-in1k',

              'vit-small-patch16-224-augreg-in1k',
              'deit-small-patch16-224-fb-in1k',

              'vit-base-patch16-224-augreg-in1k',
              'deit-base-patch16-224-fb-in1k',

              'vit-large-patch16-224-augreg-in21k-ft-in1k',
              'deit3-large-patch16-224-fb-in22k-ft-in1k']


if __name__ == '__main__':
    # run(defaultargs())
    # args = defaultargs()

    # for i in range(len(model)):
    #     args.model = model[i]
    #     args.pretrained_model_path = f'./checkpoints/{model_path[i]}/model.safetensors'
    #     args.results_save_path = f'./workdir/{model_path[i]}.perf'

    #     for r in range(0, 17 if i < 6 else 9):
    #         args.tome_r = r
    #         print(f"************** {model[i]}, tome_r: {r} **************")
    #         evaluate(args)
    # plot_performance2()
    # plot_performance(perf_dir= "./workdir/deit3-large-patch16-224-fb-in22k-ft-in1k.perf",
    #                  save_path = "./workdir/deit3-large-patch16-224-fb-in22k-ft-in1k.perf/performance.png")
    
    plot_from_csv(csv_path="./checkpoints/my-vit-tiny-patch16-224/20250810-062923-vit_tiny_patch16_224/summary.csv",
                  save_path="./checkpoints/my-vit-tiny-patch16-224/20250810-062923-vit_tiny_patch16_224/summary.png")