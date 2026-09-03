import os
import os.path as op
import torch
import numpy as np
import random
import time

from datasets import build_dataloader
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize

try:
    import ptflops
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


if __name__ == '__main__':
    args = get_args()
    set_seed(1+get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}')
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)

    # get image-text pair datasets dataloader
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)

    # Compute FLOPs using ptflops (inference only: image + text encoders)
    if PTFLOPS_AVAILABLE:
        model_for_flops = model.module if hasattr(model, 'module') else model
        base = model_for_flops.base_model

        # Image encoder FLOPs
        class ImageEncoderWrapper(torch.nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.clip_model = clip_model
            def forward(self, image):
                return self.clip_model.encode_image(image)

        # Text encoder FLOPs
        class TextEncoderWrapper(torch.nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.clip_model = clip_model
            def forward(self, text):
                return self.clip_model.encode_text(text)

        img_wrapper = ImageEncoderWrapper(base)
        txt_wrapper = TextEncoderWrapper(base)

        img_macs, _ = ptflops.get_model_complexity_info(
            img_wrapper, (3, args.img_size[0], args.img_size[1]),
            input_constructor=lambda _: torch.randn(1, 3, args.img_size[0], args.img_size[1]).half().to(device),
            print_per_layer_stat=False, verbose=False, as_strings=False
        )
        txt_macs, _ = ptflops.get_model_complexity_info(
            txt_wrapper, (77,),
            input_constructor=lambda _: torch.randint(0, 49408, (1, 77)).long().to(device),
            print_per_layer_stat=False, verbose=False, as_strings=False
        )
        total_macs = img_macs + txt_macs
        logger.info(f'Image encoder FLOPs: {img_macs / 1e9:.2f} GMac')
        logger.info(f'Text encoder FLOPs: {txt_macs / 1e9:.2f} GMac')
        logger.info(f'Total inference FLOPs: {total_macs / 1e9:.2f} GMac')
    else:
        logger.info('ptflops not installed, FLOPs not available')

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader)

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']

    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)