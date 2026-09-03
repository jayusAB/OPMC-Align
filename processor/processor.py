import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable

try:
    import ptflops
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "wbcmcir_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter(),
        "lse_pos": AverageMeter(),
        "lse_neg": AverageMeter(),
        "calib_ratio_pos": AverageMeter(),
        "calib_ratio_neg": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()
        _base = model.module if hasattr(model, 'module') else model
        if hasattr(_base, 'set_epoch'):
            _base.set_epoch(epoch)

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            ret = model(batch)

            # Hot-start: disable wbcmcir_loss for the first wbcmcir_warmup_epochs
            if epoch < args.wbcmcir_warmup_epochs:
                ret.pop('wbcmcir_loss', None)
                ret.pop('wbcmcir_stats', None)

            # Unpack wbcmcir_stats for logging
            stats = ret.pop('wbcmcir_stats', None)

            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
            meters['wbcmcir_loss'].update(ret.get('wbcmcir_loss', 0), batch_size)

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)

            if stats is not None:
                for _k, _v in stats.items():
                    if _k not in meters:
                        meters[_k] = AverageMeter()
                    meters[_k].update(_v, batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.count > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.count > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
            # Measure inference FPS (image + text encoding only)
            if PTFLOPS_AVAILABLE and epoch == start_epoch:
                try:
                    model_for_fps = model.module if hasattr(model, 'module') else model
                    base = model_for_fps.base_model
                    model_for_fps.eval()
                    dummy_image = torch.randn(train_loader.batch_size, 3, args.img_size[0], args.img_size[1]).half().to(device)
                    dummy_text = torch.randint(0, 49408, (train_loader.batch_size, 77)).long().to(device)
                    with torch.no_grad():
                        # Warmup
                        for _ in range(5):
                            _ = base.encode_image(dummy_image)
                            _ = base.encode_text(dummy_text)
                        # Measure
                        torch.cuda.synchronize()
                        fps_start = time.time()
                        for _ in range(50):
                            _ = base.encode_image(dummy_image)
                            _ = base.encode_text(dummy_text)
                        torch.cuda.synchronize()
                        fps_end = time.time()
                    fps = 50 * train_loader.batch_size / (fps_end - fps_start)
                    logger.info(f'Inference FPS: {fps:.1f} (batch_size={train_loader.batch_size})')
                except Exception as e:
                    logger.warning(f'FPS measurement failed: {e}')
                finally:
                    model_for_fps.train()
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader, args=None):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    # Measure FPS (image + text encoding only)
    if PTFLOPS_AVAILABLE and args is not None:
        device = "cuda"
        base = model.base_model
        model.eval()
        dummy_image = torch.randn(test_img_loader.batch_size, 3, args.img_size[0], args.img_size[1]).half().to(device)
        dummy_text = torch.randint(0, 49408, (test_txt_loader.batch_size, 77)).long().to(device)
        with torch.no_grad():
            for _ in range(5):
                _ = base.encode_image(dummy_image)
                _ = base.encode_text(dummy_text)
            torch.cuda.synchronize()
            fps_start = time.time()
            for _ in range(50):
                _ = base.encode_image(dummy_image)
                _ = base.encode_text(dummy_text)
            torch.cuda.synchronize()
            fps_end = time.time()
        fps = 50 * test_img_loader.batch_size / (fps_end - fps_start)
        logger.info(f'Inference FPS: {fps:.1f} (batch_size={test_img_loader.batch_size})')

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
