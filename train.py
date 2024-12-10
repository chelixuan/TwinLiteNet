import os
import torch
import pickle
from model import TwinLite as net
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
from argparse import ArgumentParser
from utils import train, val, netParams, save_checkpoint, poly_lr_scheduler
import torch.optim.lr_scheduler

from loss import TotalLoss

# added by clx ------------------------------------------------------
import math

initial_lr = 0.001
gamma = 0.1

# 通过迭代次数控制 lr decay
# def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
#     """Sets the learning rate
#     # Adapted from PyTorch Imagenet example:
#     # https://github.com/pytorch/examples/blob/master/imagenet/main.py
#     """
#     warmup_epoch = -1
#     if epoch <= warmup_epoch:
#         lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
#     else:
#         lr = initial_lr * (gamma ** (step_index))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr

# 通过 epoch 控制 lr decay
def adjust_learning_rate(optimizer, gamma, epoch, decay1, decay2):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    lr = initial_lr
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * epoch / warmup_epoch
    elif epoch >= decay1 and epoch <= decay2:
        lr = initial_lr * gamma
    elif epoch > decay2:
        lr = initial_lr * (gamma ** (2))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
# -------------------------------------------------------------------

def train_net(args):
    # load the model
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    model = net.TwinLiteNet()

    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    args.savedir = args.savedir + '/'

    # create the directory if not exist
    os.makedirs(args.savedir, exist_ok=True)

    # original ----------------------------------------------------------------------------------------
    # trainLoader = torch.utils.data.DataLoader(
    #     myDataLoader.MyDataset(),
    #     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # valLoader = torch.utils.data.DataLoader(
    #     myDataLoader.MyDataset(valid=True),
    #     batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # -------------------------------------------------------------------------------------------------- 
    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(
            train_path = args.train, val_path = args.val, 
            input_height = args.height, input_width = args.width,
            ),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    if args.validation:
        valLoader = torch.utils.data.DataLoader(
            myDataLoader.MyDataset(
                train_path = args.train, val_path = args.val,
                input_height = args.height, input_width = args.width, valid=True
                ),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # --------------------------------------------------------------------------------------------------

    if cuda_available:
        args.onGPU = True
        model = model.cuda()
        cudnn.benchmark = True

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    # criteria = TotalLoss()
    # clx edited @20241018 for focal / bce --------------------------------------------------------------
    # criteria = TotalLoss(args.pixel_error_loss, )
    # clx edited @ 20241028 for loss weight
    criteria = TotalLoss(args.pixel_error_loss, args.alpha, args.beta)
    # ---------------------------------------------------------------------------------------------------

    start_epoch = 0
    lr = args.lr

    # original ------------------------------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    # clx -----------------------------------------------------------------------------------------------
    # optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)
    # ---------------------------------------------------------------------------------------------------

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    for epoch in range(start_epoch, args.max_epochs):

        model_file_name = args.savedir + os.sep + 'model_{}.pth'.format(epoch)
        # original ----------------------------------------------------------------------------------------
        poly_lr_scheduler(args, optimizer, epoch)
        # clx sgd------------------------------------------------------------------------------------------
        # decay1 = int(0.7 * args.max_epochs)
        # decay2 = int(0.85 * args.max_epochs)
        # lr = adjust_learning_rate(optimizer, gamma, epoch, decay1, decay2)
        # -------------------------------------------------------------------------------------------------

        
        # original -- 代码应该是错的 -------------------------------------------------------------------------
        # for param_group in optimizer.param_groups:
        #     lr = param_group['lr']
        # clx ----------------------------------------------------------------------------------------------
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # --------------------------------------------------------------------------------------------------
        print("Learning rate: " +  str(lr))

        # train for one epoch
        model.train()
        # clx add max_epochs parameters
        train(args, trainLoader, model, criteria, optimizer, epoch, args.max_epochs)
        model.eval()
        # original validation --------
        # val(valLoader, model)

        # clx ----------------------------------------------------------------------------------------------
        if args.validation:
            val(valLoader, model)
        # --------------------------------------------------------------------------------------------------

        # 修改 ckpt 保存频率
        if (epoch + 1) % 100 == 0 or epoch == args.max_epochs or epoch+1 == args.max_epochs:
            torch.save(model.state_dict(), model_file_name)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr': lr
            }, args.savedir + 'checkpoint.pth.tar')
        


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=100, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=12, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                   'Change as per the GPU memory')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--savedir', default='./test_', help='directory to save the results')
    parser.add_argument('--resume', type=str, default='', help='Use this flag to load last checkpoint for training')
    parser.add_argument('--pretrained', default='', help='Pretrained ESPNetv2 weights.')

    # added by clx @ 20240711
    # parser.add_argument('--height', type=int, default=720, help='input_image_height')
    # parser.add_argument('--width', type=int, default=1280, help='input_image_width')
    parser.add_argument('--height', type=int, default=360, help='input_image_height')
    parser.add_argument('--width', type=int, default=640, help='input_image_width')
    # added by clx @ 20241018
    parser.add_argument('--pixel_error_loss', type=str, default="add", help='pixel_error_loss, can be focal/bce/add/mean')
    # added by clx @ 2024.10.28
    parser.add_argument('--alpha', type=float, default=1.0, help='weight of pixel_error_loss')
    parser.add_argument('--beta', type=float, default=1.0, help='weight of tversky_loss')
    # added by clx @ 2024.12.10
    parser.add_argument('--train', 
                        type=str, 
                        default="/home/chelx/dataset/seg_images/images/train/train_batch_01_lyon2024/", 
                        help='train image path')
    parser.add_argument('--val', 
                        type=str, 
                        default="/home/chelx/dataset/seg_images/images/val/val_batch_01_202409_lyon/", 
                        help='val image path')
    parser.add_argument('--validation', type=bool, default=True, help="if val every epoch")
    train_net(parser.parse_args())

