import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from alive_progress import alive_bar

from torch.utils.data import DataLoader
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize, Flip

from src.dataloader.dataset import MedicalDataSets
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.CMUNeXt import cmunext
from src.network.conv_based.CMUNet import CMUNet
from src.network.conv_based.ELUnet import ELUnet
from src.network.conv_based.UNeXt import UNext
from src.network.conv_based.UNet3plus import UNet3plus
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.U_Net import U_Net
from src.network.kd_model import get_model, kd_model
from src.network.transfomer_based.transformer_based_network import get_transformer_based_model
from src.utils import losses, kd_loss
from src.utils.kd_loss import BCELoss, MSELoss
from src.utils.metrics import iou_score
from src.utils.util import AverageMeter



def seed_torch(seed):
    if seed==None:
        seed= random.randint(1, 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--teacher_model', type=str, default="DSTUnet",
                    choices=["CMUNeXt", "CMUNet", "AttU_Net", "TransUnet", "R2U_Net", "U_Net", "ESPNetv2","ELUnet","ULite","CGNet","DSTUnet",
                             "PPLiteSeg","CFPNet","DDRNet","UNext", "UNetplus", "UNet3plus", "SwinUnet", "MedT", "TransUnet"], help='model')
parser.add_argument('--student_model', type=str, default="DDRNet",
                    choices=["CMUNeXt", "CMUNet", "AttU_Net", "TransUnet", "R2U_Net", "U_Net", "ESPNetv2","ELUnet","ULite","CGNet","DDRNet",
                             "PPLiteSeg","CFPNet","DDRNet","UNext", "UNetplus", "UNet3plus", "SwinUnet", "MedT", "TransUnet"], help='model')
parser.add_argument('--base_dir', type=str, default="./data/busi", help='dir')
parser.add_argument('--train_file_dir', type=str, default="busi_train.txt", help='dir')
parser.add_argument('--val_file_dir', type=str, default="busi_val.txt", help='dir')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--epoch', type=int, default=150, help='train epoch')
parser.add_argument('--img_size', type=int, default=256, help='img size of per batch')
parser.add_argument('--num_classes', type=int, default=1, help='seg num_classes')
parser.add_argument('--seed', type=int,  help='random seed')
args = parser.parse_args()
seed_torch(args.seed)






def getDataloader(args):
    img_size = args.img_size
    train_transform = Compose([
        RandomRotate90(),
        # transforms.Flip(),
        Flip(),
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])
    db_train = MedicalDataSets(base_dir=args.base_dir, split="train",
                            transform=train_transform, train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    db_val = MedicalDataSets(base_dir=args.base_dir, split="val", transform=val_transform,
                          train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    print("train num:{}, val num:{}".format(len(db_train), len(db_val)))

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
    valloader = DataLoader(db_val, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return trainloader, valloader

def main(args):

    train_result=open(os.path.join("./kd_result", '{}_{}_train_result.txt'.format(args.teacher_model , args.student_model)), 'w')

    base_lr=args.base_lr
    trainloader,valloader=getDataloader(args)
    model=kd_model(output_ch=1,teacher_model=args.teacher_model,student_model=args.student_model)
    print("train file dir:{} val file dir:{}".format(args.train_file_dir, args.val_file_dir))
    # study_optimizer = optim.SGD(model.get_parameters(args.teacher_model), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    seg_optimizer = optim.SGD(model.get_parameters(args.student_model), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    criterion = losses.__dict__['BCEDiceLoss']().cuda()
    # DKL = BCELoss(T=5).cuda()
    DKL = MSELoss().cuda()
    # DKL = losses.__dict__['BCEDiceLoss']().cuda()
    print("{} iterations per epoch".format(len(trainloader)))

    best_iou = 0
    iter_num = 0
    max_epoch = args.epoch

    max_iterations = len(trainloader) * max_epoch

    for epoch_num in range(max_epoch):
        model.train()
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'val_loss': AverageMeter(),
                      'val_iou': AverageMeter(),
                      'val_SE': AverageMeter(),
                      'val_PC': AverageMeter(),
                      'val_F1': AverageMeter(),
                      'val_ACC': AverageMeter()}
        with alive_bar(len(trainloader)+len(valloader), force_tty=True,title="epoch %d/%d"%(epoch_num+1,max_epoch)) as bar:
            for i_batch, sampled_batch in enumerate(trainloader):
                img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                img_batch, label_batch = img_batch.cuda(), label_batch.cuda()

                teacher_outputs,student_outputs = model(img_batch)

                # study_loss=DKL(torch.sigmoid(student_outputs).log(),torch.sigmoid(teacher_outputs ))
                study_loss=DKL(student_outputs,teacher_outputs)
                # study_loss=criterion(student_outputs,torch.sigmoid(teacher_outputs))
                seg_loss = criterion(student_outputs, label_batch)
                total_loss=0.5*study_loss+0.5+seg_loss
                iou, dice, _, _, _, _, _ = iou_score(student_outputs, label_batch)

                seg_optimizer.zero_grad()
                total_loss.backward()
                seg_optimizer.step()

                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in seg_optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1
                avg_meters['loss'].update(seg_loss.item(), img_batch.size(0))
                avg_meters['iou'].update(iou, img_batch.size(0))
                bar()


            model.eval()
            with torch.no_grad():
                for i_batch, sampled_batch in enumerate(valloader):
                    img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    img_batch, label_batch = img_batch.cuda(), label_batch.cuda()
                    output = model.val(img_batch)
                    loss = criterion(output, label_batch)
                    iou, _, SE, PC, F1, _, ACC = iou_score(output, label_batch)
                    avg_meters['val_loss'].update(loss.item(), img_batch.size(0))
                    avg_meters['val_iou'].update(iou, img_batch.size(0))
                    avg_meters['val_SE'].update(SE, img_batch.size(0))
                    avg_meters['val_PC'].update(PC, img_batch.size(0))
                    avg_meters['val_F1'].update(F1, img_batch.size(0))
                    avg_meters['val_ACC'].update(ACC, img_batch.size(0))
                    bar()


            print('epoch [%d/%d]  train_loss : %.4f, train_iou: %.4f - val_loss %.4f - val_iou %.4f - val_SE %.4f - '
                  'val_PC %.4f - val_F1 %.4f - val_ACC %.4f '
                % (epoch_num, max_epoch, avg_meters['loss'].avg, avg_meters['iou'].avg,
                   avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['val_SE'].avg,
                   avg_meters['val_PC'].avg, avg_meters['val_F1'].avg, avg_meters['val_ACC'].avg))
            train_result.write('epoch [%d/%d],  train_loss : %.4f, train_iou: %.4f , val_loss %.4f , val_iou %.4f , val_SE %.4f , '
                  'val_PC %.4f , val_F1 %.4f , val_ACC %.4f '
                % (epoch_num, max_epoch, avg_meters['loss'].avg, avg_meters['iou'].avg,
                   avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['val_SE'].avg,
                   avg_meters['val_PC'].avg, avg_meters['val_F1'].avg, avg_meters['val_ACC'].avg) + '\n')
            train_result.flush()

            if avg_meters['val_iou'].avg > best_iou:
                if not os.path.isdir("./checkpoint"):
                    os.makedirs("./checkpoint")
                torch.save(model.state_dict(), 'checkpoint/{}_{}.pth'.format(args.teacher_model,args.student_model))
                best_iou = avg_meters['val_iou'].avg
                print("=> saved best model")

    return "Training Finished!"


if __name__ == "__main__":
    main(args)


