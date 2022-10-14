import argparse
import os
import time

import matplotlib.pyplot as plt
import torch
from torch.optim import SGD, Adam
from torchvision import utils

from utils import create_dataloader, YOLOv1Loss, parse_cfg, build_model
from brevitas.export import StdQOpONNXManager, PytorchQuantManager, PyXIRManager

from torchviz import make_dot

parser = argparse.ArgumentParser(description='YOLOv1-pytorch')
parser.add_argument("--cfg", "-c", default="/home/blattst/git/pizza/Quantized-yolov1/cfg/config.yaml", help="Project config file path", type=str)
parser.add_argument("--weights", "-w", default="", help="Pretrained model weights path", type=str)
parser.add_argument("--output", "-o", default="output", help="Output path", type=str)
parser.add_argument("--epochs", "-e", default=100, help="Training epochs", type=int)
parser.add_argument("--lr", "-lr", default=0.002, help="Training learning rate", type=float)
parser.add_argument("--save_freq", "-sf", default=10, help="Frequency of saving model checkpoint when training",
                    type=int)
args = parser.parse_args()


def train(model, train_loader, optimizer, epoch, device, S, B, train_loss_lst):
    model.train()  # Set the module in training mode
    train_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        t_start = time.time()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # back prop
        criterion = YOLOv1Loss(S, B)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        t_batch = time.time() - t_start

        # show batch0 dataset
        if epoch == 0:
            fig = plt.figure()
            inputs = inputs.cpu()  # convert to cpu
            grid = utils.make_grid(inputs)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.savefig(os.path.join(output_path, 'batch{}.png'.format(batch_idx)))
            # plt.show()
            plt.close(fig)

        # print loss and accuracy
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]  Time: {:.4f}s  Loss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), t_batch, loss.item()))

    # record training loss
    train_loss /= len(train_loader)
    train_loss_lst.append(train_loss)
    return train_loss_lst


def validate(model, val_loader, device, S, B, val_loss_lst):
    model.eval()  # Sets the module in evaluation mode
    val_loss = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # add one batch loss
            criterion = YOLOv1Loss(S, B)
            val_loss += criterion(output, target).item()

    val_loss /= len(val_loader)
    print('Val set: Average loss: {:.4f}\n'.format(val_loss))

    # record validating loss
    val_loss_lst.append(val_loss)
    return val_loss_lst


def test(model, test_loader, device, S, B):
    model.eval()  # Sets the module in evaluation mode
    test_loss = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # add one batch loss
            criterion = YOLOv1Loss(S, B)
            test_loss += criterion(output, target).item()

    # record testing loss
    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    cfg = parse_cfg(args.cfg)
    img_path = cfg['train_path']
    S, B, num_classes, input_size = cfg['S'], cfg['B'], len(cfg['class_names']), cfg['input_size']

    # create output file folder
    start = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    output_path = os.path.join(args.output, start)
    os.makedirs(output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model
    model = build_model(args.weights, S, B, num_classes).to(device)

    # plot model structure
    graph = make_dot(model(torch.rand(1, 3, input_size, input_size).to(device)),
                     params=dict(model.named_parameters()))
    graph.render('model_structure', './', cleanup=True, format='png')


    

    # get data loader
    train_loader, val_loader, test_loader = create_dataloader(cfg['train_path'], cfg['valid_path'], cfg['test_path'], cfg['batch_size'],
                                                              input_size, S, B, cfg['class_names'])

    #optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)

    train_loss_lst, val_loss_lst = [], []

    print("Start Training with {} train, {} test, and {} valid images".format(len(train_loader), len(test_loader), len(val_loader)))

    
    # train epoch
    for epoch in range(args.epochs):
        train_loss_lst = train(model, train_loader, optimizer, epoch, device, S, B, train_loss_lst)
        val_loss_lst = validate(model, val_loader, device, S, B, val_loss_lst)

        # save model weight every save_freq epoch
        if epoch % args.save_freq == 0 and epoch >= args.epochs / 2:
            torch.save(model.state_dict(), os.path.join(output_path, 'epoch' + str(epoch) + '.pth'))

    test(model, test_loader, device, S, B)

    # save model
    torch.save(model.state_dict(), os.path.join(output_path, 'last.pth'))
    
    # export model 
    FINNManager.export(model, input_shape=(1, 3, 32, 32), export_path=os.path.join(output_path, 'finn_lenet.onnx'))
    StdQOpONNXManager.export(model, input_shape=(1, 3, 32, 32), export_path=os.path.join(output_path, 'onnx_lenet.onnx'))
    PyXIRManager.export(model, input_shape=(1, 3, 32, 32), export_path=os.path.join(output_path,'pyxir_lenet.onnx'))

    # plot loss, save params change
    fig = plt.figure()
    plt.plot(range(args.epochs), train_loss_lst, 'g', label='train loss')
    plt.plot(range(args.epochs), val_loss_lst, 'k', label='val loss')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc-loss')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_path, 'loss_curve.jpg'))
    plt.show()
    plt.close(fig)
