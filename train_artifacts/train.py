# encoding: utf-8
"""
The emotion recognition Torch models implementation.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import shutil
import matplotlib.pyplot as plt
from test import test, plot_confusion_matrix
from read_data import plotImages, FER2013, get_transforms
from Xception import MiniXception
from tqdm import tqdm
from torchsummary import summary

# CLASS_NAMES = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
CLASS_NAMES= ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
DATA_DIR = '/media/lecun/HD/Grimmat/Emotions Video/Fer2012/FER2013/FER2013'
DATA_SET_PATH = DATA_DIR

SAVE_PATH = './saved_models'
CKPT_PATH = f'{SAVE_PATH}/IOPTFacialSig_adam_best.pth.tar' #'./saved_models/IOPTFacial_sgd_best.pth.tar'

config= {'modelName': 'IEPTFacialSig_x0.3',
    'dataBase': 'FER2013',
    'Classes': CLASS_NAMES,
    'Arq': 'MiniXception',
    'lastAct' : 'sigmoid',
    'checkpoint' : CKPT_PATH,
    'batchSize' : 32,
    'epochs' : 200,
    'loss' : 'CrossEntropy',
    'optim' : 'SGD',
    'lr' : 0.01,
    'momentum': 0.3,
    'l2' : 0.01,
    'lr_decay_patience': 10,
    'stop_patience' : 30}

def main():
    log = open(os.path.join(SAVE_PATH, f"{config['modelName']}_configLog.txt"), "w+")
    text = [line + '\n' for line in str(config).split(',')]
    log.writelines(text)

    print('Training model %s in database %s ...' % (config['modelName'], DATA_SET_PATH.split('/')[-2]))
    global best_acc1
    cudnn.benchmark = True

    # initialize and load the model
    if config['Arq'] is 'MiniXception':
        model = MiniXception(len(CLASS_NAMES), last_activation=config['lastAct']).cuda()
    else:
        raise NotImplemented

    if os.path.isfile(config['checkpoint']):
        print("=> loading checkpoint")
        checkpoint = torch.load(config['checkpoint'])
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    summary(model, input_size=(1, 48, 48))

    for name, block in list(model.named_children()):
       for parameter in list(block.parameters()):
            parameter.requires_grad = True

    model.cuda()

    # print all trainable layers names
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total weigts: ', pytorch_total_params, '\n',
          'Trainable weigths: ', pytorch_trainable_params)

    # exit()

    # Data loaders
    train_data = FER2013(f'{DATA_SET_PATH}Train/', f'{DATA_SET_PATH}Train/labels.csv', get_transforms(True))
    train_loader = DataLoader(train_data, batch_size=config['batchSize'], shuffle=True)

    valid_data = FER2013(f'{DATA_SET_PATH}Valid/', f'{DATA_SET_PATH}Valid/labels.csv', get_transforms(False))
    valid_loader = DataLoader(valid_data, batch_size=config['batchSize'], shuffle=True)

    print('train data:', len(train_loader), 'val data:', len(valid_loader))
    # one_batch = next(iter(train_loader))
    # plotImages(one_batch, labels= CLASS_NAMES, title=config['modelName'], n_images=(4,4))

    # Fit the model

    best_acc, epoch = fit(model,config['epochs'], train_loader, valid_loader, config=config)
    print(f'Model train finished with best acc: {best_acc} at {epoch} epochs')
    log.writelines(f'Best Training acc: {best_acc} \n'
                 f'Trained Epochs: {epoch} \n')

    # Evaluate mode
    test_data = FER2013(f'{DATA_SET_PATH}Test/', f'{DATA_SET_PATH}Test/labels.csv', get_transforms(False))
    test_loader = DataLoader(test_data, batch_size=config['batchSize'], shuffle=True)
    targets, predictions = test(test_loader, model)
    predictions = np.argmax(predictions, axis=-1)

    cm = confusion_matrix(targets, predictions)
    plt.figure()
    plot_confusion_matrix(cm, classes=CLASS_NAMES)
    plt.savefig('{}/{}_cm.png'.format(SAVE_PATH, config['modelName']))
    clf_report = classification_report(targets, predictions, target_names=CLASS_NAMES)
    print(clf_report)
    log.writelines('\n Test Classification Report \n')
    log.writelines(clf_report)
    log.close()

def fit(model, epochs, train_loader, valid_loader, config):

    if 'l2' not in config:
        config['l2'] = None

    if 'lr' not in config:
        config['lr'] = 0.001

    if config['optim'] == 'SGD':
        optimizer = optim.SGD(params=model.parameters(), lr=config['lr'], weight_decay=config['l2'], momentum=config['momentum'])
    else:
        optimizer = optim.Adam(params=model.parameters(), lr=config['lr'], weight_decay=config['l2'])
    optimizer.zero_grad()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9,
                                                     patience=round(config['lr_decay_patience']), verbose=True)
    scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100], gamma=0.9)
    if config['loss'] == 'CrossEntropy':
        loss_func = nn.CrossEntropyLoss()
    else:
        loss_func = nn.BCELoss(reduce='sum')

    train_loss = []
    train_acc = []
    val_loss = []
    val_accuracies = []
    no_best_count = 0
    best_acc1 = 0

    for epoch in range(1, epochs + 1):
        global total, correct, total_train_loss
        total = 0
        correct = 0
        total_train_loss = 0

        t_loss, t_acc = train(epoch, model, train_loader, optimizer, loss_func)

        accuracy = 100. * float(correct) / total
        print('Epoch [%d/%d] Training Loss: %.4f, Accuracy: %.4f' % (
            epoch + 1, epochs, total_train_loss / (epoch + 1), accuracy))

        # evaluate on validation set
        v_loss, v_acc = validate(valid_loader, model, loss_func)
        validation_loss = v_loss.avg
        validation_acc = v_acc.avg
        # remember best acc and save checkpoint
        is_best = validation_acc > best_acc1
        best_acc1 = max(validation_acc, best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': config['modelName'],
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            # 'optimizer': optimizer.state_dict(),
        }, is_best, f'{config["modelName"]}_best', f'{config["modelName"]}_checkpoint')
        scheduler.step(t_loss.avg)
        scheduler2.step()
        if is_best:
            no_best_count = 0
        else: no_best_count = no_best_count + 1

        if no_best_count > config['stop_patience']:
            print('Early stop finish training after {} epochs'.format(epoch))
            break

        train_loss.append(t_loss.avg)
        train_acc.append(t_acc.avg)
        val_loss.append(v_loss.avg)
        val_accuracies.append(v_acc.avg)

    plot_training_stats(train_loss, val_loss,train_acc, val_accuracies,
                        save='{}/{}_plot'.format(SAVE_PATH, config['modelName']))

    return best_acc1, epoch

def train(epoch, model, train_loader, optimizer, loss_func):
    global total, correct, total_train_loss
    losses = AverageMeter()
    top_acc = AverageMeter()

    model.train()
    pbar = tqdm(train_loader)

    for batch in pbar:

        optimizer.zero_grad()
        # if i >15:
        #     break
        # i += 1
        data, target = batch
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        # data, target = Variable(data), Variable(target)

        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        # messure acc and get records.
        acc = accuracy(output, target)

        _, predicted = torch.max(output.data, 1)

        total += target.size(0)
        correct += predicted.eq(target.data).sum()
        total_train_loss += loss.data

        losses.update(loss.item(), data.size(0))
        top_acc.update(acc, data.size(0))

        bar_text = 'Epoch: {0} Loss {loss.val:.4f} ({loss.avg:.4f}) Acc1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, loss=losses, top1=top_acc)
        pbar.set_description(bar_text)

    return losses, top_acc

def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top_acc = AverageMeter()
    # switch to evaluate mode
    model.eval()
    pbar = tqdm(val_loader)

    with torch.no_grad():
        # i = 0
        for (input, target) in pbar:
            # if i > 15:
            #     break
            # i += 1
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top_acc.update(acc1, input.size(0))

            bar_text = 'Validation: Loss {loss.val:.4f} ({loss.avg:.4f}) Acc1 {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses, top1=top_acc)
            pbar.set_description(bar_text)

    return losses, top_acc

def save_checkpoint(state, is_best, bestname='model_best', filename='checkpoint'):
    torch.save(state, '{}/{}.pth.tar'.format(SAVE_PATH, filename))
    if is_best:
        shutil.copyfile('{}/{}.pth.tar'.format(SAVE_PATH, filename), '{}/{}.pth.tar'.format(SAVE_PATH, bestname))

def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    corrects = target.eq(output.argmax(dim=1)).sum()
    batch_size = output.shape[0]
    acc = torch.true_divide(corrects, batch_size).cpu().numpy()
    return acc

def plot_training_stats(t_loss, v_loss, t_acc, v_acc, save=False):
    plt.figure()
    plt.plot(t_loss)
    plt.plot(t_acc)
    plt.plot(v_loss)
    plt.plot(v_acc)
    plt.title('{} model statistics'.format(config['modelName']))
    plt.ylabel('loss - acc')
    plt.xlabel('epoch')
    plt.legend(['T loss', 'T acc', 'V loss', 'V_acc'], loc='upper left')

    if save is not None:
        plt.savefig(f'{save}.png')
    else:
        plt.show()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
