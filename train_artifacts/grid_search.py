import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from read_data import plotImages, FER2013, get_transforms
from Xception import MiniXception
from skorch import NeuralNet
from skorch.helper import SliceDataset
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, accuracy_score
import pandas as pd
from torch import optim

# CLASS_NAMES = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
CLASS_NAMES= ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
DATA_DIR = '/media/lecun/HD/Grimmat/Emotions Video/Fer2012/FER2013/FER2013'
DATA_SET_PATH = DATA_DIR

SAVE_PATH = './saved_models'


config= {'modelName': 'IOPTFacialSig_adamFT',
    'dataBase': 'FER2013',
    'Classes': CLASS_NAMES,
    'Arq': 'MiniXception',
    'epochs' : 15,
    'lr' : 0.01,
    'optimizer': optim.SGD,
    'batch_size': 32}


params = {
    'optimizer__weight_decay': [0.001, 0.005, 0.01],
    'optimizer__momentum': [0, 0.3, 0.9],
    # 'batch_size': [32, 64, 128],
    'module__last_activation': ['softmax', 'sigmoid', None]}

def auc_score(y, y_pred):
    # y_pred = y_pred.argmax(axis=-1)
    score = roc_auc_score(y, y_pred, multi_class='ovr')
    return score

score = make_scorer(auc_score)

def main():

    cudnn.benchmark = True

    # initialize and load the model

    model = NeuralNet(MiniXception,
                      module__n_classes = len(CLASS_NAMES),
                      lr= config['lr'],
                      criterion=nn.CrossEntropyLoss,
                      device='cuda',
                      max_epochs=config['epochs'],
                      batch_size=config['batch_size'],
                      optimizer=config['optimizer'])
    acc = ['precision_weighted', 'recall_weighted', 'f1_weighted']

    gs = GridSearchCV(model, params, refit=False, cv=2, scoring=score, verbose=3, n_jobs=6)

    # Data loaders
    train_data = FER2013(f'{DATA_SET_PATH}Train/', f'{DATA_SET_PATH}Train/labels.csv', get_transforms(True), balance=3000)
    # train_loader = DataLoader(train_data, batch_size=config['batchSize'], shuffle=True)

    valid_data = FER2013(f'{DATA_SET_PATH}Valid/', f'{DATA_SET_PATH}Valid/labels.csv', get_transforms(False))
    # valid_loader = DataLoader(valid_data, batch_size=config['batchSize'], shuffle=True)

    slice_x = SliceDataset(train_data)
    slice_y = SliceDataset(train_data, idx=1)
    #
    # t_list = []
    # my_iter = iter(slice_x)
    # iter_y = iter(slice_y)
    # for i, tensor,y  in zip(range(10),my_iter, iter_y):
    #     t_list.append(tensor)
    #     print(CLASS_NAMES[int(y.int())])
    #
    # from read_data import plotTensorList
    # plotTensorList( t_list)
    #
    # exit()

    # print('train data:', len(train_loader), 'val data:', len(valid_loader))

    gs.fit(slice_x, slice_y)
    print(gs.cv_results_)
    cv_results = pd.DataFrame(gs.cv_results_)
    cv_results.to_csv('GridSearchResults2.csv')


if __name__ == '__main__':
    main()