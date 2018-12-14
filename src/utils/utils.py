"""Utils for train and predict scripts."""
from collections import OrderedDict
from datetime import datetime as dt
from torch import nn, optim
from torchvision import models
import random
import string
import torch

DROPOUT = 0.5
LEARNING_RATE = 0.001
HIDDEN_UNITS_LI = [80, 90, 80]
TRAINING_EPOCHES = 8
OUTPUT_CLASSES = 102

INITIAL_UNITS = {
    'vgg16': 25088,
    'densenet121': 1024,
    'alexnet': 9216
}


def init_model(cnn_arch=list(INITIAL_UNITS.keys())[0],
               dropout=DROPOUT,
               hidden_units_li=HIDDEN_UNITS_LI,
               output_classes=OUTPUT_CLASSES,
               lr=LEARNING_RATE):
    """Initializes CNN. Returns model, criterion and optimizer"""
    # Initialize the model

    if cnn_arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        model = ''
        print(f'Architecture not listed in {INITIAL_UNITS.keys()}')

    # Freeze params
    try:
        for param in model.parameters():
            param.requires_grad = False
    except AttributeError:
        raise
    except Exception as e:
        print(f'Error encountered: {e}')
        raise

    h_l = len(hidden_units_li)

    h_l_li = []
    for i in range(0, h_l-1):
        h_l_li.append((f'hidden_layer{i+1}', nn.Linear(
            hidden_units_li[i], hidden_units_li[i+1])))
        h_l_li.append((f'relu{i+2}', nn.ReLU()))

    initial_layer = [('dropout', nn.Dropout(dropout)),
                     ('inputs', nn.Linear(
                         INITIAL_UNITS[cnn_arch], hidden_units_li[0])),
                     ('relu1', nn.ReLU())]

    final_layer = [(f'hidden_layer{h_l}', nn.Linear(hidden_units_li[h_l-1], output_classes)),
                   ('output', nn.LogSoftmax(dim=1))]

    # Replace the classifier
    classifier = nn.Sequential(
        OrderedDict(
            initial_layer + h_l_li + final_layer
        )
    )

    model.classifier = classifier

    if torch.cuda.is_available():
        model.cuda()

    # Initialize criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    return model, criterion, optimizer


def construct_model_filename():
    """Construct an identifier for modelling dataframe (mdf).

    Includes name includes "mdf_", date created and 6 alphanumeric str.
    """
    random_string = create_random_six_chars()
    todays_date = dt.now().strftime("%Y-%m-%d")
    return f'model_{todays_date}_{random_string}.pth'


def create_random_six_chars():
    """Generate random 6 characters."""
    random_chars = 6
    random_string = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits)
                            for _ in range(random_chars))
    return random_string
