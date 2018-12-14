"""Script to predict using pytorch model."""
from datetime import datetime as dt
from torch import transforms
import Image
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import torch

from .utils import utils

TOP_K = 5
MODEL_FILEPATH = './models'


def load_model(filename, filepath=MODEL_FILEPATH):
    """Loads a model checkpoint."""
    print(f'Loading model from {filepath}/{filename}')
    ckpt = torch.load(f'{filepath}/{filename}')

    print(f'Model loaded.')
    print(ckpt.keys())

    # Initialize model
    model, _, _ = utils.init_model(ckpt['cnn_arch'],
                                   ckpt['dropout'],
                                   ckpt['hidden_units_li'],
                                   ckpt['output_classes'],
                                   ckpt['learning_rate'])

    # Initialize class_to_idx and weights
    model.idx_to_class = ckpt['idx_to_class']
    model.load_state_dict(ckpt['state_dict'])

    return model


def load_dictionary(filepath):
    """Load data dictionary."""
    with open(f'{filepath}/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def predict_image(filepath, model, topk=5):
    """Make prediction for an image."""
    # Open and preprocess image
    image = Image.open(filepath)
    img_tensor = process_image(image)
    print(f'Tensor shape: {img_tensor.shape}')

    # Forward pass with CNN
    with torch.no_grad():
        if torch.cuda.is_available():
            model.cuda()
            output = model.forward(img_tensor.cuda())
        else:
            output = model.forward(img_tensor)

    probability = torch.exp(output)
    top_probabilities, top_classes = probability.topk(topk, dim=1)

    # convert probs and classes to list
    top_probabilities = list(top_probabilities.cpu().numpy()[0])
    top_classes = list(top_classes.cpu().numpy()[0])
    top_classes = [model.idx_to_class[x] for x in top_classes]

    return top_probabilities, top_classes


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # configs
    basewidth = 224
    colors = 255

    # centre crop and resize
    width, height = image.size
    new_width = new_height = min(width, height)

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    image = image.crop((left, top, right, bottom)
                       ).resize((basewidth, basewidth))

    # convert to array
    img_arr = np.array(image)

    # normalize by to 0 and 1
    img_arr = img_arr / float(colors)

    # normalize by specific method
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_arr = (img_arr - mean) / std

    # rearrange
    return torch.tensor(img_arr.transpose((2, 0, 1)))


def plot_img_metrics(filename, filepath, probs, classes, dictionary):
    """Plot top k classes for the image."""
    # open/ preprocess image
    img = Image.open(f'{filepath}/{filename}')
    img_processed = process_image(img).numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_processed = std * img_processed + mean
    img_processed = np.clip(img_processed, 0, 1)

    # preprocess
    max_prob_idx = probs.index(max(probs))
    classes_num = [int(no) for no in classes]
    classes_name = [' '.join(
        [x.capitalize() for x in dictionary[str(no)].split()]) for no in classes_num]

    # setup image
    yticks = np.arange(len(classes))
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    # show flower image
    ax[0].imshow(img_processed)
    ax[0].set_title('Flower with highest likelihood - ' +
                    classes_name[max_prob_idx])

    # show barchart with probabilities
    ax[1].barh(yticks, probs)
    ax[1].set_yticks(yticks)
    ax[1].set_yticklabels(classes_name)
    ax[1].invert_yaxis()
    ax[1].set_title('Probabilities')

    output_filename = re.sub('.jpg', '', filename)
    fig.save(f'{filename}/{output_filename}_topk-plot.jpg')


if __name__ == '__main__':
    # utils.test()

    params = {
        'model_filename': 'model_checkpoint.pth',
        'dictionary_filepath': './',
        'pimage_filename': 'filename',
        'pimage_filepath': './'
    }

    # Load model
    print(f'{dt.now()} Loading model.')
    model = load_model(params['model_filename'])

    # Open dictionary
    print(f'{dt.now()} Loading dictionary.')
    cat_to_name = load_dictionary(params['model_filename'])

    # Predict
    print(f'{dt.now()} Making prediction for image.')
    top_probabilities, top_classes = predict_image(
        params['pimage_filename'],
        params['pimage_filepath'],
        5)

    # Plotting metrics
    print(f'{dt.now()} Plotting metrics for image.')
    plot_img_metrics(params['pimage_filename'],
                     params['pimage_filepath'],
                     top_probabilities, top_classes, cat_to_name)
