"""Script to train pytorch model."""
from collections import Counter
from datetime import datetime as dt
from torchvision import (models, transforms, datasets)
import torch
import torch.utils.data as data

from .utils import utils
from .utils.utils import (
    HIDDEN_UNITS_LI,
    DROPOUT,
    LEARNING_RATE,
    HIDDEN_UNITS,
    TRAINING_EPOCHES)


MODEL_FILEPATH = './models'


def load_data(folder_path, train_batch_size=5, valid_batch_size=5):
    """Loads from data folder. Returns test and train dataloader iterables."""
    # Params
    train_dir = folder_path + '/train'
    valid_dir = folder_path + '/valid'
    img_dim = 224
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]

    # Define your transforms for the training and validation sets
    data_transforms = {'train':
                       transforms.Compose([
                           transforms.RandomRotation(degrees=45),
                           transforms.RandomResizedCrop(
                               img_dim, scale=(0.8, 1.3)),
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomVerticalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize(
                               mean=normalize_mean, std=normalize_std)
                       ]),
                       'valid':
                       transforms.Compose([
                           transforms.Resize(img_dim),
                           transforms.CenterCrop((img_dim, img_dim)),
                           transforms.ToTensor(),
                           transforms.Normalize(
                               mean=normalize_mean, std=normalize_std)
                       ])
                       }

    # Load the datasets with ImageFolder
    image_datasets_train = datasets.ImageFolder(
        root=train_dir, transform=data_transforms['train'])
    image_datasets_valid = datasets.ImageFolder(
        root=valid_dir, transform=data_transforms['valid'])

    print(f'{dt.now()} Loading images now.')
    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders_train = data.DataLoader(image_datasets_train,
                                        batch_size=train_batch_size,
                                        shuffle=True,
                                        num_workers=4)
    dataloaders_valid = data.DataLoader(image_datasets_valid,
                                        batch_size=valid_batch_size,
                                        shuffle=True,
                                        num_workers=4)

    # Get number of classes
    class_counter = dict(Counter(img[1] for img in image_datasets_train.imgs))
    classes = len(class_counter)
    print(f'Number of classes in training data: {classes}')
    print(f'Total number of training examples: {sum(class_counter.values())}')

    class_to_idx = image_datasets_train.class_to_idx

    return dataloaders_train, dataloaders_valid, class_to_idx, classes


def train_model(model,
                criterion,
                optimizer,
                train_dataloader,
                valid_dataloader,
                epochs=TRAINING_EPOCHES):
    """Trains the CNN model. Prints the loss through the epochs."""

    # Params
    print_every_step = 10

    # Initialize
    running_loss = 0
    check_cuda = torch.cuda.is_available()

    print(f'{dt.now()} Model training has started.')

    # Start training
    for epoch in range(epochs):
        running_loss = 0
        for step, (train_images, train_labels) in enumerate(train_dataloader):

            # Move to cuda if GPU is used
            if check_cuda:
                train_images, train_labels = train_images.to(
                    'cuda'), train_labels.to('cuda')

            # Reset gradient for training
            optimizer.zero_grad()

            # Forward propagation
            train_logps = model.forward(train_images)
            train_loss = criterion(train_logps, train_labels)
            running_loss += train_loss.item()
            # loss is a Tensor w shape (1,), .item() get scalar

            # Backward propagation
            train_loss.backward()

            # Step forward
            optimizer.step()

            if step % print_every_step == 0:
                model.eval()  # To evaluation mode
                valid_loss = 0
                accuracy = 0

                # At every print step, we evaluate the model
                for _, (valid_images, valid_labels) in enumerate(valid_dataloader):
                    optimizer.zero_grad()

                    if check_cuda:
                        valid_images, valid_labels = \
                            valid_images.to('cuda'), valid_labels.to('cuda')
                        model.to('cuda')

                    valid_logps = model.forward(valid_images)
                    loss = criterion(valid_logps, valid_labels)
                    valid_loss += loss.item()

                    # Calculate accuracy
                    probabilities = torch.exp(valid_logps)
                    _, top_classes = probabilities.topk(1, dim=1)
                    prop_accurate_labels = \
                        top_classes == valid_labels.view(*top_classes.shape)
                    accuracy += torch.mean(
                        prop_accurate_labels.type(torch.FloatTensor)).item()

                # Print
                validation_size = len(valid_dataloader)
                print(f'{dt.now()} Epoch {epoch+1}/{epochs}..'
                      f'Train loss: {running_loss/print_every_step: .3f}..'
                      f'Validation loss: {valid_loss/validation_size: .3f}..'
                      f'Validation accuracy: {accuracy/validation_size: .3f}..')

                # Reset
                running_loss = 0

                # Back to training mode
                model.train()

    print(f'{dt.now()} Model training has ended.')


def save_model(model, cnn_arch, hidden_units_li, dropout, learning_rate, output_classes,
               class_to_idx,
               filepath=MODEL_FILEPATH):
    """Saves model parameters to path."""
    model.cpu()

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    checkpoint = {
        'idx_to_class': idx_to_class,
        'output_classes': output_classes,
        'cnn_arch': cnn_arch,
        'hidden_units_li': hidden_units_li,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'state_dict': model.state_dict(),
    }
    filename = utils.construct_model_filename()
    torch.save(checkpoint, f'{filepath}/{filename}')
    print(f'Model saved at {filepath}/{filename}')


if __name__ == '__main__':

    params = {
        'data_filepath': './data/flower_data',
        'train_batch_size': 128,
        'validation_batch_size': 64,
        'model_arch': 'vgg16',
        'dropout': DROPOUT,
        'hidden_units_li': HIDDEN_UNITS_LI,
        'learning_rate': LEARNING_RATE,
    }

    # Load test and train data data
    print(f'{dt.now()} Loading test and train data.')
    dataloaders_train, dataloaders_valid, class_to_idx, classes = load_data(
        params['data_filepath'],
        params['train_batch_size'],
        params['validation_batch_size'])

    # Initialize model
    print(f'{dt.now()} Initializing model of choice.')
    model, criterion, optimizer = \
        utils.init_model(params['model_arch'],
                         params['dropout'],
                         params['hidden_units_li'],
                         classes,
                         params['learning_rate'])

    # Train model
    print(f'{dt.now()} Train model.')
    train_model(model, criterion, optimizer,
                dataloaders_train, dataloaders_valid)

    # Save model
    print(f'{dt.now()} Saving model weights, hyperparameters and maps.')
    save_model(model,
               params['model_arch'],
               params['hidden_units_li'],
               params['dropout'],
               params['learning_rate'],
               classes,
               class_to_idx)
