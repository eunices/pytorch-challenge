# Lecture 33 in CNN

# __init__
self.conv1 = nn.Conv2d(in_channels, out_channels,
                       kernel_size, stride=1, padding=0)
self.pool = nn.MaxPool2d(2, 2)


# forward
x = F.relu(self.convl(x))
x = self.pool(x)


# using sequential
def __init__(self):
        super(ModelName, self).__init__()
        self.features = nn.Sequential(
              nn.Conv2d(1, 16, 2, stride=2),
              nn.MaxPool2d(2, 2),
              nn.ReLU(True),

              nn.Conv2d(16, 32, 3, padding=1),
              nn.MaxPool2d(2, 2),
              nn.ReLU(True)
         )


class Net(nn.Module):
      def __init__(self):
    super(Net, self).__init__()
    
    hidden_1 = 512
    hidden_2 = 512
    
    self.fc1 = nn.Linear(img_dim*img_dim*3, hidden_1)
    self.fc2 = nn.Linear(hidden_1, hidden_2)
    self.fc3 = nn.Linear(hidden_2, 10)
    
    self.dropout = nn.Dropout(0.2)
    
  def forward(self, x):
    x = x.view(-1, img_dim*img_dim*3)
    x = F.relu(self.fcl(x))
    x = self.dropout(x)
    x = F.relu(self.fc2(x))
    x = self.dropout(x)
    return x
 
model = Net()
print(model)

# def load_checkpoint(filepath):
#     checkpoint = torch.load(filepath)
#     model = Network(checkpoint['input_size'], checkpoint['output_size'], checkpoint['hidden_layers'])
#     model.load_state_dict(checkpoint['state_dict'])
#     return model

# model = load_checkpoint('model/checkpoint.pth')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

vgg16 = models.vgg16(pretrained=True)

# freeze parameters
for param in vgg16.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(img_dim*img_dim*3, 500)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(p=0.2)),
    ('fc2', nn.Linear(500, len(species_counts))),
    ('output', nn.LogSoftmax(dim=1))
]))

torch.cuda.is_available()

criterion = nn.NLLLoss()

optimizer = optim.Adam(vgg16.classifier.parameters(), lr=0.003)

vgg16.cuda()

# Lecture 21 on Introduction to pytorch
epochs = 1
steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    for images, labels in dataloaders_train:
        steps += 1
        images, labels =  images.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        logps = vgg16(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if step % print_every == 0:
            vgg16.eval()
            test_loss = 0
            accuracy = 0
            
            for images, labels in dataloaders_valid:
                images, labels =  images.cuda(), labels.cuda()
                
                logps = model(images)
                loss = criterion(logps, labels)
                test_loss += loss.item()
                
                # calc acc
                ps = torch.exp(logps)
                top_ps, top_class = ps.topk(1, dim=1)  # along columns
                equality = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                
            print(f'Epoch {epoch+1}/{epochs}..'
                  f'Train loss: {running_loss/print_every:.3f}..'
                  f'Test loss: {test_loss/len(dataloaders_valid):.3f}..'
                  f'Test accuracy: {accuracy/len(dataloaders_valid):.3f}..')

            running_loss = 0
            model.train()
                

# Save model
# checkpoint = {
#     'input_size': 748,
#     'output_size': 10, 
#     'hidden_layers': [lyr.out_features for lyr in model.hidden_layers],
#     'state_dict': model.state_dict()
# }

# torch.save(checkpoint, 'model/checkpoint.pth')
