import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from skimage import io
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import time
import copy

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

plt.ion() # interactive mode

data_dir = '/media/jayden/JaydenHD/cholec80'

class cholec_dataset(Dataset):
    """Dataset containing cholecystectomy image and tool labels"""

    def __init__(self, directory,phase, transform=None):
        self.directory = directory
        self.phase = phase
        self.images = self.get_images(directory, phase)
        self.tool_annotations = self.get_tool_annotations(directory, phase)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = io.imread(img_name)
        tool_list = self.tool_annotations[idx]
        tools = list()
        for tool in tool_list:
            tools.append(int(tool))

        tools = torch.FloatTensor(tools)
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'tools': tools}
        return sample

    def get_images(self, directory, phase):
        if phase == 'Test':
            lower_bound = 40
            upper_bound = 79

        else:
            lower_bound = 1
            upper_bound = 40

        image_names = list()

        for i in range(lower_bound, upper_bound):
            video_path = '{}/video{}'.format(directory, str(i).zfill(2))
            if not os.path.isdir(video_path):
                continue
            image_num = 0
            while (1):
                image_path = ('{}/image{}.tif').format(video_path, str(image_num))
                if os.path.isfile(image_path):
                    image_names.append(image_path)
                    image_num += 25
                else:
                    break
            # pops last element from list
            # As the last frame isn't annotated in the dataset
            image_names.pop()
        return image_names

    def get_tool_annotations(self, directory, phase):
        if phase == 'Test':
            lower_bound = 40
            upper_bound = 79

        else:
            lower_bound = 1
            upper_bound = 40

        tool_annotations = list()
        for i in range(lower_bound, upper_bound):
            tool_path = '{}/video{}-tool.txt'.format(directory, str(i).zfill(2))
            if os.path.isfile(tool_path):
                with open(tool_path, 'r') as file:
                    next(file)
                    image_num = 0
                    for line in file:
                        line = line.strip('\n').split('\t')
                        line.remove(str(image_num))
                        image_num += 25
                        tool_annotations.append(line)

        return tool_annotations

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


print('Loading data...')

image_datasets = {x: cholec_dataset(os.path.join(data_dir, x), x,
                                    transform)
                  for x in ['Train', 'Validation', 'Test']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=64,
                                              shuffle=True, num_workers=8)
               for x in ['Train', 'Validation', 'Test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Validation', 'Test']}
class_names = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'SpecimenBag']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Data loaded!')

if torch.cuda.is_available():
    print('Computing on gpu')
else:
    print('Computing on cpu')

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            iterator = 1
            for batch in iter(dataloaders[phase]):
                inputs = batch['image']
                labels = batch['tools']

                if iterator%50 == 0:
                    print('Iteration {}'.format(iterator))
                iterator += 1

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs)

                    loss = criterion(preds, labels)

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # compute running corrects
                for i in range(len(preds)):
                    prediction = preds[i]
                    label = labels[i]
                    corrects = 0.0
                    for j in range(len(prediction)):
                        if prediction[j] > 0.5 and label[j] == 1:
                            corrects += 1

                        elif prediction[j] < 0.5 and label[j] == 0:
                            corrects += 1
                    running_corrects += (corrects/7)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'Validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_model(model):
    model.eval()

    running_corrects = 0.0

    with torch.no_grad():
        iterator = 1
        for batch in iter(dataloaders['Test']):
            inputs = batch['image']
            labels = batch['tools']

            if iterator % 50 == 0:
                print('Iteration {}'.format(iterator))
            iterator += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.sigmoid(outputs)

            # compute running corrects
            for i in range(len(preds)):
                prediction = preds[i]
                label = labels[i]
                corrects = 0.0
                for j in range(len(prediction)):
                    if prediction[j] > 0.5 and label[j] == 1:
                        corrects += 1

                    elif prediction[j] < 0.5 and label[j] == 0:
                        corrects += 1
                running_corrects += (corrects / 7)

    test_acc = running_corrects / dataset_sizes['Test']
    print('Test accuracy: {}'.format(test_acc))

model_conv = torchvision.models.resnet152(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 7)
model_conv = model_conv.to(device)

criterion = nn.BCELoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

print('Training model')
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=3)

# Save model
print('Saving model...')
torch.save(model_conv.state_dict(), '/media/jayden/JaydenHD/cholec80/resnet152.pt')
print('Saved!')

print('Testing model.')
test_model(model_conv)