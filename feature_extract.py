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

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

plt.ion() # Interactive mode

data_dir = '/media/jayden/JaydenHD/cholec80'

class cholec_dataset(Dataset):

    def __init__(self, directory, phase, transform=None):
        self.directory = directory
        self.phase = phase
        self.images, self.video_numbers = self.get_images(directory, phase)
        print(len(self.images))
        print(len(self.video_numbers))
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = io.imread(img_name)

        number = self.video_numbers[idx]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'number': number}
        return sample

    def __len__(self):
        return len(self.images)

    def get_images(self, directory, phase):
        if phase == 'Test':
            lower_bound = 40
            upper_bound = 79

        else:
            lower_bound = 37
            upper_bound = 40

        image_names = list()
        video_numbers = list()

        for i in range(lower_bound, upper_bound):
            video_path = '{}/video{}'.format(directory, str(i).zfill(2))
            if not os.path.isdir(video_path):
                continue
            image_num = 0
            while (1):
                image_path = ('{}/image{}.tif').format(video_path, str(image_num))
                if os.path.isfile(image_path):
                    image_names.append(image_path)
                    video_numbers.append(str(i).zfill(2))
                    image_num += 25
                else:
                    break
        return image_names, video_numbers

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

dataloaders = {x: DataLoader(image_datasets[x], batch_size=1,
                                              shuffle=False, num_workers=0)
               for x in ['Train', 'Validation', 'Test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Validation', 'Test']}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Data loaded!')

if torch.cuda.is_available():
    print('Computing on gpu')
else:
    print('Computing on cpu')

def extract_features(model):

    with torch.no_grad():
        for phase in ['Train', 'Validation']:
            video_number = '0'
            output_string = ''
            for batch in iter(dataloaders[phase]):
                inputs = batch['image']
                number = batch['number']

                inputs = inputs.to(device)

                if number != video_number:
                    if video_number != '0':
                        file = open('{}/{}/video_features{}.txt'.format(data_dir, phase, video_number[0]), 'w')
                        file.write(output_string)
                        file.close()
                        output_string = output_string.split(' ')
                        print(len(output_string))
                        print('Video{} ended'.format(video_number[0]))
                    output_string = ''
                    video_number = number
                    print('Video{} started'.format(video_number[0]))

                output = model(inputs)
                for i in range(2048):
                    output_value = output[0][i][0][0]
                    output_string += '%.4f' % (output_value.item())
                    if i == 2047:
                        output_string += '\n'
                    else:
                        output_string += ' '
            file = open('{}/{}/video_features{}.txt'.format(data_dir, phase, video_number[0]), 'w')
            file.write(output_string)
            file.close()
            output_string = output_string.split(' ')
            print(len(output_string))
            print('Video{} ended'.format(video_number[0]))

model_conv = torchvision.models.resnet152()
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 7)
model_conv = model_conv.to(device)
model_conv.load_state_dict(torch.load('/media/jayden/JaydenHD/cholec80/resnet152.pt'))
model_conv.eval()
model_conv = torch.nn.Sequential(*(list(model_conv.children())[:-1]))

extract_features(model_conv)