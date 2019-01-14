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

plt.ion() # Interactive mode

data_dir = '/media/jayden/JaydenHD/cholec80'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class cholec_dataset(Dataset):

    def __init__(self, directory, phase, transform=None):
        self.directory = directory
        self.phase_to_idx = {'TrocarPlacement': 0, 'Preparation': 1, 'CalotTriangleDissection': 2, 'ClippingCutting': 3,
                             'GallbladderDissection': 4, 'GallbladderPackaging': 5, 'CleaningCoagulation': 6,
                             'GallbladderRetraction': 7}
        self.sequence_length = 200
        self.phase = phase
        self.images = self.get_images(directory, phase)
        self.phase_annotations = self.get_phase_annotations(directory, phase)
        if len(self.images) != len(self.phase_annotations):
            print("There's a problem")
            print(len(self.images))
            print(len(self.phase_annotations))
        self.transform = transform

    def __getitem__(self, idx):

        img_sequence = self.images[idx]

        images = torch.randn(len(img_sequence), 1, 2048)

        for i in range(len(img_sequence)):
            line = img_sequence[i]
            line = line.split(' ')
            for j in range(len(line)):
                images[i][0][j] = float(line[j])

        images = torch.FloatTensor(images)
        phases = torch.FloatTensor(self.phase_annotations[idx])

        sample = {'images': images, 'phases': phases}
        return sample

    def __len__(self):
        return len(self.images)

    def get_images(self, directory, phase):
        if phase == 'Test':
            lower_bound = 40
            upper_bound = 79

        else:
            lower_bound = 1
            upper_bound = 40

        image_features = list()

        for i in range(lower_bound, upper_bound):
            video_path = '{}/video_features{}.txt'.format(directory, str(i).zfill(2))
            if os.path.isfile(video_path):
                with open(video_path, 'r') as file:
                    for line in file:
                        image_features.append(line.strip('\n'))

        image_sequences = list()

        length = int(len(image_features)/self.sequence_length)

        for i in range(length):
            sequence = list()
            start = self.sequence_length*i
            for j in range(start, start+self.sequence_length):
                sequence.append(image_features[j])
            image_sequences.append(sequence)
        return image_sequences

    def get_phase_annotations(self, directory, phase):
        if phase == 'Test':
            lower_bound = 40
            upper_bound = 79

        else:
            lower_bound = 1
            upper_bound = 40

        phase_annotations = list()
        for i in range(lower_bound, upper_bound):
            phase_path = '{}/video{}-phase.txt'.format(directory, str(i).zfill(2))
            if os.path.isfile(phase_path):
                with open(phase_path, 'r') as file:
                    next(file)
                    image_num = 0
                    for line in file:
                        if image_num == 0 or image_num % 25 == 0:
                            line = line.strip('\n').split('\t')
                            line.remove(str(image_num))
                            line = self.phase_to_idx[line[0]]
                            phase_annotations.append(line)
                        image_num += 1

        phase_sequences = list()

        length = int(len(phase_annotations)/self.sequence_length)

        for i in range(length):
            sequence = list()
            start = self.sequence_length*i
            for j in range(start, start+self.sequence_length):
                sequence.append(phase_annotations[j])
            phase_sequences.append(sequence)
        return phase_sequences

class PhaseTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, softmax):
        super(PhaseTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.softmax = softmax
        self.output_dim = output_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, 1)

        self.hidden2out = nn.Linear(hidden_dim, output_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(1, 1, self.hidden_dim).type(torch.FloatTensor).to(device),
                torch.randn(1, 1, self.hidden_dim).type(torch.FloatTensor).to(device))

    def forward(self, input):
        hidden_output, self.hidden = self.lstm(input, self.hidden)
        output = self.hidden2out(hidden_output)
        output = softmax(output)
        output_resized = torch.randn(len(output), self.output_dim)
        for i in range(len(output)):
            output_resized[i] = output[i][0]
        output_resized = torch.FloatTensor(output_resized)
        return output_resized

print('Loading data...')

image_datasets = {x: cholec_dataset(os.path.join(data_dir, x), x)
                  for x in ['Train', 'Validation', 'Test']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=1,
                                              shuffle=False, num_workers=0)
               for x in ['Train', 'Validation', 'Test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Validation', 'Test']}

print('Data loaded!')

if torch.cuda.is_available():
    print('Computing on gpu')
else:
    print('Computing on cpu')


def feature_to_input(feature):
    lstm_input = torch.randn(1, 1, 2048)

    for i in range(2048):
        lstm_input[0][0][i] = feature[0][i][0][0]

    return lstm_input


def train_model(lstm_model, criterion, optimiser, num_epochs=4):
    since = time.time()

    best_model_wts = copy.deepcopy(lstm_model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                lstm_model.train()
            else:
                lstm_model.eval()

            running_loss = 0.0
            running_corrects = 0.0

#            iterator = 1

            for batch in iter(dataloaders[phase]):

                inputs = batch['images']
                phases = batch['phases']

                # if iterator % 1 == 0:
                #     print('Iteration {}'.format(iterator))
                # iterator += 1

                inputs = inputs.to(device)
                phases = phases.to(device)

                # zero the parameter gradients
                optimiser.zero_grad()

                lstm_model.hidden = lstm_model.init_hidden()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    output = lstm_model(inputs[0])

                    output = output.to(device)
                    loss = criterion(output, (phases.long())[0])

                    # backward + optimise only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimiser.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # compute prediction
                sequence_corrects = 0.0
                for i in range(len(output)):
                    predictions = output[i]
                    correct_phase = phases[0][i]

                    largest_prediction = 0
                    largest_prediction_index = 0
                    for j in range(len(predictions)):
                        if predictions[j] > largest_prediction:
                            largest_prediction = predictions[j]
                            largest_prediction_index = j
                    if largest_prediction_index == correct_phase:
                        sequence_corrects += 1

                sequence_corrects = sequence_corrects / len(output)
                running_corrects += sequence_corrects
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'Validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(lstm_model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    lstm_model.load_state_dict(best_model_wts)
    return lstm_model


softmax = nn.Softmax(2)
model_lstm = PhaseTagger(2048, 128, 8, softmax)
model_lstm = model_lstm.to(device)
optimiser = optim.Adam(model_lstm.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion = nn.CrossEntropyLoss()

train_model(model_lstm, criterion, optimiser)



