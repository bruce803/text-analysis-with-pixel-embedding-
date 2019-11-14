import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from model import NetAgMaxOverTime
from sklearn import metrics
import numpy as np
from torch.optim import lr_scheduler
import copy
import time
import os
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier

try:
    import cPickle as pickle
except ImportError:
    import pickle


if sys.version_info[0] > 2:
    is_python3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_python3 = False


class AgDataset(Dataset):

    def __init__(self, sample_list, root='../ipython/data/agpkl/train'):
        '''

        :param root_text: the dir store text data
        :param label_file: label file path
        '''
        self.root = root
        self.sample_files = sample_list


    def __getitem__(self, index):
        pkl_path = os.path.join(self.root, self.sample_files[index])
        with open(pkl_path, 'rb') as pkl:
            Xy = pickle.load(pkl) # ag data didn't sparse
            X = Xy[1] # is still sparse in COO
            y = Xy[0]
            X = self.sequence_pad(X.todense()) ## padding or truncating
        return X, y, self.sample_files[index]

    def __len__(self):
        return len(self.sample_files)

    def sequence_pad(self, ndarray, max_len_seq=80, binary=True):
        '''

        :param ndarray: 3-dim char array, 1st, 2nd dimension corresponds to the size of char-img, 3rd dimension is the sentence length
        :param max_len_seq: threshold of length of a sentence
        :param binary: True, convert gray matrix to binary; False, gray matrix
        :return: truncated or padded 3d array
        '''

        def to_binary(ndarray, threshold=25):
            '''
            make all pixels > threshold as 1, otherwise 0
            '''
            return 1 * (ndarray > threshold)

        seq_len = ndarray.shape[2]

        if seq_len >= max_len_seq:  #truncating
            ndarray = ndarray[...,0:max_len_seq]

        # padding
        else:
            lag_len = max_len_seq - seq_len
            ndarray = np.concatenate((ndarray, np.zeros((ndarray.shape[0], ndarray.shape[1], lag_len))), axis=2)

        if binary:
            return to_binary(ndarray)
        else:
            return ndarray



print('begin to load data...')

train_dir = '..data/agpkl/train1'
test_dir = '../data/agpkl/test1'


train_val_list = os.listdir(train_dir)
train_list, val_list = train_test_split(train_val_list, test_size=0.2)

test_list = os.listdir(test_dir)


trainset=AgDataset(train_list, train_dir)  # single train set
trainloader=DataLoader(trainset, batch_size=100, shuffle=True, num_workers=1)

valset=AgDataset(val_list, train_dir)  # single train set
valloader=DataLoader(trainset, batch_size=100, shuffle=True, num_workers=1)

testset=AgDataset(test_list, test_dir)  # single train set
testloader=DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)



dataloderDict = {'train': trainloader,
                 'val': valloader}

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, _ in dataloderDict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                inputs = inputs.unsqueeze(1) # the second dim corresponding to channel
                # print('inputs shape after unsqueeze:', inputs.shape)
                inputs = inputs.float()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, x_after_conv1 = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # print(preds)
                    loss = criterion(outputs, labels)
                    # print(labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            # dataset_sizes[phase]:len(dataloderDict[phase].dataset)
            epoch_loss = running_loss / len(trainloader.dataset)
            epoch_acc = running_corrects.double() / len(trainloader.dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
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




# optimization
net = NetAgMaxOverTime(0.5)
net.to(device)

# optimization
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.00015, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
# plateau_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9)

net_ft = train_model(net, criterion, optimizer, exp_lr_scheduler, num_epochs=10)

torch.save(net_ft.state_dict(), 'net-AG-2gram.model')

# testing
print('Testing...')
correct = 0
total = 0

num_test_samples = len(testloader.dataset)
y_predicted = np.zeros(num_test_samples, dtype=np.int32)
real_labels = np.zeros(num_test_samples, dtype=np.int32)
fea_map_conv1 = np.zeros((num_test_samples, 50, 79))

class Dictionary(dict):

    # __init__ function
    def __init__(self):
        self = dict()

        # Function to add key:value

    def add(self, key, value):
        self[key] = value

sample2feaMap_dict = Dictionary()
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        start_id = i * testloader.batch_size
        end_id = min((i + 1) * testloader.batch_size, num_test_samples)
        matrice, labels, sample_file_name = data
        # print('X input is:', matrice)
        # print('y input is:', labels)
        matrice = matrice.to(device)
        labels = labels.to(device)
        matrice = matrice.unsqueeze(1) # the second dim corresponding to channel
        matrice = matrice.float()
        outputs, fea_map = net_ft(matrice)
        _, predicted = torch.max(outputs.data, 1)
        # print('y predicted is:', predicted)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        y_predicted[start_id:end_id] = predicted.cpu().numpy()
        real_labels[start_id:end_id] = labels.cpu().numpy()
        sample2feaMap_dict.add(sample_file_name, fea_map.cpu().numpy()) # add to {sample_file_name: conv_fea_map}
        fea_map_conv1[start_id:end_id, ...] = fea_map.cpu().numpy()



print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))

# if (correct / total) >= 0.8:
#     torch.save(net_ft.state_dict(), 'net-AG.model')

print("Precision, Recall and F1-Score...")
print(metrics.classification_report(real_labels, y_predicted))

print("Confusion Matrix...")
cm = metrics.confusion_matrix(real_labels, y_predicted)
print(cm)


with open('featureMapConv1-2gram.pkl', 'wb') as file:
    pickle.dump([sample2feaMap_dict, fea_map_conv1], file)





