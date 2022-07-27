import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
from GAN.gan_model import Discriminator, Generator, initialize_weights
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from scipy import stats

from play_space import feature

map_A = {'a': 'hallway_entrance', 'b': 'living_area_1',   'c': 'living_area_2',
          'd': 'living_area_3',     'e': 'living_area_4',       'f': 'bathroom',
          'g': 'bedroom_1',      'h': 'bedroom_2', 1: 'hallway_entrance',  2: 'bedroom',
            3: 'living_area',    4: 'bathroom'}

map_C = {'a': 'living_room_1', 'b': 'living_room_2',   'c': 'kitchen_1',
          'd': 'kitchen_2',     'e': 'kitchen_3',       'f': 'hallway_upper',
          'g': 'bathroom',      'h': 'bedroom-two',     'i': 'bedroom-one_1',
          'j': 'bedroom-one_2', 'k': 'study',
         1: 'living_room', 2: 'kitchen', 3: 'stairs', 4: 'outside',
         5: 'hallway', 6: 'bathroom', 7: 'bedroom-2', 8: 'bedroom-1', 9: 'study',
         'living': 1, 'kitchen': 2, 'stairs': 3, 'outside': 4,
         'hallway': 5, 'bathroom': 6, 'bedroom-2': 7, 'bedroom-1': 8, 'study': 9
         }

map_D = {'a': 'living_room_B1', 'b': 'living_room_B2',   'c': 'living_room_A1',
          'd': 'living_room_A2',     'e': 'hallway_lower',       'f': 'kitchen',
          'g': 'bathroom',      'h': 'hallway_upper',     'i': 'bedroom-two',
          'j': 'bedroom-one_1', 'k': 'bedroom-one_2',
         1: 'hallway_lower', 2: 'living_area_A', 3: 'living_area_B', 4: 'kitchen',
         5: 'outside', 6: 'stairs', 7: 'bathroom', 8: 'hallway_upper', 9: 'bedroom_2', 10:'bedroom_1',
         'hallway_lower': 1, 'living_area_A': 2, 'living_area_B': 3, 'kitchen': 4,
         'outside': 5, 'stairs': 6, 'bathroom': 7, 'hallway_upper': 8, 'bedroom_2': 9, 'bedroom_1':10}

map_B = {'a': 'living_room_B1*', 'b': 'living_room_B2*',   'c': 'living_room_A1*',
          'd': 'living_room_A2*',     'e': 'hallway_lower*',       'f': 'kitchen*',
          'g': 'bathroom',      'h': 'hallway_upper',     'i': 'bedroom-two',
          'j': 'bedroom-one_1', 'k': 'bedroom-one_2',
         1: 'hallway_lower', 2: 'living_area_A', 3: 'living_area_B', 4: 'kitchen',
         5: 'outside', 6: 'stairs', 7: 'bathroom', 8: 'hallway_upper', 9: 'bedroom_2', 10:'bedroom_1',
         'hallway_lower': 1, 'living_area_A': 2, 'living_area_B': 3, 'kitchen': 4,
         'outside': 5, 'stairs': 6, 'bathroom': 7, 'hallway_upper': 8, 'bedroom_2': 9, 'bedroom_1':10}

house_map ={'A':map_A, 'B':map_B, 'C':map_C, 'D':map_D}
num_room_house = {'A':4, 'B':11, 'C':9, 'D':10}


class MLPClassifier(nn.Module):
    def __init__(self,n_features, n_classes):
        super(MLPClassifier, self).__init__()

        self.model = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.ReLU(),

                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(64, n_classes),
                nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

def MinMaxScaler(data):
    """Min-Max Normalizer.
    Args:
      - data: raw data
    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val

    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)

    return norm_data, min_val, max_val

def renormalization(norm_data, min_val, max_val):
    return (norm_data * max_val) + min_val

def windowing(ori_data, y, seq_len = 20, hop_size = 10):
    windowed_data = []
    windowed_label = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len, hop_size):
        _x = ori_data[i:i + seq_len]
        _y = stats.mode(y[i:i + seq_len])[0][0]
        windowed_data.append(_x)
        windowed_label.append(_y)

    idx = np.random.permutation(len(windowed_data))
    data = []
    label = []
    for i in range(len(windowed_data)):
        data.append(windowed_data[idx[i]])
        label.append(windowed_label[idx[i]])

    data= np.asarray(data)
    label = np.asarray(label)
    return data, label

def plot_all_rssi(rssi_plot, label_plot='house_C', ap_map=None):
    rssi_plot = pd.DataFrame((rssi_plot), columns=['g', 'd', 'f', 'k', 'c', 'b', 'a', 'j', 'h', 'i', 'e'])
    if ap_map:
        rssi_plot = rssi_plot.rename(ap_map, axis='columns')
    #plot the data
    sns.set_theme(rc={'figure.figsize': (20, 3)})
    sns.lineplot(data=rssi_plot, legend=True)
    # sns.displot(rssi_plot, kind='kde', fill=fill, height=5, aspect=2.5)
    plt.title("%s" % (label_plot), fontsize=14)
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    # plt.legend(loc='lower left', bbox_to_anchor=(0, 1.01, 1.0, 0.5), markerfirst=True,
    #            mode="expand", borderaxespad=0, ncol=9, handletextpad=0.01, )
    plt.xlabel("Timepoint")
    # plt.xticks(fontsize=22)
    plt.ylabel("RSSI(dB)")
    # plt.yticks(fontsize=20)
    plt.tight_layout()

    plt.show()
    plt.close()

def plot_line_rssi_gan(rssi_plot, label_plot, transpose=False, house = None, house_map=None,
                       full_scale=False, ymin=None, ymax=None, save=False, save_dir=None,model_name=None):
    if transpose:
        rssi_plot =  np.transpose(rssi_plot)

    # house_map = {'A': map_A, 'B': map_A, 'C': map_A, 'D': map_A}
    house_aps_label = {'A': ['f', 'd', 'c', 'b', 'a', 'h', 'e', 'g'],
                       'B': ['g', 'h', 'd', 'j', 'i', 'b', 'c', 'e', 'k', 'a', 'f'],
                       'C': ['g', 'd', 'f', 'k', 'c', 'b', 'a', 'j', 'h', 'i', 'e'],
                       'D': ['g', 'e', 'j', 'k', 'b', 'a', 'i', 'f', 'd', 'h', 'c']}

    rssi_plot = pd.DataFrame((rssi_plot), columns=house_aps_label[house])

    if house_map:
        rssi_plot = rssi_plot.rename(house_map, axis='columns')
        label_plot = house_map[label_plot]

    #plot the data
    sns.set_theme(rc={'figure.figsize': (8, 4)})
    sns.lineplot(data=rssi_plot, legend=True)
    # sns.displot(rssi_plot, kind='kde', fill=fill, height=5, aspect=2.5)
    # plt.xlim(-0.1, 21)
    plt.ylim(ymin, ymax) #(-4, 3.5)
    if full_scale:
        plt.ylim(-130, 5)
    plt.title("label %s" % (label_plot), fontsize=14)
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.xlabel("Timepoint", fontsize=18)
    # plt.xticks(fontsize=22)
    plt.ylabel("RSSI(dB)", fontsize=18)
    # plt.yticks(fontsize=20)
    plt.tight_layout()
    if save:
        plt.savefig(save_dir + model_name + '_' + label_plot +'.png')
    else:
        plt.show()
    plt.close()

def load_GAN_model(gen_model, save_directory, model_name, num_epochs, device):
    # model = "ConGAN_wgp"
    pretrained_dir = save_directory + model_name + '_epoch' + str(num_epochs) + '.pt'
    state_dict = torch.load(pretrained_dir, map_location=device)
    gen_model.load_state_dict(state_dict, strict=True)
    return gen_model

def binary_convert(X, y, label):
    new_X = X[y == label]
    new_y = np.ones(len(new_X))

    other_X = X[y != label]
    other_y = np.zeros(len(other_X))

    X_all_new = np.concatenate((new_X, other_X), axis=0)
    y_all_new = np.concatenate((new_y, other_y), axis=0)

    return X_all_new, y_all_new

def get_mean_stat(windowed_data, windowed_label, number_APs, room_num, save=False, plot_name=None):
    '''
    :param windowed_data: (numpy) windowed_data of all rooms (sample_size, 11, 20)
    :param windowed_label: (numpy) label for each windowed_data (start at 1)
    :param number_APs: (int) total number of APs
    :param room_num: (int) total number of room
    :return: line plot of mean RSSI for each AP
    '''
    for room_label in range(room_num):
        cur_room_rssi = windowed_data[windowed_label == room_label + 1]
        cur_room_rssi = torch.from_numpy(cur_room_rssi)

        ap1 = cur_room_rssi[:, 0, :] # get first ap
        ap_plot = torch.full((1, 20), torch.mean(ap1)) # get mean of first ap
        for i in range(1, number_APs):
            ap = cur_room_rssi[:, i, :]
            ap_line = torch.full((1, 20), torch.mean(ap))
            ap_plot = torch.cat((ap_plot, ap_line), 0)
        if save:
            plot_line_rssi_gan(ap_plot.cpu().detach().numpy(), room_label + 1, transpose=True, label_map=label_map,
                               ap_map=ap_map, ymin=-0.1, ymax=1.1, save=True, model_name=str(plot_name))
        else:
            plot_line_rssi_gan(ap_plot.cpu().detach().numpy(), room_label + 1, transpose=True, label_map=label_map,
                               ap_map=ap_map, ymin=-0.1, ymax=1.1, save=False)

def generate_fake_data(gen_model, Z_DIM, num_fake, total_room, aps, device):
    labels_house = torch.from_numpy(np.full((num_fake[0]), 0)).to(device)
    gen_model = gen_model.to(device)
    noise = torch.randn(num_fake[0], Z_DIM, aps, 20).uniform_(0, 1).to(device)
    gen_model.eval()
    fake_house = gen_model(noise, labels_house)

    for room_number in range(1, total_room):
        labels_room = torch.from_numpy(np.full((num_fake[room_number]), room_number)).to(device)
        noise = torch.randn(num_fake[room_number], Z_DIM, aps, 20).uniform_(0, 1).to(device)
        gen_model.eval()
        fake_room = gen_model(noise, labels_room)

        fake_house = torch.cat((fake_house, fake_room), 0).to(device)
        labels_house = torch.cat((labels_house, labels_room), 0).to(device)
    return fake_house, labels_house

def train(loader, model, optimizer, criterion):
    model.train()
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        output = model(x)
        optimizer.zero_grad()
        loss = criterion(output, y.long())

        loss.backward(retain_graph=False)  # calculate Gradients
        optimizer.step()  # update Weights
        loss_epoch += loss.item()

        # calculate accuracy
        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        # create list of true labels and prediction labels for the whole batch
        y = y.cpu().detach().numpy()
        predicted = predicted.cpu().detach().numpy()
        # outputs_list.append(predicted)
        # targets_list.append(y)
        if step == 0:
            outputs_list = predicted
            targets_list = y
        else:
            outputs_list = np.concatenate((outputs_list, predicted), axis=None)
            targets_list = np.concatenate((targets_list, y), axis=None)

        # calculate F1 score
    f1_epoch = f1_score(targets_list, outputs_list, average='weighted')

    return loss_epoch, accuracy_epoch, f1_epoch

def test(loader, model, criterion):
    loss_epoch = 0
    accuracy_epoch = 0
    # outputs_list = []
    # targets_list = []
    model.eval()
    for step, (x, y) in enumerate(loader):
        output = model(x)
        loss = criterion(output, y.long())
        loss_epoch += loss.item()

        # calculate accuracy
        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        # create list of true labels and prediction labels for the whole batch
        y = y.cpu().detach().numpy()
        predicted = predicted.cpu().detach().numpy()
        if step == 0:
            outputs_list = predicted
            targets_list = y
        else:
            outputs_list = np.concatenate((outputs_list, predicted), axis=None)
            targets_list = np.concatenate((targets_list, y), axis=None)

    # calculate F1 score
    f1_epoch = f1_score(targets_list, outputs_list, average='weighted')

    # print('TESTING loss:', loss_epoch/len(loader),
    #       'accuracy:', accuracy_epoch*100/len(loader),
    #       'f1:', f1_epoch*100)

    return loss_epoch/len(loader), accuracy_epoch*100/len(loader), f1_epoch*100

def model_test(X, y, model):
    # model prediction
    output = model(X)
    predicted = output.argmax(1)
    # predicted = predicted.cpu().detach().numpy()
    # calculate accuracy
    acc = (predicted == y).sum().item() / y.size(0)
    print('Accuracy:', acc * 100)
    # calculate F1
    predicted = predicted.cpu().detach().numpy()
    true = y.cpu().detach().numpy()
    f1 = f1_score(true, predicted, average='weighted')
    print('F1 score:', f1 * 100)

    return predicted, true

