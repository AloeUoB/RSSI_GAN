import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
from GAN.con_gan_rssi.model_wgan_gp import Discriminator, Generator, initialize_weights
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from scipy import stats

from play_space import feature

map_A = {'a': 'hallway_entrance', 'b': 'living_area_1',   'c': 'living_area_2',
          'd': 'living_area_3',     'e': 'living_area_4',       'f': 'bathroom',
          'g': 'bedroom_1',      'h': 'bedroom_2', 1: 'hallway_entrance',  2: 'bedroom',
            3: 'living_area',    4: 'bathroom', 'hallway_entrance':1, 'bedroom':2,
            'living_area':3,  'bathroom':4,'pad':'padding_ap'}

map_B = {'a': 'hallway_lower', 'b': 'living_room',   'c': 'dining_room_1',
          'd': 'dining_room_2',     'e': 'kitchen_1',       'f': 'kitchen_2',
          'g': 'bathroom',      'h': 'hallway_upper',     'i': 'bedroom-two',
          'j': 'bedroom-one', 'k': 'toilet',
         1: 'hallway_lower', 2: 'kitchen', 3: 'living_room', 4: 'dining_room', 5: 'stairs_lower',
        6: 'stairs_upper', 7: 'bathroom', 8: 'hallway_upper', 9: 'bedroom_2', 10:'bedroom_1', 11:'toilet',
         'hallway_lower': 1, 'kitchen': 2, 'living_room': 3, 'dining_room': 4, 'stairs_lower': 5,
        'stairs_upper': 6, 'bathroom': 7, 'hallway_upper': 8, 'bedroom_2': 9, 'bedroom_1':10,'pad':'padding_ap'}

map_C = {'a': 'living_room_1', 'b': 'living_room_2',   'c': 'kitchen_1',
          'd': 'kitchen_2',     'e': 'kitchen_3',       'f': 'hallway_upper',
          'g': 'bathroom',      'h': 'bedroom-two',     'i': 'bedroom-one_1',
          'j': 'bedroom-one_2', 'k': 'study',
         1: 'living_room', 2: 'kitchen', 3: 'stairs', 4: 'outside',
         5: 'hallway', 6: 'bathroom', 7: 'bedroom-2', 8: 'bedroom-1', 9: 'study',
         'living': 1, 'kitchen': 2, 'stairs': 3, 'outside': 4,
         'hallway': 5, 'bathroom': 6, 'bedroom-2': 7, 'bedroom-1': 8, 'study': 9,
         'pad':'padding_ap'
         }

map_D = {'a': 'living_room_B1', 'b': 'living_room_B2',   'c': 'living_room_A1',
          'd': 'living_room_A2',     'e': 'hallway_lower',       'f': 'kitchen',
          'g': 'bathroom',      'h': 'hallway_upper',     'i': 'bedroom-two',
          'j': 'bedroom-one_1', 'k': 'bedroom-one_2',
         1: 'hallway_lower', 2: 'living_area_A', 3: 'living_area_B', 4: 'kitchen',
         5: 'outside', 6: 'stairs', 7: 'bathroom', 8: 'hallway_upper', 9: 'bedroom_2', 10:'bedroom_1',
         'hallway_lower': 1, 'living_area_A': 2, 'living_area_B': 3, 'kitchen': 4,
         'outside': 5, 'stairs': 6, 'bathroom': 7, 'hallway_upper': 8, 'bedroom_2': 9, 'bedroom_1':10,'pad':'padding_ap'}

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

class MLPClassifier_big(nn.Module):
    def __init__(self,n_features, n_classes):
        super(MLPClassifier, self).__init__()

        self.model = nn.Sequential(
                nn.Linear(n_features, 1024),
                nn.ReLU(),

                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),

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

def windowing(ori_data, y, seq_len = 20, hop_size = 10, shuffle=True):
    windowed_data = []
    windowed_label = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len, hop_size):
        _x = ori_data[i:i + seq_len]
        _y = stats.mode(y[i:i + seq_len])[0][0]
        windowed_data.append(_x)
        windowed_label.append(_y)
    if shuffle:
        idx = np.random.permutation(len(windowed_data))
        data = []
        label = []
        for i in range(len(windowed_data)):
            data.append(windowed_data[idx[i]])
            label.append(windowed_label[idx[i]])
    else:
        data = windowed_data
        label = windowed_label
    data = np.asarray(data)
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

def get_col_use(house_name, reduce_ap):
    if house_name == 'A':
        col_idx_use = [1, 2, 3, 4, 5, 6, 7, 8]
    else:
        col_idx_use = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    col_idx_use_label = col_idx_use[len(col_idx_use) - 1] + 1

    if reduce_ap:
        if house_name == 'C':
            col_idx_use = [1, 2, 4, 7, 9, 10]
    return col_idx_use, col_idx_use_label

def load_house_data(data_directory, house_name, datatype='fp', reduce_ap=False):
    col_idx_use, col_idx_use_label = get_col_use(house_name, reduce_ap)
    # load raw data
    house_file = 'csv_house_' + house_name + '_'+datatype+'.csv'
    ori_data = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=col_idx_use)
    label = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=[col_idx_use_label])
    # data normalisation
    norm_data, min_val, max_val = MinMaxScaler(ori_data)
    # get window data
    windowed_data, windowed_label = windowing(norm_data, label, seq_len=20, hop_size=10)
    # get data information
    NUM_CLASSES = num_room_house[house_name]
    APs = len(col_idx_use)

    return windowed_data, windowed_label, APs, NUM_CLASSES

def plot_line_rssi_gan(rssi_plot, label_plot, transpose=False, house = None, house_map=None, reduce=False,
                       full_scale=False, ymin=None, ymax=None, save=False, save_dir=None, model_name=None, plot_idx=None, pad=False):
    if transpose:
        rssi_plot =  np.transpose(rssi_plot)

    # house_map = {'A': map_A, 'B': map_A, 'C': map_A, 'D': map_A}
    if reduce:
        house_aps_label = {'A': ['f', 'd', 'c', 'b', 'a', 'h', 'e', 'g'],
                           'B': ['g', 'h', 'd', 'j', 'i', 'b', 'c', 'e', 'k', 'a', 'f'],
                           'C': ['g', 'd', 'k', 'a', 'h', 'i'],
                           'D': ['g', 'e', 'j', 'k', 'b', 'a', 'i', 'f', 'd', 'h', 'c']}
    else:
        if pad:
            house_aps_label = {'A': ['f', 'd', 'c', 'b', 'a', 'h', 'e', 'g'],
                               'B': ['g', 'h', 'd', 'j', 'i', 'b', 'c', 'e', 'k', 'a', 'f','pad'],
                               'C': ['g', 'd', 'f', 'k', 'c', 'b', 'a', 'j', 'h', 'i', 'e','pad'],
                               'D': ['g', 'e', 'j', 'k', 'b', 'a', 'i', 'f', 'd', 'h', 'c','pad']}
        else:
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
        plt.savefig(save_dir + str(plot_idx) + 'e_' + model_name + '_' +label_plot + '.png')
    else:
        plt.show()
    plt.close()

def plot_line_rssi_house(rssi_plot, label_plot, transpose=False, house = None, house_map=None, reduce=False,
                       full_scale=False, ymin=None, ymax=None, save=False, save_dir=None, model_name=None, plot_idx=None, pad=False):
    if transpose:
        rssi_plot =  np.transpose(rssi_plot)

    # house_map = {'A': map_A, 'B': map_A, 'C': map_A, 'D': map_A}
    if reduce:
        house_aps_label = {'A': ['f', 'd', 'c', 'b', 'a', 'h', 'e', 'g'],
                           'B': ['g', 'h', 'd', 'j', 'i', 'b', 'c', 'e', 'k', 'a', 'f'],
                           'C': ['g', 'd', 'k', 'a', 'h', 'i'],
                           'D': ['g', 'e', 'j', 'k', 'b', 'a', 'i', 'f', 'd', 'h', 'c']}
    else:
        if pad:
            house_aps_label = {'A': ['f', 'd', 'c', 'b', 'a', 'h', 'e', 'g'],
                               'B': ['g', 'h', 'd', 'j', 'i', 'b', 'c', 'e', 'k', 'a', 'f','pad'],
                               'C': ['g', 'd', 'f', 'k', 'c', 'b', 'a', 'j', 'h', 'i', 'e','pad'],
                               'D': ['g', 'e', 'j', 'k', 'b', 'a', 'i', 'f', 'd', 'h', 'c','pad']}
        else:
            house_aps_label = {'A': ['f', 'd', 'c', 'b', 'a', 'h', 'e', 'g'],
                               'B': ['g', 'h', 'd', 'j', 'i', 'b', 'c', 'e', 'k', 'a', 'f'],
                               'C': ['g', 'd', 'f', 'k', 'c', 'b', 'a', 'j', 'h', 'i', 'e'],
                               'D': ['g', 'e', 'j', 'k', 'b', 'a', 'i', 'f', 'd', 'h', 'c']}

    rssi_plot = pd.DataFrame((rssi_plot), columns=house_aps_label[house])

    if house_map:
        rssi_plot = rssi_plot.rename(house_map, axis='columns')
        label_plot = house

    #plot the data
    sns.set_theme(rc={'figure.figsize': (8, 4)})
    sns.lineplot(data=rssi_plot, legend=True)
    # sns.displot(rssi_plot, kind='kde', fill=fill, height=5, aspect=2.5)
    # plt.xlim(-0.1, 21)
    plt.ylim(ymin, ymax) #(-4, 3.5)
    if full_scale:
        plt.ylim(-130, 5)
    plt.title("House %s" % (house), fontsize=14)
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.xlabel("Timepoint", fontsize=18)
    # plt.xticks(fontsize=22)
    plt.ylabel("RSSI(dB)", fontsize=18)
    # plt.yticks(fontsize=20)
    plt.tight_layout()
    if save:
        plt.savefig(save_dir + str(plot_idx) +'e_'+ model_name + label_plot +'.png')
    else:
        plt.show()
    plt.close()

def plot_line_rssi_all(rssi_plot, label_plot, transpose=False, house = None, house_map=None, reduce=False,
                       full_scale=False, ymin=None, ymax=None, save=False, save_dir=None, model_name=None):
    if transpose:
        rssi_plot = np.transpose(rssi_plot)

    # house_map = {'A': map_A, 'B': map_A, 'C': map_A, 'D': map_A}
    if reduce:
        house_aps_label = {'A': ['f', 'd', 'c', 'b', 'a', 'h', 'e', 'g'],
                           'B': ['g', 'h', 'd', 'j', 'i', 'b', 'c', 'e', 'k', 'a', 'f'],
                           'C': ['g', 'd', 'k', 'a', 'h', 'i'],
                           'D': ['g', 'e', 'j', 'k', 'b', 'a', 'i', 'f', 'd', 'h', 'c']}
    else:
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
        plt.savefig('GAN/lineplot_all/House_'+house+'/'+label_plot+'/' + label_plot+model_name +'.png')
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

def get_mean_stat(windowed_data, windowed_label, house, house_map, number_APs, room_num, transpose=False, save=False, plot_name=None):
    '''
    :param windowed_data: (numpy) windowed_data of all rooms (sample_size, 11, 20)
    :param windowed_label: (numpy) label for each windowed_data (start at 1)
    :param number_APs: (int) total number of APs
    :param room_num: (int) total number of room
    :return: line plot of mean RSSI for each AP
    '''
    if transpose:
        windowed_data = np.transpose(windowed_data, (0, 2, 1))
    keep_mean = []
    for room_label in range(room_num):
        cur_room_rssi = windowed_data[windowed_label == room_label]
        cur_room_rssi = torch.from_numpy(cur_room_rssi)

        ap1 = cur_room_rssi[:, 0, :] # get first ap
        keep_mean.append(torch.mean(ap1).detach().numpy())
        ap_plot = torch.full((1, 20), torch.mean(ap1)) # get mean of first ap
        for i in range(1, number_APs):
            ap = cur_room_rssi[:, i, :]
            ap_line = torch.full((1, 20), torch.mean(ap))
            keep_mean.append(torch.mean(ap).detach().numpy())
            ap_plot = torch.cat((ap_plot, ap_line), 0)
        # if save:
        #     plot_line_rssi_gan(ap_plot.cpu().detach().numpy(), room_label + 1, transpose=True, house=house, house_map=house_map,
        #                        ymin=-0.1, ymax=1.1, save=True, model_name=str(plot_name))
        # else:
        #     plot_line_rssi_gan(ap_plot.cpu().detach().numpy(), room_label + 1, transpose=True, house=house, house_map=house_map,
        #                        ymin=-0.1, ymax=1.1, save=False)
    return np.array(keep_mean).reshape(room_num, number_APs)

def get_std_stat(windowed_data, windowed_label, house, house_map, number_APs, room_num, transpose=False, save=False, plot_name=None):
    '''
    :param windowed_data: (numpy) windowed_data of all rooms (sample_size, 11, 20)
    :param windowed_label: (numpy) label for each windowed_data (start at 1)
    :param number_APs: (int) total number of APs
    :param room_num: (int) total number of room
    :return: line plot of mean RSSI for each AP
    '''
    if transpose:
        windowed_data = np.transpose(windowed_data, (0, 2, 1))
    keep_std = []
    for room_label in range(room_num):
        cur_room_rssi = windowed_data[windowed_label == room_label]
        cur_room_rssi = torch.from_numpy(cur_room_rssi)

        ap1 = cur_room_rssi[:, 0, :] # get first ap
        keep_std.append(torch.std(ap1).detach().numpy())
        ap_plot = torch.full((1, 20), torch.std(ap1)) # get mean of first ap
        for i in range(1, number_APs):
            ap = cur_room_rssi[:, i, :]
            ap_line = torch.full((1, 20), torch.std(ap))
            keep_std.append(torch.std(ap).detach().numpy())
            ap_plot = torch.cat((ap_plot, ap_line), 0)
        # if save:
        #     plot_line_rssi_gan(ap_plot.cpu().detach().numpy(), room_label + 1, transpose=True, house=house, house_map=house_map,
        #                        ymin=-0.1, ymax=1.1, save=True, model_name=str(plot_name))
        # else:
        #     plot_line_rssi_gan(ap_plot.cpu().detach().numpy(), room_label + 1, transpose=True, house=house, house_map=house_map,
        #                        ymin=-0.1, ymax=1.1, save=False)
    return np.array(keep_std).reshape(room_num, number_APs)

def get_stats(ori_data, label, house, house_map, norm = True, APs=11, rooms=9 ): # house_file = 'csv_house_' + house_name + '_fl.csv'
    if norm:
        norm_data, min_val, max_val = MinMaxScaler(ori_data)
    else:
        norm_data = ori_data
    windowed_data, windowed_label = windowing(norm_data, label, seq_len=20, hop_size=10)
    windowed_data = np.transpose(windowed_data, (0, 2, 1))
    get_mean_stat(windowed_data, windowed_label, house, house_map, number_APs=APs, room_num=rooms, save=False, plot_name=None)


def generate_fake_data(gen_model, Z_DIM, num_fake, total_room, aps, device):
    labels_house = torch.from_numpy(np.full((num_fake[0]), 0)).to(device)
    gen_model = gen_model.to(device)
    noise = torch.randn(num_fake[0], Z_DIM, aps, 20).uniform_(0, 1).to(device)
    gen_model.eval()
    fake_house = gen_model(noise, labels_house)
    fake_house = fake_house.cpu().detach()

    for room_number in range(1, total_room):
        labels_room = torch.from_numpy(np.full((num_fake[room_number]), room_number)).to(device)
        noise = torch.randn(num_fake[room_number], Z_DIM, aps, 20).uniform_(0, 1).to(device)
        gen_model.eval()
        fake_room = gen_model(noise, labels_room)
        fake_room = fake_room.cpu().detach()

        fake_house = torch.cat((fake_house, fake_room), 0)
        labels_house = torch.cat((labels_house, labels_room), 0)
        del noise, fake_room
        torch.cuda.empty_cache()
    return fake_house, labels_house

def train(loader, model, optimizer, criterion): #'weighted','macro' , f1_type='weighted'
    # from tqdm import tqdm
    # steps = list(enumerate(loader))
    # pbar = tqdm(steps)
    model.train()
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        output = model(x)
        loss = criterion(output, y.long())
        optimizer.zero_grad()
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
    f1_epoch_w = f1_score(targets_list, outputs_list, average= 'weighted')
    f1_epoch_m = f1_score(targets_list, outputs_list, average= 'macro')

    return loss_epoch, accuracy_epoch, f1_epoch_w, f1_epoch_m

def train_val(train_loader, val_loader, model, optimizer, criterion): #'weighted','macro' , f1_type='weighted'
    # from tqdm import tqdm
    # steps = list(enumerate(loader))
    # pbar = tqdm(steps)
    model.train()
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(train_loader):
        output = model(x)
        loss = criterion(output, y.long())
        optimizer.zero_grad()
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
    f1_epoch_w = f1_score(targets_list, outputs_list, average= 'weighted')
    f1_epoch_m = f1_score(targets_list, outputs_list, average= 'macro')

    return loss_epoch, accuracy_epoch, f1_epoch_w, f1_epoch_m

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
    f1_epoch_weight = f1_score(targets_list, outputs_list, average='weighted')
    f1_epoch_mac = f1_score(targets_list, outputs_list, average='macro')
    mcc = matthews_corrcoef(targets_list, outputs_list)
    # print('TESTING loss:', loss_epoch/len(loader),
    #       'accuracy:', accuracy_epoch*100/len(loader),
    #       'f1:', f1_epoch*100)

    return loss_epoch/len(loader), accuracy_epoch*100/len(loader), f1_epoch_weight*100, f1_epoch_mac*100, mcc

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

def get_fake_rssi(real_data_labels, num_epochs, total_number, low_bound, NUM_CLASSES, APs, save_directory, model_name, device):
    # generator parameter
    Z_DIM = 11
    CHANNELS_RSSI = 1
    FEATURES_GEN = 32
    WINDOW_SIZE = 20
    GEN_EMBEDDING = 100
    # load pretrained generator
    gen = Generator(Z_DIM, CHANNELS_RSSI, FEATURES_GEN, NUM_CLASSES, APs, WINDOW_SIZE, GEN_EMBEDDING).to(device)
    gen = load_GAN_model(gen, save_directory, model_name, num_epochs, device)
    # get number of fake data
    unique_y, counts_y = np.unique(real_data_labels, return_counts=True)
    num_fake = []
    for i in range (0, NUM_CLASSES):
        if low_bound:
            if counts_y[i] > low_bound:
                num_fake.append(0)
                continue
        if total_number-counts_y[i] > 0:
            num_fake.append(total_number-counts_y[i])
        else:
            num_fake.append(0)
    # generate fake data
    fake_data, y_fake = generate_fake_data(gen, Z_DIM, num_fake=num_fake, total_room=NUM_CLASSES, aps= APs, device=device)
    return fake_data, y_fake, gen

def get_fake_rssi_equal(real_data_labels, num_epochs, total_number, NUM_CLASSES, APs, save_directory,
                  model_name, device):
    # generator parameter
    Z_DIM = 11
    CHANNELS_RSSI = 1
    FEATURES_GEN = 32
    WINDOW_SIZE = 20
    GEN_EMBEDDING = 100
    # load pretrained generator
    gen = Generator(Z_DIM, CHANNELS_RSSI, FEATURES_GEN, NUM_CLASSES, APs, WINDOW_SIZE, GEN_EMBEDDING).to(device)
    gen = load_GAN_model(gen, save_directory, model_name, num_epochs, device)
    # get number of fake data
    unique_y, counts_y = np.unique(real_data_labels, return_counts=True)
    num_fake = []
    for i in range(0, NUM_CLASSES):
        num_fake.append(total_number - min(counts_y))
    # generate fake data
    fake_data, y_fake = generate_fake_data(gen, Z_DIM, num_fake=num_fake, total_room=NUM_CLASSES, aps=APs,
                                           device=device)
    return fake_data, y_fake, gen

def get_real_equal(real_data, real_data_labels, APs):
    unique_y, counts_y = np.unique(real_data_labels, return_counts=True)
    num_real = min(counts_y)
    data_select_all = np.asarray([]).reshape(-1,20,APs)
    labels_select_all = np.asarray([])
    for i in range(len(unique_y)):
        room_data = real_data[real_data_labels == i+1]
        idx_select = np.random.choice(len(room_data), num_real, replace=False)

        data_select = room_data[idx_select]
        labels_select = np.full(num_real, i+1)

        data_select_all = np.concatenate((data_select_all, data_select), axis=0)
        labels_select_all = np.concatenate((labels_select_all, labels_select), axis=0)

    return data_select_all, labels_select_all

from general_utils import uniq_count
def select_data_potion(X, y, potion, no_label):
    idx = y.argsort()
    unique_y, counts_y = np.unique(y, return_counts=True)
    counter = []
    accum = 0
    for i in range(len(counts_y)):
        counter.append(accum)
        accum += counts_y[i]
    counter.append(len(X))
    select_idx = np.array([], dtype=int)
    for i in range(0, no_label):
        idx_room = idx[counter[i]:counter[i+1]]
        idx_room_sort = idx_room[idx_room.argsort()]
        select_idx_room = idx_room_sort[0:counts_y[i] * potion// 100]
        select_idx = np.append(select_idx, select_idx_room)

    y_select = y[select_idx]
    X_select = X[select_idx]

    return X_select, y_select

