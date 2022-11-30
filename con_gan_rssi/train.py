import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from GAN.con_gan_rssi.model import Discriminator, Generator, initialize_weights
from visualise import plot_line_rssi_one
from sklearn.metrics import accuracy_score
from GAN.gan_evaluation import MinMaxScaler

ap_map = {'a': 'living_room_1',
              'b': 'living_room_2',
              'c': 'kitchen_1',
              'd': 'kitchen_2',
              'e': 'kitchen_3',
              'f': 'hallway_upper',
              'g': 'bathroom',
              'h': 'bedroom-two',
              'i': 'bedroom-one_1',
              'j': 'bedroom-one_2',
              'k': 'study',
              }
label_map = {1: 'living_room',
             2: 'kitchen',
             3: 'stairs',
             4: 'outside',
             5: 'hallway',
             6: 'bathroom',
             7: 'bedroom-2',
             8: 'bedroom-1',
             9: 'study',
            'living_room':1,
             'kitchen'  :2,
             'stairs'   :3,
             'outside'  :4,
             'hallway'  :5,
             'bathroom' :6,
             'bedroom-2':7,
             'bedroom-1':8,
             'study'    :9,
             }


# def plot_line_rssi_one(rssi, label, idx, label_map=None, ap_map=None, save=False):
#     rssi_plot = rssi[idx]
#     rssi_plot = pd.DataFrame(np.transpose(rssi_plot), columns=['g', 'd', 'f', 'k', 'c', 'b', 'a', 'j', 'h', 'i', 'e'])
#     label_plot = label[idx]
#     if ap_map:
#         rssi_plot = rssi_plot.rename(ap_map, axis='columns')
#     if label_map:
#         label_plot = label_map[label_plot]
#     # plot the data
#     # sns.set(rc={'figure.figsize': (8, 4)})
#     sns.set(rc={'figure.figsize': (11, 5)})
#     sns.lineplot(data=rssi_plot, legend=True)
#     # sns.displot(rssi_plot, kind='kde', fill=fill, height=5, aspect=2.5)
#     # plt.xlim(-0.1, 21)
#     plt.ylim(-130, -30)
#
#     plt.title("label %s " % (label_plot), fontsize=18)
#     plt.xlabel("Timepoint", fontsize=18)
#     plt.xticks(fontsize=18)
#     plt.ylabel("RSSI(dB)", fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
#
#     # plt.legend(loc='lower left', bbox_to_anchor=(0, 1.01, 1.0, 0.5), markerfirst=True,
#     #            mode="expand", borderaxespad=0, ncol=9, handletextpad=0.01, )
#     plt.tight_layout()
#     if save:
#         plt.savefig('GAN/lineplot_all/'+ label_plot + '/'+ label_plot +'_idx' + str(idx) + '.png')
#     plt.show()
#     plt.close()

def uniq_count(y, show=None):
    unique_y, counts_y= np.unique(y, return_counts=True)
    if show == 'on':
        print("Total Number:", len(y))
        print("Unique value:", unique_y)
        print("counts value:", counts_y)
    return unique_y

def data_normalise(x):
    x_max = np.max(x)
    x_min = np.min(x)
    return (x-x_min)/x_max-x_min

def write_aps(x, t, type, epoch_step):
    writer_ap1.add_scalar("Line Plot"+ type + epoch_step, x[0, :][t], t)
    writer_ap2.add_scalar("Line Plot"+ type + epoch_step, x[1, :][t], t)
    writer_ap3.add_scalar("Line Plot"+ type + epoch_step, x[2, :][t], t)
    writer_ap4.add_scalar("Line Plot"+ type + epoch_step, x[3, :][t], t)
    writer_ap5.add_scalar("Line Plot"+ type + epoch_step, x[4, :][t], t)
    writer_ap6.add_scalar("Line Plot"+ type + epoch_step, x[5, :][t], t)
    writer_ap7.add_scalar("Line Plot"+ type + epoch_step, x[6, :][t], t)
    writer_ap8.add_scalar("Line Plot"+ type + epoch_step, x[7, :][t], t)
    writer_ap9.add_scalar("Line Plot"+ type + epoch_step, x[8, :][t], t)
    writer_ap10.add_scalar("Line Plot"+ type + epoch_step, x[9, :][t], t)
    writer_ap11.add_scalar("Line Plot"+ type + epoch_step, x[10, :][t], t)

label_map = {1: 'living_room',
             2: 'kitchen',
             3: 'stairs',
             4: 'outside',
             5: 'hallway',
             6: 'bathroom',
             7: 'bedroom-2',
             8: 'bedroom-1',
             9: 'study',
            'living_room':1,
             'kitchen'  :2,
             'stairs'   :3,
             'outside'  :4,
             'hallway'  :5,
             'bathroom' :6,
             'bedroom-2':7,
             'bedroom-1':8,
             'study'    :9,
             }
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
if __name__ == "__main__":
    # load dataset
    data_directory = os.path.join('..', 'SimCLR', 'localisation', 'data', ''.format(os.path.sep))
    save_directory = os.path.join('..', 'SimCLR', 'GAN', 'save_GANs', ''.format(os.path.sep))
    # data_directory = os.path.join('..', 'aloe', 'localisation', 'data', ''.format(os.path.sep))
    # save_directory = os.path.join('..', 'aloe', 'GAN', 'save_GANs', ''.format(os.path.sep))
    # acc_fp, rssi_fp, y_fp = torch.load(data_directory + 'house_C_fp_4s_na-120.pt')
    # uniq_count(y_fp, show='on')
    ori_data = np.loadtxt(data_directory + 'csv_house_C_fp.csv', delimiter=",", skiprows=1,
                          usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    label = np.loadtxt(data_directory + 'csv_house_C_fp.csv', delimiter=",", skiprows=1, usecols=[12])
    norm_data, min_val, max_val = MinMaxScaler(ori_data)
    rssi_fp, y_fp = windowing(norm_data, label, seq_len=20, hop_size=10)
    y_fp = y_fp-1
    rssi_fp = np.transpose(rssi_fp, (0, 2, 1))
    rssi_fp2D = torch.from_numpy(rssi_fp)
    rssi_fp2D = rssi_fp2D.unsqueeze(1)

    # Hyperparameters etc.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
    BATCH_SIZE = 64
    WINDOW_SIZE = 20
    APs = 11
    NOISE_DIM = 11

    CHANNELS_RSSI = 1
    GEN_EMBEDDING = 100
    NUM_CLASSES = 9

    FEATURES_DISC = 32
    FEATURES_GEN = 32

    dataset = torch.utils.data.TensorDataset(rssi_fp2D.float(),
                                             torch.from_numpy(y_fp).int())

    dataloader = torch.utils.data.DataLoader(dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               drop_last=True,
                                               )
    gen = Generator(NOISE_DIM, CHANNELS_RSSI, FEATURES_GEN, NUM_CLASSES, APs, WINDOW_SIZE, GEN_EMBEDDING).to(device)
    disc = Discriminator(CHANNELS_RSSI, FEATURES_DISC, NUM_CLASSES, APs, WINDOW_SIZE).to(device)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.FloatTensor(BATCH_SIZE, NOISE_DIM, 11, 20).uniform_(0, 1).to(device)
    # fixed_noise = torch.randn(32, NOISE_DIM, 11, 20).to(device)
    # fixed_noise = torch.from_numpy(np.random.normal(loc=-70, scale=20, size=(32, NOISE_DIM, 20))).float().to(device)

    # dist input: norm_x
    # gen input: rand_noise
    model_name = "ConGAN_BCELoss"
    writer_loss = SummaryWriter(f"logs/"+model_name+"/loss/")

    writer_ap1 = SummaryWriter(f"logs/"+model_name+"/bathroom")
    writer_ap2 = SummaryWriter(f"logs/"+model_name+"/kitchen_2")
    writer_ap3 = SummaryWriter(f"logs/"+model_name+"/hall_up")
    writer_ap4 = SummaryWriter(f"logs/"+model_name+"/study")
    writer_ap5 = SummaryWriter(f"logs/"+model_name+"/kitchen_1")
    writer_ap6 = SummaryWriter(f"logs/"+model_name+"/living_2")
    writer_ap7 = SummaryWriter(f"logs/"+model_name+"/living_1")
    writer_ap8 = SummaryWriter(f"logs/"+model_name+"/bedroom-one_2")
    writer_ap9 = SummaryWriter(f"logs/"+model_name+"/bedroom-two")
    writer_ap10 = SummaryWriter(f"logs/"+model_name+"/bedroom-one_1")
    writer_ap11 = SummaryWriter(f"logs/"+model_name+"/kitchen_3")

    # step = 0
    gen.train()
    disc.train()
    # loss_gen_epoch = []
    # loss_disc_epoch = []
    NUM_EPOCHS = 500
    for epoch in range(0, NUM_EPOCHS):
        # Target labels not needed! <3 unsupervised
        accuracy_real = []
        accuracy_fake = []
        for batch_idx, (real, labels) in enumerate(dataloader):
            real = real.to(device)
            labels = labels.to(device)

            #create random noise in range -120 to 0
            noise = torch.FloatTensor(BATCH_SIZE, NOISE_DIM, 11, 20).uniform_(0, 1).to(device)
            # noise = torch.randn(BATCH_SIZE, NOISE_DIM, 11, 20).to(device)
            # noise = torch.from_numpy(np.random.normal(loc=-70, scale=20, size=(BATCH_SIZE, NOISE_DIM, 20))).float().to(device)
            fake = gen(noise, labels)

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            # temp_disc_real = disc(real)
            disc_real = disc(real, labels).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))

            # temp_disc_fake = disc(fake)
            disc_fake = disc(fake.detach(), labels).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            # predicted_real = disc(real).argmax(1)
            # predicted_fake = disc(fake).argmax(1)
            #
            # disc_real_acc = accuracy_score(y_real, predicted_real)
            # disc_fake_acc = accuracy_score(y_fake, predicted_fake)

            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()
            # loss_disc_epoch += loss_disc.item()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            output = disc(fake, labels).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            # loss_gen_epoch += loss_gen.item()

            writer_loss.add_scalar("Loss/disc_train_epoch", loss_disc.item(), epoch)
            writer_loss.add_scalar("Loss/gen_train_epoch", loss_gen.item(), epoch)

        # Print losses occasionally and print to tensorboard
        # if batch_idx % 10 == 0:
        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] \
              Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
        )

        with torch.no_grad():
            fake = gen(noise, labels)
            x_fake = fake[0].view(11,20)
            x_real = real[0].view(11,20)

            for t in (range(20)):
                write_aps(x_fake, t, type='fake/idx0/', epoch_step=str(epoch))
                write_aps(x_real, t, type='real/idx0/', epoch_step=str(epoch))

        # step += 1
        if epoch != 0 and epoch % 50 == 0:
            torch.save(gen.state_dict(), save_directory + model_name +'_epoch' + str(epoch) + '.pt')

        if epoch != 0 and epoch % 50 == 0:
            plot_line_rssi_one(x_fake.cpu().detach().numpy(), 1, label_map=label_map, ap_map=ap_map, full_scale=False)
            plot_line_rssi_one(x_real.cpu().detach().numpy(), 1, label_map=label_map, ap_map=ap_map, full_scale=False)
            # print(fake[0][0])

    torch.save(gen.state_dict(), save_directory + model_name +'_epoch' + str(NUM_EPOCHS) + '.pt')
    plot_line_rssi_one(x_fake.cpu().detach().numpy(), 1, label_map=label_map, ap_map=ap_map, full_scale=False)
    plot_line_rssi_one(x_real.cpu().detach().numpy(), 1, label_map=label_map, ap_map=ap_map, full_scale=False)

    # writer_real.add_scalar("Loss/dist_train", loss_disc_epoch / len(dataloader), epoch)
    # writer_fake.add_scalar("Loss/gen_train", loss_gen_epoch / len(dataloader), epoch)



    # writer_real_f = SummaryWriter(f"logs/real_norm_x_rand_noise/final500")
    # writer_fake_f = SummaryWriter(f"logs/fake_norm_x_rand_noise/final500")
    # fake = gen(fixed_noise)
    # x_fake = fake[0]
    # x_real = real[0]
    # for t in (range(20)):
    #     for ap in range(11):
    #         writer_real_f.add_scalar("Ap/ap" + str(ap), x_real[ap, :][t], t)
    #         writer_fake_f.add_scalar("Ap/ap" + str(ap), x_fake[ap, :][t], t)
