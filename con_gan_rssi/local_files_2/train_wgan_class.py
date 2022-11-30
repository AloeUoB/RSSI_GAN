import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from utils import gradient_penalty
from model_wgan_gp import Discriminator, Generator, initialize_weights

from GAN.gan_utils import load_GAN_model, generate_fake_data,\
    MinMaxScaler, renormalization, windowing, get_mean_stat,\
    plot_line_rssi_gan, MLPClassifier,\
    binary_convert, ap_map, label_map, train, test

def MinMaxScaler(data):
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val
    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)
    return norm_data, min_val, max_val

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

def plot_line_rssi_one(rssi_plot, label_plot, label_map=None, ap_map=None, full_scale=False, ymin=-5.0, ymax=5.0):
    rssi_plot = pd.DataFrame(np.transpose(rssi_plot), columns=['g', 'd', 'f', 'k', 'c', 'b', 'a', 'j', 'h', 'i', 'e'])
    if ap_map:
        rssi_plot = rssi_plot.rename(ap_map, axis='columns')
    if label_map:
        label_plot = label_map[label_plot]
    #plot the data
    sns.set_theme(rc={'figure.figsize': (8, 4)})
    sns.lineplot(data=rssi_plot, legend=True)
    # sns.displot(rssi_plot, kind='kde', fill=fill, height=5, aspect=2.5)
    # plt.xlim(-0.1, 21)
    plt.ylim(ymin, ymax) #(-4, 3.5)
    if full_scale:
        plt.ylim(-130, 5)
    # print(label_plot)
    plt.title("label %s" % (label_plot), fontsize=14)
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)

    # plt.legend(loc='lower left', bbox_to_anchor=(0, 1.01, 1.0, 0.5), markerfirst=True,
    #            mode="expand", borderaxespad=0, ncol=9, handletextpad=0.01, )
    plt.xlabel("Timepoint", fontsize=18)
    # plt.xticks(fontsize=22)
    plt.ylabel("RSSI(dB)", fontsize=18)
    # plt.yticks(fontsize=20)
    plt.tight_layout()

    plt.show()
    plt.close()

ap_map = {'a': 'living_room_1', 'b': 'living_room_2',   'c': 'kitchen_1',   'd': 'kitchen_2',
          'e': 'kitchen_3',     'f': 'hallway_upper',   'g': 'bathroom',    'h': 'bedroom-two',
          'i': 'bedroom-one_1', 'j': 'bedroom-one_2',   'k': 'study',}

label_map = {1: 'living_room',  2: 'kitchen',   3: 'stairs',    4: 'outside',   5: 'hallway',
             6: 'bathroom',     7: 'bedroom-2', 8: 'bedroom-1', 9: 'study',
            'living_room':1,    'kitchen'  :2,  'stairs'   :3,  'outside'  :4,
             'hallway'  :5,     'bathroom' :6,  'bedroom-2':7,  'bedroom-1':8,  'study'    :9,}

num_room_house = {'A':4, 'B':11, 'C':9, 'D':10}

if __name__ == "__main__":
    # load dataset
    # data_directory = os.path.join('..', 'SimCLR', 'localisation', 'data', ''.format(os.path.sep))
    # save_directory = os.path.join('..', 'SimCLR', 'GAN', 'save_GANs', ''.format(os.path.sep))
    data_directory = os.path.join('..', 'aloe', 'localisation', 'data', ''.format(os.path.sep))
    save_directory = os.path.join('..', 'aloe', 'GAN', 'save_GANs', ''.format(os.path.sep))
    # acc_fp, rssi_fp, y_fp = torch.load(data_directory + 'house_C_fp_4s_na-120.pt')
    house_name = 'B'
    reduce_ap = False
    house_file = 'csv_house_'+house_name+'_fp.csv'

    if house_name == 'A':
        col_idx_use = [1, 2, 3, 4, 5, 6, 7, 8]
    else:
        col_idx_use = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    col_idx_use_label = col_idx_use[len(col_idx_use)-1]+1

    if reduce_ap:
        if house_name == 'C':
            col_idx_use = [1, 2, 4, 7, 9, 10]

    ori_data = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=col_idx_use)
    label = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=[col_idx_use_label])

    norm_data, min_val, max_val = MinMaxScaler(ori_data)
    rssi_fp, y_fp = windowing(norm_data, label, seq_len=20, hop_size=10)
    y_fp = y_fp-1
    rssi_fp = np.transpose(rssi_fp, (0, 2, 1))

    X_train, X_val, y_train, y_val = train_test_split(rssi_fp, y_fp, test_size=0.3, shuffle=True,
                                      stratify=y_fp, random_state=42)

    X_train2D = torch.from_numpy(X_train)
    X_train2D = X_train2D.unsqueeze(1)

    # X_val2D = torch.from_numpy(X_val)
    # X_val2D = X_val2D.unsqueeze(1)
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train).int()
    X_val = torch.from_numpy(X_val)
    y_val = torch.from_numpy(y_val).int()
    X_val = torch.reshape(X_val, (-1,  len(X_val[0])*len(X_val[0,0])))

    # Hyperparameters etc.
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 64
    WINDOW_SIZE = 20
    CHANNELS_RSSI = 1
    GEN_EMBEDDING = 100
    Z_DIM = 11
    FEATURES_CRITIC = 32
    FEATURES_GEN = 32
    CRITIC_ITERATIONS = 5
    LAMBDA_GP = 10
    NUM_CLASSES = num_room_house[house_name]
    APs = len(col_idx_use)

    dataset_train = torch.utils.data.TensorDataset(X_train2D.float(), y_train)
    # dataset_val = torch.utils.data.TensorDataset(X_val2D.float(),
    #                                                torch.from_numpy(y_val).int())

    loader_train = torch.utils.data.DataLoader(dataset_train,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         drop_last=True,)

    # loader_val = torch.utils.data.DataLoader(dataset_train,
    #                                            batch_size=BATCH_SIZE,
    #                                            shuffle=True,
    #                                            drop_last=True, )

    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])
    gen = Generator(Z_DIM, CHANNELS_RSSI, FEATURES_GEN, NUM_CLASSES, APs, WINDOW_SIZE, GEN_EMBEDDING).to(device)
    critic = Discriminator(CHANNELS_RSSI, FEATURES_CRITIC, NUM_CLASSES, APs, WINDOW_SIZE).to(device)
    initialize_weights(gen)
    initialize_weights(critic)

    # initializate optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    # for tensorboard plotting
    fixed_noise = torch.randn(32, Z_DIM, APs, 20).uniform_(0, 1).to(device)
    fixed_noise_plot = fixed_noise[0][0]
    # plot_line_rssi_one(fixed_noise_plot .cpu().detach().numpy(), 1, label_map=label_map, ap_map=ap_map, ymin=-0.1, ymax=1.1)
    model_name = "ConGAN_wgp_task_house_" + house_name +"_reduce_" + str(reduce_ap)
    writer_loss = SummaryWriter(f"logs/" + model_name + "/loss/")
    #
    # writer_ap1 = SummaryWriter(f"logs/" + model_name + "/bathroom")
    # writer_ap2 = SummaryWriter(f"logs/" + model_name + "/kitchen_2")
    # writer_ap3 = SummaryWriter(f"logs/" + model_name + "/hall_up")
    # writer_ap4 = SummaryWriter(f"logs/" + model_name + "/study")
    # writer_ap5 = SummaryWriter(f"logs/" + model_name + "/kitchen_1")
    # writer_ap6 = SummaryWriter(f"logs/" + model_name + "/living_2")
    # writer_ap7 = SummaryWriter(f"logs/" + model_name + "/living_1")
    # writer_ap8 = SummaryWriter(f"logs/" + model_name + "/bedroom-one_2")
    # writer_ap9 = SummaryWriter(f"logs/" + model_name + "/bedroom-two")
    # writer_ap10 = SummaryWriter(f"logs/" + model_name + "/bedroom-one_1")
    # writer_ap11 = SummaryWriter(f"logs/" + model_name + "/kitchen_3")

    gen.train()
    critic.train()

    NUM_EPOCHS = 500
    for epoch in range(NUM_EPOCHS):
        if epoch == 50:
            pass
        for batch_idx, (real, labels) in enumerate(loader_train):
            real = real.to(device)
            cur_batch_size = real.shape[0]
            labels = labels.to(device)

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM, APs, 20).uniform_(0, 1).to(device)
                fake = gen(noise, labels)
                critic_real = critic(real, labels).reshape(-1)
                critic_fake = critic(fake, labels).reshape(-1)
                gp = gradient_penalty(critic, labels, real, fake, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = critic(fake, labels).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            writer_loss.add_scalar("Loss/disc_train_epoch", loss_critic.item(), epoch)
            writer_loss.add_scalar("Loss/gen_train_epoch", loss_gen.item(), epoch)

            # Print losses occasionally and print to tensorboard
        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] \
                      Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
        )

        # with torch.no_grad():
        #     num_fake = []
        #     # unique_y, counts_y = np.unique(windowed_label, return_counts=True)
        #     # total_number = 1000
        #     for i in range(0, NUM_CLASSES):
        #         # if total_number - counts_y[i] > 0:
        #         #     num_fake.append(total_number - counts_y[i])
        #         # else:
        #         num_fake.append(300)
        #     fake_val, fake_y_val = generate_fake_data(gen, Z_DIM, num_fake=num_fake, total_room=NUM_CLASSES, aps= APs, device=device)
        #
        # fake_val = fake_val.view(len(fake_val), 11, 20)
        # fake_val = fake_val.cpu().detach().numpy()
        # fake_y_val = fake_y_val.cpu().detach().numpy()
        # fake_val = torch.from_numpy(fake_val)
        # fake_y_val = torch.from_numpy(fake_y_val)
        #
        # val_data_train = torch.cat((X_train, fake_val), dim=0)
        # val_label_train = torch.cat((y_train, fake_y_val), dim=0)
        # val_data_train = torch.reshape(val_data_train, (-1, 220))
        # val_data_train = val_data_train.to(device)
        # val_label_train = val_label_train.to(device)
        #
        #
        # dataset_val = torch.utils.data.TensorDataset(val_data_train.float(), val_label_train)
        # loader_val = torch.utils.data.DataLoader(dataset_val,
        #                                          batch_size=128,
        #                                          shuffle=True,
        #                                          drop_last=True, )
        #
        # dataset_val_test = torch.utils.data.TensorDataset(X_val.float().to(device), y_val.to(device))
        # loader_val_test = torch.utils.data.DataLoader(dataset_val_test,
        #                                          batch_size=len(X_val),
        #                                          shuffle=False,
        #                                          drop_last=True, )
        #
        # clf = MLPClassifier(220, NUM_CLASSES).to(device)
        # optimizer = torch.optim.Adam(clf.parameters(), lr=3e-4)
        # criterion = torch.nn.CrossEntropyLoss()
        # clf_epochs = 200
        # for clf_epoch in range(clf_epochs):
        #     loss_epoch, accuracy_epoch, f1_epoch = train(loader_val, clf, optimizer, criterion)
        #     # train(train_loader, clf, optimizer, criterion)
        #     if clf_epoch % 50 == 0:
        #         print(
        #             f"Clf_Epoch [{clf_epoch}/{clf_epochs}]\t "
        #             f"Clf_Loss: {loss_epoch / len( loader_val)}\t "
        #             f"Clf_Accuracy: {accuracy_epoch * 100 / len( loader_val)}\t "
        #             f"Clf_F1: {f1_epoch * 100}\t")
        #
        #     writer_loss.add_scalar("Loss/"+str(epoch)+"/val_train_epoch", loss_epoch/len(loader_val), clf_epoch)
        #     writer_loss.add_scalar("Acc/"+str(epoch)+"/accuracy_val_train", accuracy_epoch* 100/len(loader_val), clf_epoch)
        #     writer_loss.add_scalar("F1/"+str(epoch)+"/F1_val_train", f1_epoch*100, clf_epoch)
        #
        # loss, accuracy, f1 = test(loader_val_test, clf, criterion)
        # print(
        #     f"Epoch [{epoch}/{NUM_EPOCHS}]\t "
        #     f"Clf_Loss: {loss}\t "
        #     f"Clf_Accuracy: {accuracy }\t "
        #     f"Clf_F1: {f1}\t")
        #
        # writer_loss.add_scalar("Loss/val_test_epoch", loss, epoch)
        # writer_loss.add_scalar("Acc/accuracy_val_test", accuracy, epoch)
        # writer_loss.add_scalar("F1/F1_val_test", f1, epoch)

            # x_fake = fake[0].view(11, 20)
            # x_real = real[0].view(11, 20)
            # for t in (range(20)):
            #     write_aps(x_fake, t, type='fake/idx0/', epoch_step=str(epoch))
            #     write_aps(x_real, t, type='real/idx0/', epoch_step=str(epoch))

            # step += 1
        if epoch != 0 and epoch % 20 == 0:
            torch.save(gen.state_dict(), save_directory + model_name + '_epoch' + str(epoch) + '.pt')

        # if epoch != 0 and epoch % 1 == 0:
        #     cur_labels_numpy = labels.cpu().detach().numpy()
            # print(cur_labels_numpy[0])
            # plot_line_rssi_one(x_fake.cpu().detach().numpy(), cur_labels_numpy[0]+1, label_map=label_map, ap_map=ap_map, ymin=-0.1, ymax=1.1)
            # plot_line_rssi_one(x_real.cpu().detach().numpy(), cur_labels_numpy[0]+1, label_map=label_map, ap_map=ap_map, ymin=-0.1, ymax=1.1)
            # print(fake[0][0])

    torch.save(gen.state_dict(), save_directory + model_name + '_epoch' + str(NUM_EPOCHS) + '.pt')
    # fake = gen(noise, labels)
    # x_fake = fake[0].view(11, 20)
    # x_real = real[0].view(11, 20)
    # cur_labels_numpy = labels.cpu().detach().numpy()
    # plot_line_rssi_one(x_fake.cpu().detach().numpy(), cur_labels_numpy[0], label_map=label_map, ap_map=ap_map, ymin=-0.1, ymax=1.1)
    # plot_line_rssi_one(x_real.cpu().detach().numpy(), cur_labels_numpy[0], label_map=label_map, ap_map=ap_map, ymin=-0.1, ymax=1.1)
