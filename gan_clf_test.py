import numpy as np
from gan_evaluation import multiclass
from GAN.con_gan_rssi.model_wgan_gp import Discriminator, Generator, initialize_weights
from GAN.gan_utils import load_GAN_model, generate_fake_data,\
    MinMaxScaler, renormalization, windowing, get_mean_stat,\
    plot_line_rssi_gan, MLPClassifier, map_A, map_C, map_D,\
    binary_convert, num_room_house, train, test

def gan_select_epoch(house_name, reduce_ap, NUM_CLASSES, APs, total_number, num_epochs, save_directory,
                     windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, device):
    Z_DIM = 11
    CHANNELS_RSSI = 1
    FEATURES_GEN = 32
    WINDOW_SIZE = 20
    GEN_EMBEDDING = 100
    GANmodel = "conGAN-CNN_house_" + house_name
    model_name = "ConGAN_wgp_rep_house_" + house_name + "_reduce_" + str(reduce_ap)
    gen = Generator(Z_DIM, CHANNELS_RSSI, FEATURES_GEN, NUM_CLASSES, APs, WINDOW_SIZE, GEN_EMBEDDING).to(device)
    gen = load_GAN_model(gen, save_directory, model_name, num_epochs, device)
    # generate fake data
    unique_y, counts_y = np.unique(windowed_label, return_counts=True)
    # get number of fake data
    num_fake = []
    for i in range(0, NUM_CLASSES):
        if total_number - counts_y[i] > 0:
            num_fake.append(total_number - counts_y[i])
        else:
            num_fake.append(0)
    fake_data, y_fake = generate_fake_data(gen, Z_DIM, num_fake=num_fake, total_room=NUM_CLASSES, aps=APs, device=device)

    acc, f1 = multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                         APs, NUM_CLASSES, GANmodel, device, test_set='flive', exp=2, runs=3,
                         show_epoch=False, flatten=True, confusion_met=False, rt=True)

    return acc, f1

if __name__ == "__main__":
    data_directory = os.path.join('..', 'aloe', 'localisation', 'data', ''.format(os.path.sep))
    # data_directory = os.path.join('..', 'SimCLR', 'localisation', 'data', ''.format(os.path.sep))
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    house_name = 'D'
    reduce_ap = False
    if house_name == 'A':
        col_idx_use = [1, 2, 3, 4, 5, 6, 7, 8]
    else:
        col_idx_use = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    col_idx_use_label = col_idx_use[len(col_idx_use) - 1] + 1

    if reduce_ap:
        if house_name == 'C':
            col_idx_use = [1, 2, 4, 7, 9, 10]

    house_file = 'csv_house_' + house_name + '_fp.csv'
    ori_data = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=col_idx_use)
    label = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=[col_idx_use_label])

    house_file = 'csv_house_' + house_name + '_fl.csv'
    ori_data_fl = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=col_idx_use)
    label_fl = np.loadtxt(data_directory + 'csv_house_C_fl.csv', delimiter=",", skiprows=1, usecols=[col_idx_use_label])

    # data normalisation
    norm_data, min_val, max_val = MinMaxScaler(ori_data)
    norm_data_fl, min_val_fl, max_val_fl = MinMaxScaler(ori_data_fl)
    # get window data
    windowed_data, windowed_label = windowing(norm_data, label, seq_len=20, hop_size=10)
    windowed_data_fl, windowed_label_fl = windowing(norm_data_fl, label_fl, seq_len=20, hop_size=10)

    plot_line_rssi_gan(windowed_data[0], windowed_label[0], transpose=False, house=house_name,
                       house_map=map_D,
                       full_scale=False, ymin=-0.1, ymax=1.1, save=False, model_name='house_D_real')

    get_stats = False
    if get_stats:
        ori_data = np.loadtxt(data_directory + 'csv_house_C_fp.csv', delimiter=",", skiprows=1,
                              usecols= col_idx_use)
        label = np.loadtxt(data_directory + 'csv_house_C_fp.csv', delimiter=",", skiprows=1, usecols=[col_idx_use_label])
        norm_data, min_val, max_val = MinMaxScaler(ori_data)
        windowed_data, windowed_label = windowing(norm_data, label, seq_len=20, hop_size=10)
        windowed_data = np.transpose(windowed_data, (0, 2, 1))
        get_mean_stat(windowed_data, windowed_label, number_APs=11, room_num=9, save=False, plot_name=None)

    conGAN = False
    if conGAN:
        GANmodel = "conGAN-CNN_house_"+house_name
        # save_directory = os.path.join('..', 'SimCLR', 'GAN', 'save_GANs', ''.format(os.path.sep))
        save_directory = os.path.join('..', 'aloe', 'GAN', 'save_GANs', ''.format(os.path.sep))
        # model_name = "ConGAN_wgp"
        # model_name = "ConGAN_wgp_task_house_" + house_name
        model_name = "ConGAN_wgp_task_house_" + house_name + "_reduce_" + str(reduce_ap)
        # generator parameter
        num_epochs = 460
        Z_DIM = 11
        CHANNELS_RSSI = 1
        FEATURES_GEN = 32
        NUM_CLASSES = num_room_house[house_name]
        APs = len(col_idx_use)
        WINDOW_SIZE = 20
        GEN_EMBEDDING = 100
        # load pretrained generator
        gen = Generator(Z_DIM, CHANNELS_RSSI, FEATURES_GEN, NUM_CLASSES, APs, WINDOW_SIZE, GEN_EMBEDDING).to(device)
        gen = load_GAN_model(gen,save_directory, model_name, num_epochs, device)
        # generate fake data
        total_number = 1000
        unique_y, counts_y = np.unique(windowed_label, return_counts=True)
        # get number of fake data
        num_fake=[]
        for i in range (0, NUM_CLASSES):
            if total_number-counts_y[i] > 0:
                num_fake.append(total_number-counts_y[i])
            else:
                num_fake.append(0)
        fake_data, y_fake = generate_fake_data(gen, Z_DIM, num_fake=num_fake, total_room=NUM_CLASSES, aps= APs, device=device)

    tGAN=False
    if tGAN:
        save_directory = os.path.join('..', 'aloe', 'TimeGAN', ''.format(os.path.sep))
        ori_data = np.loadtxt(data_directory + 'csv_house_C_fp.csv', delimiter=",", skiprows=1, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        label = np.loadtxt(data_directory + 'csv_house_C_fp.csv', delimiter=",", skiprows=1, usecols=[12])
        # plot_all_rssi(ori_data[0:150], ap_map=ap_map)
        ori_data_fl = np.loadtxt(data_directory + 'csv_house_C_fl.csv', delimiter=",", skiprows=1,usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        label_fl = np.loadtxt(data_directory + 'csv_house_C_fl.csv', delimiter=",", skiprows=1, usecols=[12])
        GANmodel = 'lstm'
        room = 'stairs'
        iteration = 10000
        generated_data = torch.load(save_directory + 'data/fake_' +  room+ '_C_10hop_' + str(GANmodel) + str(iteration) + '.pt')
        # generated_data = torch.load(save_directory + 'data/fake_live_C' + str(10000) + '.pt')

        def load_fake_date(save_directory, room, GANmodel, iteration):
            generated_data = torch.load(save_directory + 'data/fake_' + room + '_C_10hop_' + str(GANmodel) + str(iteration) + '.pt')
            return generated_data

        fake_live = load_fake_date(save_directory, 'live', GANmodel, iteration)
        fake_kitchen = load_fake_date(save_directory, 'kitchen', GANmodel, iteration)
        fake_stairs = load_fake_date(save_directory, 'stairs', GANmodel, iteration)
        fake_outside = load_fake_date(save_directory, 'outside', GANmodel, iteration)
        fake_hallway = load_fake_date(save_directory, 'hallway', GANmodel, iteration)
        fake_bathroom = load_fake_date(save_directory, 'bathroom', GANmodel, iteration)
        fake_bedroom2 = load_fake_date(save_directory, 'bedroom-2', GANmodel, iteration)
        fake_bedroom1 = load_fake_date(save_directory, 'bedroom-1', GANmodel, iteration)
        fake_study = load_fake_date(save_directory, 'study', GANmodel, iteration)

        fake_data = np.concatenate((fake_live, fake_kitchen,fake_stairs,fake_outside,
                                   fake_hallway,fake_bathroom,fake_bedroom2,fake_bedroom1,fake_study),axis=0)

        y_fake = np.concatenate((np.full((len(fake_live)), 0), np.full((len(fake_kitchen)), 1),
                                 np.full((len(fake_stairs)), 2), np.full((len(fake_outside)), 3),
                                 np.full((len(fake_hallway)), 4), np.full((len(fake_bathroom)), 5),
                                 np.full((len(fake_bedroom2)), 6), np.full((len(fake_bedroom1)), 7),
                                 np.full((len(fake_study)), 8)), axis=0)

        renorm = False
        if renorm:
            windowed_data, windowed_label = windowing(ori_data, label, seq_len=20, hop_size=10)
            windowed_data_fl, windowed_label_fl = windowing(ori_data_fl, label_fl, seq_len=20, hop_size=10)

            norm_data, min_val, max_val = MinMaxScaler(ori_data)
            generated_data = renormalization(generated_data, min_val, max_val)

        else:
            norm_data, min_val, max_val = MinMaxScaler(ori_data)
            norm_data_fl, min_val_fl, max_val_fl = MinMaxScaler(ori_data_fl)

            windowed_data, windowed_label = windowing(norm_data, label, seq_len=20, hop_size=10)
            windowed_data_fl, windowed_label_fl = windowing(norm_data_fl, label_fl, seq_len=20, hop_size=10)

    dcGAN=False
    if dcGAN:
        save_directory = os.path.join('..', 'aloe', 'GAN', 'save_GANs', ''.format(os.path.sep))
        _, ori_data, windowed_label = torch.load(data_directory + 'house_C_fp_4s_na-120.pt')
        _, ori_data_fl, windowed_label_fl = torch.load(data_directory + 'house_C_4s_na-120.pt')

        trans = transforms.Normalize(np.mean(ori_data), np.std(ori_data))
        windowed_data = trans(torch.from_numpy(ori_data))
        windowed_data_fl = trans(torch.from_numpy(ori_data_fl))

        model_name = 'Relu_Noact_living'
        NUM_EPOCHS = 500
        NOISE_DIM = 11
        APs = 11
        FEATURES_GEN = 32
        pretrained_dir = save_directory + model_name + '_epoch' + str(NUM_EPOCHS) + '.pt'
        state_dict = torch.load(pretrained_dir, map_location=device)
        gen = Generator(NOISE_DIM, APs, FEATURES_GEN).to(device)
        gen.load_state_dict(state_dict, strict=True)
        noise = torch.randn(500, NOISE_DIM, 20).to(device)
        generated_data = gen(noise)

    old = False
    if old:
        room_label = 1
        room_data = windowed_data[windowed_label == room_label]
        idx = 40

        keep = [fake_live, fake_kitchen,
                fake_stairs, fake_outside,
                fake_hallway, fake_bathroom,
                fake_bedroom2, fake_bedroom1, fake_study]
        for i, data in enumerate(keep):
            generated_data = renormalization(data, min_val, max_val)
            plot_line_rssi_gan(generated_data[idx], i + 1, label_map=label_map, ap_map=ap_map, ymin=-121, ymax=-20)

        if tGAN:
            if renorm:
                plot_line_rssi_gan(room_data[idx], room_label, label_map=label_map, ap_map=ap_map, ymin=-121, ymax=-20)
                plot_line_rssi_gan(generated_data[idx], room_label, label_map=label_map, ap_map=ap_map, ymin=-121,
                                   ymax=-20)

        if dcGAN:
            from visualise import plot_line_rssi_one

            plot_line_rssi_one(room_data[idx].numpy(), room_label, label_map=label_map, ap_map=ap_map, ymin=-5, ymax=5)
            plot_line_rssi_one(generated_data[idx].cpu().detach().numpy(), room_label, label_map=label_map,
                               ap_map=ap_map, ymin=-5, ymax=5)

    # binary classification
    binary = False
    if binary:
        X_fp, y_fp = binary_convert(windowed_data, windowed_label, label=1)
        X_fl, y_fl = binary_convert(windowed_data_fl, windowed_label_fl, label=1)
        X1, X2, y1, y2 = train_test_split(X_fp, y_fp, test_size=0.3, shuffle=True, stratify=y_fp, random_state=42)
        X_train_, y_train_ = torch.from_numpy(X1), torch.from_numpy(y1)

        test_set = 'fp'
        if test_set == 'fp':
            X_test = torch.from_numpy(X2)
            y_test = torch.from_numpy(y2)

        if test_set == 'flive':
            X_test = torch.from_numpy(X_fl)
            y_test = torch.from_numpy(y_fl)

        exp = 1
        if exp == 1:  # Experiment 1
            X_train = X_train_.to(device)
            y_train = y_train_.to(device)

        if exp == 2:  # Experiment 2
            X_train = torch.cat((X_train_, torch.from_numpy(generated_data)), dim=0)
            y_train = torch.cat((y_train_, torch.ones(len(generated_data))), dim=0)

        X_test = X_test.to(device)
        y_test = y_test.to(device)

        flatten = True
        if flatten:
            X_train = torch.reshape(X_train, (-1, 220)).to(device)
            X_test = torch.reshape(X_test, (-1, 220)).to(device)

        train_dataset = torch.utils.data.TensorDataset(X_train.float().to(device), y_train.float().to(device))
        test_dataset = torch.utils.data.TensorDataset(X_test.float().to(device), y_test.float().to(device))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=128,
                                                   shuffle=True,
                                                   drop_last=True, )
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=128,
                                                  shuffle=False,
                                                  drop_last=True, )

        from SimCLR.simclr.modules import LogisticRegression, MLPClassifier

        model_type = "mlp"
        if model_type == "log":
            model = LogisticRegression(220, 2).to(device)
        if model_type == "mlp":
            model = MLPClassifier(220, 2).to(device)

        print(test_set, 'experiment', exp, model_type, 're-norm', renorm, 'model', GANmodel)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss()
        epochs = 200

        train(epochs, train_loader, model, optimizer, criterion)
        test(test_loader, model, criterion)

        exp = 2
        if exp == 1:  # Experiment 1
            X_train = X_train_.to(device)
            y_train = y_train_.to(device)

        if exp == 2:  # Experiment 2
            X_train = torch.cat((X_train_, torch.from_numpy(generated_data)), dim=0)
            y_train = torch.cat((y_train_, torch.ones(len(generated_data))), dim=0)

        X_test = X_test.to(device)
        y_test = y_test.to(device)

        flatten = True
        if flatten:
            X_train = torch.reshape(X_train, (-1, 220)).to(device)
            X_test = torch.reshape(X_test, (-1, 220)).to(device)

        train_dataset = torch.utils.data.TensorDataset(X_train.float().to(device), y_train.float().to(device))
        test_dataset = torch.utils.data.TensorDataset(X_test.float().to(device), y_test.float().to(device))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=128,
                                                   shuffle=True,
                                                   drop_last=True, )
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=128,
                                                  shuffle=False,
                                                  drop_last=True, )

        from SimCLR.simclr.modules import LogisticRegression, MLPClassifier

        model_type = "mlp"
        if model_type == "log":
            model = LogisticRegression(220, 2).to(device)
        if model_type == "mlp":
            model = MLPClassifier(220, 2).to(device)

        print(test_set, 'experiment', exp, model_type, 're-norm', renorm, 'model', GANmodel)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss()
        epochs = 200

        train(epochs, train_loader, model, optimizer, criterion)
        test(test_loader, model, criterion)
