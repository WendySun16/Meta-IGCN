import numpy as np
from tqdm import tqdm
import os
import torch
import torch.optim as optim
from utils import normalize, load_ori_data
from models import AutoEncoder, SE


nfeat = 4005
nhid = 128
npre = 32
nout = 4005

epochs = 150
batch_size = 8
learning_rate = 0.001

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device('cuda')


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.path = "/data/autoencoder/train.npy"
        self.features = torch.from_numpy(np.float32(np.load(self.path, allow_pickle=True)))


    def __getitem__(self, index):
        return self.features[index]


    def __len__(self):
        return len(self.features)
    

class ValDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.path = "/data/autoencoder/val.npy"
        self.features = torch.from_numpy(np.float32(np.load(self.path, allow_pickle=True)))


    def __getitem__(self, index):
        return self.features[index]


    def __len__(self):
        return len(self.features)
    

# SE block
def get_weight(features):
    features_T = features.T
    features_T = np.mean(features_T, axis=1)
    features_T = torch.FloatTensor(features_T)

    model = SE(nfeat = 4005, ratio = 16)
    model = model.to(device)
    features_T = features_T.to(device)
    output = model(features_T)

    output = output.cpu().detach().numpy()

    output = output.T
    weight = np.tile(output, (features.shape[0], 1))
    features = np.multiply(features, weight)
    return features
    

def shuffle_feat(data):
    np.random.seed(42)
    index_list = np.random.permutation(len(data))
    val_size = len(data) // 5
    val_idx = index_list[:val_size]
    train_idx = index_list[val_size:]

    train = normalize(data[train_idx])
    val = normalize(data[val_idx])

    train_file = "/data/autoencoder/train.npy"
    np.save(train_file, train)
    val_file = "/data/autoencoder/val.npy"
    np.save(val_file, val)
    
    return train, val


def train():
    train_data = TrainDataset()
    val_data = ValDataset()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle = False)

    loss_func = torch.nn.MSELoss()
    model = AutoEncoder(nfeat, nhid, npre, nout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = 100

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        iter = 0
        for feat in tqdm(train_loader, desc=f'train || epoch: {epoch + 1}'):
            feat = feat.to(device)
            _, high_feat = model(feat)
            loss = loss_func(high_feat, feat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            iter += 1
        print('Train loss:', train_loss / iter)

        model.eval()
        val_loss = 0
        val_iter = 0
        for feat in tqdm(val_loader, desc=f'val || epoch: {epoch + 1}'):
            feat = feat.to(device)
            _, high_feat = model(feat)
            loss = loss_func(high_feat, feat)
            val_loss += loss.item()
            val_iter += 1
        loss_mean = val_loss / val_iter
        print('Validation loss:', loss_mean)
        if loss_mean < best_loss:
            best_loss = loss_mean
            best_model = "/data/autoencoder/best_model.pkl"
            torch.save(model.state_dict(), best_model)


def main():
    # step1: get original data
    dataset1_csv = "/data/alzheimer_data/dataset1.csv"
    dataset1_path = "/data/alzheimer_data/dataset1"
    dataset2_csv = "/data/alzheimer_data/dataset2.csv"
    dataset2_path = "/data/alzheimer_data/dataset2"
    dataset3_csv = "/data/alzheimer_data/dataset3.csv"
    dataset3_path = "/data/alzheimer_data/dataset3"
    feature1, label1, sex1, age1, apoe1 = load_ori_data(dataset1_csv, dataset1_path)
    feature2, label2, sex2, age2, apoe2 = load_ori_data(dataset2_csv, dataset2_path)
    feature3, label3, sex3, age3, apoe3 = load_ori_data(dataset3_csv, dataset3_path)

    features = feature1 + feature2 + feature3
    labels = label1 + label2 + label3
    sexes = sex1 + sex2 + sex3
    ages = age1 + age2 + age3
    apoes = apoe1 + apoe2 + apoe3

    sexes_file = "/data/ad/sexes.npy"
    np.save(sexes_file, sexes)
    ages_file = "/data/ad/ages.npy"
    np.save(ages_file, ages)
    apoes_file = "/data/ad/apoes.npy"
    np.save(apoes_file, apoes)
    features_file = "/data/ad/features.npy"
    np.save(features_file, features)
    labels_file = "/data/ad/labels.npy"
    np.save(labels_file, labels)

    # step2: SE block
    features_file = "/data/ad/features.npy"
    features = np.load(features_file, allow_pickle=True)
    features = get_weight(features)
    features_weight_file = "/data/ad/features_weight.npy"
    np.save(features_weight_file, features)

    # step3: divide AE training set and validation set
    features = np.load("/data/ad/features_weight.npy", allow_pickle=True)
    train_data, val_data = shuffle_feat(features)

    # step4: AE training
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] ='0'
    train()

    # step5: reduce the dimensions of all data
    coder = AutoEncoder(nfeat, nhid, npre, nout)
    coder.load_state_dict(torch.load("/data/autoencoder/best_model.pkl"))
    features = normalize(features)
    features = torch.from_numpy(np.float32(features))
    encoded, _ = coder(features)
    encoded = encoded.detach().numpy()
    encoded_file = "/data/ad/features_encoded.npy"
    np.save(encoded_file, encoded)
    

if  __name__=='__main__':
    main()