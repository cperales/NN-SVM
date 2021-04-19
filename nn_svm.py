# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer, metrics
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.utils._testing import ignore_warnings
from sklearn.utils import warnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
from collections import OrderedDict

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


class NNSVM(LightningModule):
    def __init__(self, W, batch_size, train_dataset):
        super().__init__()
        self._set_hparams({'batch_size': batch_size, 'learning_rate': 5e-3})
        self.train_dataset = train_dataset
        input_dim = self.train_dataset.get_input_dim()
        self.phi = nn.Sequential(nn.Linear(input_dim, W),
                                 nn.ReLU(True),
                                 nn.Linear(W, W),
                                 nn.ReLU(True),
                                 nn.Linear(W, W),
                                 nn.ReLU(True),
                                 nn.Linear(W, W),
                                 nn.ReLU(True),
                                 nn.Linear(W, W),
                                 nn.ReLU(True),
                                 nn.Linear(W, W),
                                 nn.ReLU(True),
                                 )
        self.svm = LinearSVC()
        self.loss = nn.BCELoss()
        self.accuracy = metrics.Accuracy()

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.05)
        return [optimizer], [scheduler]

    def forward(self, x):
        phi_x = self.phi(x).clone().detach().cpu().numpy()
        y = torch.from_numpy(self.svm.predict(phi_x)).requires_grad_(True)
        if self.on_gpu:
            y = y.cuda()
        return y

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def test_step(self, data_batch, batch_nb):
        x, y = data_batch
        y = y.view(1, -1)[0]
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        acc = self.accuracy(y_pred.long(), y.long())
        output = dict({
            'test_loss': loss,
            'test_acc': acc,  # everything must be a tensor
        })
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        phi_x = self.phi(x)
        phi_x_copy = phi_x[:, :]
        y_copy = y[:]
        self.svm.fit(phi_x_copy.detach().cpu().numpy(), y_copy.detach().cpu().numpy())
        y_pred = self(x)
        y = y.view(1, -1)[0]
        loss = self.loss(y_pred, y)

        acc = self.accuracy(y_pred.long(), y.long())
        output = OrderedDict({
            'loss': loss,
            'train_acc': acc,  # everything must be a tensor
        })
        return loss

    def test_epoch_end(self, outputs):
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        results = {'log': {'avg_test_loss': avg_loss, 'avg_acc': avg_acc}}
        return results


class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.input_dim = data.shape[1]
        self.samples = [(torch.from_numpy(d).float(), torch.from_numpy(t).float()) for d, t in zip(data, target)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_input_dim(self):
        return self.input_dim


if __name__ == '__main__':
    # Breast cancer dataset
    data, target = load_breast_cancer(return_X_y=True)
    input_dim = data.shape[1]
    target = target.reshape(-1, 1)
    data_train, data_test, target_train, target_test = \
        train_test_split(data, target, test_size=0.33)

    # Linear SVM
    svm = LinearSVC()
    svm.fit(data_train, target_train)
    print('Linear SVM accuracy =', np.mean(svm.predict(data_train) == target_train))
    print('Class 0 in train', (target_train == 0).sum() / target_train.shape[0], '%')
    print('Class 1 in train', (target_train == 1).sum() / target_train.shape[0], '%')
    print('Class 0 in test', (target_test == 0).sum() / target_test.shape[0], '%')
    print('Class 1 in test', (target_test == 1).sum() / target_test.shape[0], '%')

    # NN Kernel SVM
    train_dataset = CustomDataset(data_train, target_train)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, num_workers=4)
    test_dataset = CustomDataset(data_test, target_test)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, num_workers=4)

    W = input_dim * 10
    batch_size = len(data_train)

    model = NNSVM(W=W, batch_size=batch_size, train_dataset=train_dataset)
    trainer = Trainer(max_epochs=10)  # , gpus=-1)  # For GPUs
    trainer.fit(model=model, train_dataloader=train_dataloader)
    test_output = trainer.test(model=model, test_dataloaders=test_dataloader, verbose=True)
