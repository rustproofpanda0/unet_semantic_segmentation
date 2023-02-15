import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import ImageDataset
from unet import UNet
from trainer import Trainer
from losses import DiceLoss


device = torch.device("cpu")
if(torch.cuda.is_available()):
    device = torch.device("cuda")
if(torch.backends.mps.is_available()):
    device = torch.device("mps")
print(f"device = {device}")


def target_transform(img):
    img[img == img.min()] = 0
    img[img == img.max()] = 1
    img = img.to(torch.float)
    # return img.to(device)
    return img

def inp_transform(img):
    img = img.to(torch.float)
    # return img.to(device)
    return img


train_dataset = ImageDataset("./archive/binary_dataset_new/binary_dataset",
                       transform=inp_transform,
                       target_transform=target_transform,
                       use_cache=True)
train_dataloader = dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)


model = UNet(in_channels=3, out_channels=1, init_features=32).to(device)

# loss_func = torch.nn.BCELoss()
loss_func = DiceLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

trainer = Trainer(model=model,
                  loss_func=loss_func,
                  optimizer=optimizer,
                  train_dataloader=train_dataloader,
                  valid_dataloader=None,
                  epochs=225,
                  lr_scheduler=None,
                  device=device)

tl, vl = trainer.train()

torch.save(model.state_dict(), "model.pt")

np.save('train_loss.npy', np.array(tl))




