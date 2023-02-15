import numpy as np
import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, loss_func, optimizer,
                 train_dataloader, valid_dataloader,
                 epochs, lr_scheduler, device):
        
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.epochs = epochs
        self.lr_scheduler = lr_scheduler
        self.device = device

        self.train_loss = []
        self.valid_loss = []

    def train(self):

        try:
        
            pbar = tqdm(range(self.epochs))
            for ep in pbar:

                self._train()

                if self.valid_dataloader is not None:
                    self._validate()

        finally:
            return self.train_loss, self.valid_loss
        
        # pbar = tqdm(range(self.epochs))
        # for ep in pbar:

        #     self._train()

        #     if self.valid_dataloader is not None:
        #         self._validate()

        # return self.train_loss, self.valid_loss

    def _train(self):

        train_losses = []
        self.model.train()
        batch_iter = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))

        for i, (input, target) in batch_iter:

            input, target = input.to(self.device), target.to(self.device) 
            self.optimizer.zero_grad()
            
            
            # print(input.dtype)
            # print(input.device)
            # print(target.dtype)
            # print(target.device)

            # raise


            pred = self.model(input)


            loss = self.loss_func(pred, target)
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()
            self.optimizer.step()

            batch_iter.set_description(f"Training loss = {loss_value:.4f}")

        self.train_loss.append(np.mean(train_losses))

        batch_iter.close()


    def _validate(self):

        valid_losses = []
        self.model.eval()

        batch_iter = tqdm(enumerate(self.valid_dataloader), total=len(self.valid_dataloader))

        for i, (input, target) in batch_iter:

            with torch.no_grad():
                pred = self.model(input)
                loss = self.loss_func(pred, target)
                loss_value  = loss.item()
                valid_losses(loss_value)

                batch_iter.set_description(f'Validation: loss {loss_value:.4f}')

        self.valid_loss.append(np.mean(valid_losses))

        batch_iter.close()
