import tqdm
import torch

from nice import Quantizer

from typing import Union
from torch.utils.data import DataLoader

class Trainer():

    def __init__(self,
                 model: torch.nn.parallel.DistributedDataParallel,
                 dataloaders: list[DataLoader],
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 gpu_id: int):
        self.model = model.module
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dataloader = dataloaders[0]
        self.val_dataloader = dataloaders[1]

        self.device = gpu_id

    def train(self):
        train_loss = torch.zeros(1).to(self.device)
        
        self.model.train()
        for batch in self.train_dataloader:
            x = batch[0].to(self.device)
            x = Quantizer.dequantize(x)
            
            self.optimizer.zero_grad(set_to_none=True)

            z = self.model.forward(x)
            loss = self.model.loss(z)
            train_loss += loss
                
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

        return train_loss / len(self.train_dataloader)
    
    def test(self, dataloader: DataLoader):
        test_loss = torch.zeros(1).to(self.device)

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device)
                x = Quantizer.dequantize(x)

                z = self.model.forward(x)

                test_loss += self.model.loss(z)

        return test_loss / len(dataloader)
    
    def run(self, epochs: int = 1500,
            warmup_epochs: int = 20,
            param_group_key: str = "momentum",
            post_warmup_value: Union[tuple, float] = 0.5,
            save_freq: int = 10):
        train_losses = torch.tensor([], device=self.device)
        val_losses = torch.tensor([], device=self.device)

        pbar = tqdm.tqdm(range(1, epochs+1))
        for epoch in pbar:
            if epoch == warmup_epochs:
                for param_group in self.optimizer.param_groups:
                    param_group[param_group_key] = post_warmup_value

            avg_train_loss = self.train()
            train_losses = torch.cat((train_losses, avg_train_loss))

            avg_val_loss = self.test(self.val_dataloader)
            val_losses = torch.cat((val_losses, avg_val_loss))

            loss_curves = torch.stack((train_losses, val_losses))
            torch.save(loss_curves, "loss_curves.pt")

            epoch_train_loss = round(avg_train_loss.item(), 2)
            epoch_val_loss = round(avg_val_loss.item(), 2)

            pbar.set_postfix({
                "Epoch": epoch,
                "Train Loss": epoch_train_loss,
                "Validation Loss": epoch_val_loss
            })

            if self.device == 0 and epoch % save_freq == 0:
                train_loss_log = f"train_loss_{epoch_train_loss}"
                val_loss_log = f"val_loss_{epoch_val_loss}"
                
                torch.save(self.model.state_dict(),
                        f"nice_epoch_{epoch}_{train_loss_log}_{val_loss_log}.pt")