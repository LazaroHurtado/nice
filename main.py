import torch
import torch.utils
import torchvision
import torch.distributed as dist

from nice import Nice
from trainer import Trainer

from torchvision.transforms import v2 as T
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

DATALOADER_ARGS = {
    "batch_size": 1024,
    "num_workers": 4,
    "pin_memory": True,
    }

def build_dataloader(dataset: Dataset):
    return DataLoader(dataset,
                      sampler=DistributedSampler(dataset),
                      **DATALOADER_ARGS)

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    transform = torchvision.transforms.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale = True),
        T.Lambda(lambda x : x.reshape(-1)),
    ])

    test_dataset = torchvision.datasets.MNIST(
            root="./data",
            download=True,
            transform=transform
        )
    train_dataset = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )
    dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    train_dataset, val_dataset = random_split(dataset, [0.85, 0.15])

    train_dataloader = build_dataloader(train_dataset)
    val_dataloader = build_dataloader(val_dataset)

    model = Nice(784, device=rank).to(rank)
    model = DDP(model)

    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=1e-3,
                                    momentum=1e-8,
                                    weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.995)

    trainer = Trainer(model,
                      [train_dataloader, val_dataloader],
                      optimizer,
                      scheduler,
                      rank)
    trainer.run()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
