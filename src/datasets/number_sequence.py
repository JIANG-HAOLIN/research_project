import torch.utils.data as data
import torch
from functools import partial
class ReverseDataset(data.Dataset):

    def __init__(self, num_categories: int = 10, seq_len: int = 17, size: int = 1000):
        """

        Args:
            num_categories: number of different digits
            seq_len: number of digits in a sequence
            size: how many sequence in a dataset ?
        """
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len
        self.size = size

        self.data = torch.randint(self.num_categories, size=(self.size, self.seq_len))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp_data = self.data[idx]
        labels = torch.flip(inp_data, dims=(0,))
        return inp_data, labels



def get_loaders(num_categories: int = 10,
                seq_len: int = 17,
                train_size: int = 50000,
                val_size: int = 1000,
                test_size: int = 10000,
                batch_size: int = 128,
                **kwargs):
    train_loader = data.DataLoader(ReverseDataset(num_categories, seq_len, train_size), batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader   = data.DataLoader(ReverseDataset(num_categories, seq_len, val_size), batch_size=batch_size)
    test_loader  = data.DataLoader(ReverseDataset(num_categories, seq_len, test_size), batch_size=batch_size)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    dataset = ReverseDataset()
    train_loader = data.DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
    inp_data, labels = train_loader.dataset[0]
    print("Input data:", inp_data)
    print("Labels:    ", labels)

