import torch


def get_dataloaders(train_data=None, test_data=None, val_data=None, batch_size=32):
    train_loader = test_loader = val_loader = None
    if train_data:
        train_loader = torch.utils.data.DataLoader(
            train_data, shuffle=True, batch_size=batch_size
        )
    if test_data:
        test_loader = torch.utils.data.DataLoader(
            test_data, shuffle=False, batch_size=batch_size
        )
    if val_data:
        val_loader = torch.utils.data.DataLoader(
            val_data, shuffle=False, batch_size=batch_size
        )
    return train_loader, test_loader, val_loader
