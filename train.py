import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader


def train(model, data, epochs=3, batch_size=64, learning_rate=0.01, device='cpu', loss_func=None, optimizer=None,
          evaluate=None,
          log=None):
    """
    model: nn model
    data: each item is a tuple (image, label)
    epochs: int
    batch_size: int
    learning_rate: float
    """

    # Prepare parameter
    if not loss_func:
        loss_func = nn.CrossEntropyLoss()

    # Split data into batches
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    # Optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Start training
    for epoch in range(epochs):
        print("=" * 10, "Epoch", epoch, "=" * 10)
        total_loss = 0
        # print(len(dataloader))
        for data, cls in tqdm.tqdm(dataloader):
            # for data, cls in dataloader:
            # zero the gradients
            optimizer.zero_grad()
            # forward
            outputs = model(data)
            cls = torch.flatten(cls)
            loss = loss_func(outputs, cls)
            # backward
            loss.backward()
            # optimize
            optimizer.step()
            # statistics
            total_loss += loss.data
            print(f"loss: {loss}, avg_loss: {total_loss} / {epoch}")

        scheduler.step(total_loss)
