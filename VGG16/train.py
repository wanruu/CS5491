from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train(model, data, epochs=300, batch_size=64, learning_rate=0.01, loss_func=None, optimizer=None):
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
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load data into batches
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    # scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Start training
    for epoch in range(epochs):
        print("=" * 10, "Epoch", epoch, "=" * 10)
        total_loss = 0
        for data, label in tqdm(dataloader):
            # zero the gradients
            optimizer.zero_grad()
            # forward
            outputs = model(data)
            label = torch.flatten(label) - 1
            loss = loss_func(outputs, label)
            # backward
            loss.backward()
            # optimize
            optimizer.step()
            # statistics
            total_loss += loss.data
        print(f"loss: {total_loss}")
        scheduler.step(total_loss)

    torch.save(model, "vgg16.pkl")

