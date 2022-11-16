import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def training(model, data, epochs=1, batch_size=64, learning_rate=0.01, loss_func=None):
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
    dataloader = DataLoader(data, batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # TODO: can try different optimizer

    # scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Start training
    for epoch in range(epochs):
        print("="*10, "Epoch", epoch, "="*10)
        
        total_loss = 0
        for idx, data in enumerate(dataloader):
            # get data
            images, labels = data
            # zero the gradients
            optimizer.zero_grad()
            # forward
            outputs = model(images)
            labels = torch.flatten(labels)
            loss = loss_func(outputs, labels)
            # backward
            loss.backward()
            # optimize
            optimizer.step()
            # statistics
            total_loss += loss.data
            print(f"loss: {loss}, avg_loss: {total_loss/(idx+1)}")


        scheduler.step(total_loss)
