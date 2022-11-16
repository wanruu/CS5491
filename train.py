import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
from configuration import *
from evaluate import evalMatrix


def train(model, data, epochs=3, batch_size=64, learning_rate=0.01, device='cpu', loss_func=None, optimizer=None,
          evaluator: evalMatrix = None, log=None, save_model='best_model.pt'):
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

    model.to(device)

    # Start training
    best_loss = np.inf
    for epoch in range(epochs):
        print("=" * 10, "Epoch", epoch, "=" * 10)
        total_loss = 0
        evaluator.clear()
        for data, cls in tqdm.tqdm(dataloader):
            # for data, cls in dataloader:
            # zero the gradients
            data = data.to(device)
            cls = cls.to(device)
            optimizer.zero_grad()
            # forward
            outputs = model(data)
            cls = torch.flatten(cls)
            print(outputs.shape, cls.shape)
            loss = loss_func(outputs, cls)
            if evaluator is not None:
                evaluator.record(outputs, cls)
            # backward
            loss.backward()
            # optimize
            optimizer.step()
            # statistics
            total_loss += loss.data
            print(f"loss: {loss}, avg_loss: {total_loss} / {epoch}")

        if evaluator is not None and log is not None:
            log.record(epoch=epoch, evaluator=evaluator, loss=total_loss/len(dataloader), state='train', auto_write=True)

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model, SaveModel + save_model)

        scheduler.step(total_loss)
