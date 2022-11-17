import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
from configuration import *
from evaluate import evalMatrix


def train(model, train_data, val_data, epochs=3, batch_size=64, learning_rate=0.01, device='cpu', loss_func=None,
          optimizer=None, train_evaluator: evalMatrix = None, val_evaluator: evalMatrix = None, log=None, num_workers=2,
          save_model='best_model.pt'):
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
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                  num_workers=num_workers)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=num_workers)

    # Optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    model.to(device)

    # Start training
    best_acc = -1
    for epoch in range(epochs):
        print("=" * 10, "Epoch", epoch, "=" * 10)
        total_loss = 0
        train_evaluator.clear()
        val_evaluator.clear()
        model.train()
        for data, cls in tqdm.tqdm(train_dataloader):
            # for data, cls in dataloader:
            # zero the gradients
            data = data.to(device)
            cls = cls.to(device)
            optimizer.zero_grad()
            # forward
            outputs = model(data)
            cls = torch.flatten(cls)
            loss = loss_func(outputs, cls)
            if train_evaluator is not None:
                train_evaluator.record(outputs, cls)
            # backward
            loss.backward()
            # optimize
            optimizer.step()
            # statistics
            total_loss += loss.data.item()

        print('avg_loss:', total_loss / (len(train_dataloader)))

        if train_evaluator is not None and log is not None:
            log.record(epoch=epoch, evaluator=train_evaluator, loss=float(total_loss) / len(train_dataloader),
                       state='train',
                       auto_write=True)

        _validation(model, criterion=loss_func, loader=val_dataloader, evaluater=val_evaluator, device=device)

        if val_dataloader is not None and log is not None:
            log.record(epoch=epoch, evaluator=val_evaluator, loss=0, state='test', auto_write=True)
            if best_acc < val_evaluator.acc:
                best_acc = val_evaluator.acc
                torch.save(model, SaveModel + save_model)

        scheduler.step(total_loss)


def _validation(model, criterion, loader, device, evaluater=None):
    evaluater.clear()
    model.eval()
    epoch_loss_ = 0
    with torch.no_grad():
        for src, cls in tqdm.tqdm(loader, desc='test '):
            src = src.to(device)
            cls = cls.to(device)
            output = model(src)
            cls = torch.flatten(cls)
            evaluater.record(output, cls)
            loss = criterion(output, cls)
            epoch_loss_ += loss.item()
    return epoch_loss_ / len(loader)
