import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader


class EarlyStopping:
    def __init__(self, patience=30, min_delta=0):
        """
        params patience : early stop only if epoches of no improvement >= patience.
        params min_delta: an absolute change of less than min_delta, will count as no improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.min_loss = float("inf")
        self.cnt = 0
        self.flag = False

    def __call__(self, loss):
        if (self.min_loss - loss) < self.min_delta:
            self.cnt += 1
        else:
            self.min_loss = loss
            self.cnt = 0
        if self.cnt >= self.patience:
            self.flag = True



def train(model, dataset, epochs=300, batch_size=64, learning_rate=0.01, loss_func=None, optimizer=None, early_stopping=None,
          use_gpu=False, save_path="/", save_intervals=50):
    """
    params model         : nn model
    params dataset       : type of MyDataset
    params epochs        : int
    params batch_size    : int
    params learning_rate : float
    params loss_func     : function to calculate loss
    params optimizer     : 
    params early_stopping: 
    params use_gpu       : True/False
    params save_path     : 
    params save_intervals: save model every _ epoches
    return: None
    """
    # Prepare
    loss_func = loss_func if loss_func else nn.CrossEntropyLoss()
    optimizer = optimizer if optimizer else torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = early_stopping if early_stopping else EarlyStopping(patience=30, min_delta=0)
    model_name = model.name if model.name else "Unnamed"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)  # to reduce learning rate
    if(use_gpu):
        model = model.cuda()
        loss_func = loss_func.cuda()
    model.train()

    # Load data into batches
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)    

    # Start training by epoch
    for epoch in range(epochs):
        print("=" * 10, "Epoch", epoch, "=" * 10)

        total_loss = 0
        top1_correct, total = 0, 0
        for data, labels in tqdm(dataloader):
            # Use GPU
            if(use_gpu):
                data = data.cuda()
                labels = labels.cuda()
            # Zero the gradients
            optimizer.zero_grad()
            # Forward
            outputs = model(data)
            labels = torch.flatten(labels) - 1  # [200]
            loss = loss_func(outputs, labels)
            # Backward
            loss.backward()
            # Optimize
            optimizer.step()
            # Statistics
            total_loss += loss.data
            _, top1 = torch.max(outputs.data, 1)
            top1_correct += (top1 == labels).sum()
            total += labels.size(0)
        
        print(f"loss: {total_loss}, accuracy: {top1_correct/total*100}%")
        scheduler.step(total_loss)

        # Early stopping
        early_stopping(total_loss)
        if early_stopping.flag:
            print(f"Early stop at epoch {epoch}.")
            break

        # Save model every {save_intervals} epoch
        if epoch % save_intervals == 1:
            print("Saving model...")
            torch.save(model.state_dict(), f"{save_path}/{model_name}-epoch={epoch}.pt")

    # Save final model
    print("Saving final model...")
    if use_gpu:
        model = model.cpu()
    torch.save(model.state_dict(), f"{save_path}/{model_name}.pt")

    return model




def train_dfl(model, dataset, epochs=300, batch_size=64, learning_rate=0.01, loss_func=None, optimizer=None, early_stopping=None,
              use_gpu=False, save_path="/", save_intervals=50):
    """
    params model         : nn model
    params dataset       : type of MyDataset
    params epochs        : int
    params batch_size    : int
    params learning_rate : float
    params loss_func     : function to calculate loss
    params optimizer     : 
    params early_stopping: 
    params use_gpu       : True/False
    params save_path     : 
    params save_intervals: save model every _ epoches
    return: None
    """
    # Prepare
    loss_func = loss_func if loss_func else nn.CrossEntropyLoss()
    optimizer = optimizer if optimizer else torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = early_stopping if early_stopping else EarlyStopping(patience=30, min_delta=0)
    model_name = model.name if model.name else "Unnamed"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)  # to reduce learning rate
    if(use_gpu):
        model = model.cuda()
        loss_func = loss_func.cuda()
    model.train()

    # Load data into batches
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)    

    # Start training by epoch
    for epoch in range(epochs):
        print("=" * 10, "Epoch", epoch, "=" * 10)

        total_loss = 0
        top1_correct, total = 0, 0
        for data, labels in tqdm(dataloader):
            # Use GPU
            if(use_gpu):
                data = data.cuda()
                labels = labels.cuda()
            # Zero the gradients
            optimizer.zero_grad()
            # Forward
            g, p, side = model(data)  # [32, 200]

            labels = torch.flatten(labels) - 1  # [200]
            # loss = loss_func(outputs, labels)
            loss1 = loss_func(g, labels)
            loss2 = loss_func(p, labels)
            loss3 = loss_func(side, labels)
            loss = loss1 + loss2 + 0.1 * loss3
            outputs = g + p + 0.1 * side
            
            # Backward
            loss.backward()
            # Optimize
            optimizer.step()
            # Statistics
            total_loss += loss.data
            
            _, top1 = torch.max(outputs.data, 1)
            top1_correct += (top1 == labels).sum()
            total += labels.size(0)
        
        print(f"loss: {total_loss}, accuracy: {top1_correct/total*100}%")
        scheduler.step(total_loss)

        # Early stopping
        early_stopping(total_loss)
        if early_stopping.flag:
            print(f"Early stop at epoch {epoch}.")
            break

        # Save model every {save_intervals} epoch
        if epoch % save_intervals == 1:
            print("Saving model...")
            torch.save(model.state_dict(), f"{save_path}/{model_name}-epoch={epoch}.pt")

    # Save final model
    print("Saving final model...")
    if use_gpu:
        model = model.cpu()
    torch.save(model.state_dict(), f"{save_path}/{model_name}.pt")

    return model
