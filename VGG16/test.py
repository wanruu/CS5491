import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


def test(model, dataset, batch_size=64, use_gpu=False):
    """
    params model_path: "xxx.pkl"
    params dataset   : type of MyDataset
    params batch_size: int
    params use_gpu   : whether use gpu
    """
    if use_gpu:
        model = model.cuda()
    model.eval()

    # Load data into batches
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12)
    
    # Start predicting
    top1_correct, top5_correct, total = 0, 0, 0
    for data, labels in tqdm(dataloader):
        # Switch to GPU to accelerate
        if(use_gpu):
            data = data.cuda()
            labels = labels.cuda()
        
        # Predict
        outputs = model(data)

        # Real label
        labels = torch.flatten(labels) - 1
        
        # Metric 1: top 1 accuracy
        _, top1 = torch.max(outputs.data, 1)
        top1_correct += (top1 == labels).sum()

        # Metric 2: top 5 accuracy
        _, top5 = torch.topk(outputs, 5, dim=1)
        top5_correct += sum([int(labels[idx] in top5[idx]) for idx in range(top5.shape[0])])

        # Count
        total += labels.size(0)

    print(f"Top 1: {top1_correct/total*100}%, Top 5: {top5_correct/total*100}%")


# For testing only.
if __name__ == "__main__":
    from data import MyDataset
    from model import VGG16
    from config import *

    print("Initializing dataset...")
    dataset = MyDataset(train=False, img_shape=(RESIZE, RESIZE), path=DATA_PATH)
    
    print("Loading model...")
    # path = "checkpoint/VGG16.pt"
    # model = VGG16(CLASS_NUM, CONV, FC, DROPOUT)
    # model.load_state_dict(torch.load(path))
    # test(model, dataset, batch_size=BATCH_SIZE, use_gpu=GPU)
    
    for i in range(1,10,5):
        print(f"============= {i} =============")
        path = f"VGG16-epoch={i}.pt"
        model = VGG16(CLASS_NUM, CONV, FC, DROPOUT)
        model.load_state_dict(torch.load(path))
        test(model, dataset, batch_size=BATCH_SIZE, use_gpu=GPU)