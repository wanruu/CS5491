import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


def test(model, dataset, batch_size=64, use_gpu=False):
    """
    params model     : 
    params dataset   : type of MyDataset
    params batch_size: int
    params use_gpu   : whether use gpu
    """
    # Prepare.
    model_name = model.name if model.name else "Unnamed"
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
        if "DFL" in model_name:
            g, p, side = model(data)
            outputs = g + p + 0.1 * side
        else:
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
    from model.vgg16 import VGG16
    from model.dfl_vgg16 import DFL_VGG16
    from model.dfl_vgg16_pre import DFL_VGG16_Pre
    from config import *


    dataset = MyDataset(train=False, path=DATA_PATH, masked=False, transform=TEST_TRANS)
    
    # path = "checkpoint/2022-11-19 14:45:09.608551/DFL_VGG16.pt"
    # path = "checkpoint/DFL_VGG16(aug+pretrained).pt"
    # vgg16 = VGG16(CLASS_NUM, CONV, FC, DROPOUT)
    # model = DFL_VGG16(class_num=CLASS_NUM, k=K, vgg=vgg16)
    # model = DFL_VGG16_Pre(class_num=CLASS_NUM, k=K)
    # model.load_state_dict(torch.load(path))
    # test(model, dataset, batch_size=BATCH_SIZE, use_gpu=GPU)

    for i in range(0,500,10):
        print(f"============= {i} =============")
        path = f"checkpoint/2022-11-19 19:29:58.659012/VGG16-epoch={i}.pt"
        model = VGG16(CLASS_NUM, CONV, FC, DROPOUT)
        # model = DFL_VGG16(class_num=CLASS_NUM, k=K, vgg=vgg16)
        # model = DFL_VGG16_Pre(class_num=CLASS_NUM, k=K)
        model.load_state_dict(torch.load(path))
        test(model, dataset, batch_size=BATCH_SIZE, use_gpu=GPU)



    # temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'\nThe model has {temp:,} trainable parameters')



