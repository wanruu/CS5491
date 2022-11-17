import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


def test(model_path, dataset, batch_size=64, use_gpu=False):
    """
    params model_path: "xxx.pkl"
    params dataset   : type of MyDataset
    params batch_size: int
    params use_gpu   : whether use gpu
    """
    # Load data into batches
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12)
    
    # Load model
    if use_gpu:
        model = torch.load(model_path, map_location=torch.device("cuda"))
    else:
        model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()
    
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
    import argparse
    from data import MyDataset
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="checkpoint/VGG16.pkl")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    # Get data
    print("Initializing dataset...")
    dataset = MyDataset(train=True, img_shape=(192, 192), path="../../CUB_200_2011/CUB_200_2011/")
    
    # Testing
    # test(args.path, dataset, batch_size=args.batch_size, use_gpu=torch.cuda.is_available())
    
    for i in range(1,50,5):
        print(f"============= {i} =============")
        path = f"checkpoint/aug_attempt1/VGG16-epoch={i}.pkl"
        test(path, dataset, batch_size=32, use_gpu=True)