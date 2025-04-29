import torch
from sklearn.metrics import roc_auc_score

def evaluate(model, dataloader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            p = model(x).cpu().numpy().ravel()
            preds.extend(p)
            labels.extend(y.cpu().numpy().ravel())
    auc = roc_auc_score(labels, preds)
    print(f"AUC: {auc:.4f}")
