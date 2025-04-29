import torch
from torch.utils.data import DataLoader, Dataset
from models import Simple2DCNN

class DeforestDataset(Dataset):
    def __init__(self, tiles_dir):
        # TODO: load list of tile file paths
        pass

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        # TODO: return (input_tensor, label_tensor)
        pass

def train_loop(config):
    dataset = DeforestDataset(config['tiles_dir'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    model = Simple2DCNN(in_channels=config['in_channels'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.BCELoss()

    for epoch in range(config['epochs']):
        for x, y in loader:
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

if __name__ == '__main__':
    import yaml
    cfg = yaml.safe_load(open('config.yaml'))
    train_loop(cfg)
