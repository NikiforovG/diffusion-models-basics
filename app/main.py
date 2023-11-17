import torch

from src.model import ContextUnet


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    print(device)

    n_feat = 64  # 64 hidden dimension feature
    n_cfeat = 5  # context vector is of size 5
    height = 16  # 16x16 image
    model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
