import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


tfm = transforms.Compose([transforms.ToTensor()])
trainset = datasets.FashionMNIST(root=".", train=True, download=True, transform=tfm)
testset  = datasets.FashionMNIST(root=".", train=False, download=True, transform=tfm)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader  = DataLoader(testset, batch_size=256)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2), # 14x14 -> 7x7
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7,128), nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self,x):
        return self.fc(self.body(x))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Net().to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()


for epoch in range(5):
    model.train()
    pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}")
    for x,y in pbar:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    
    model.eval(); correct=total=0
    with torch.no_grad():
        for x,y in testloader:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    acc = correct/total
    print(f"Test accuracy after epoch {epoch+1}: {acc:.3f}")


torch.save(model.state_dict(), "model.pt")
print("Saved model.pt")
