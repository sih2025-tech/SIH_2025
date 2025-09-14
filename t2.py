import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

# ----- Configuration -----
main_folder = 'C:/Users/Rukmini/Desktop/PlantDiseaseDataset'  # Replace with your path
batch_size = 32
epochs = 5
val_split = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Data Transforms and Loading -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(main_folder, transform=transform)
num_classes = len(dataset.classes)
val_size = int(val_split * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ----- Model Setup -----
# EfficientNet
efficientnet = models.efficientnet_b0(pretrained=True)
efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, num_classes)
efficientnet.to(device)
# ResNet18
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
resnet18.to(device)
models_list = [efficientnet, resnet18]

# ----- Training Function -----
def train_model(model, train_loader, val_loader, epochs=epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = 100 * correct / total
        print(f"{model.__class__.__name__} - Epoch {epoch+1}, Validation Accuracy: {acc:.2f}%")
    return model

# ----- Train Models -----
trained_models = []
for m in models_list:
    print(f"Training {m.__class__.__name__}")
    trained_model = train_model(m, train_loader, val_loader, epochs=epochs)
    trained_models.append(trained_model)

# ----- Ensemble Prediction -----
def ensemble_predict(models, images):
    outputs = [torch.softmax(m(images), dim=1) for m in models]
    avg_output = torch.stack(outputs).mean(dim=0)
    _, pred = torch.max(avg_output, 1)
    return pred

correct = 0
total = 0
for images, labels in val_loader:
    images, labels = images.to(device), labels.to(device)
    preds = ensemble_predict(trained_models, images)
    correct += (preds == labels).sum().item()
    total += labels.size(0)
print(f"Ensemble Validation Accuracy: {100 * correct / total:.2f}%")
