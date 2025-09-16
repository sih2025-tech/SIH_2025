import os
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from torchvision.datasets import ImageFolder

# Allow loading truncated images partially corrupted but loadable
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Custom ImageFolder to skip corrupted images gracefully
class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        for _ in range(3):  # Try 3 times to get a valid image
            try:
                return super().__getitem__(index)
            except (IOError, OSError) as e:
                print(f"Warning: Corrupted image at index {index} skipped. Error: {e}")
                index = (index + 1) % len(self)
        raise RuntimeError(f"Cannot load image after multiple attempts starting from index {index}")

def remove_corrupted_images(dataset_dir):
    corrupted_files = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Initial check for corruption
            except (IOError, SyntaxError):
                corrupted_files.append(file_path)
                print(f"Corrupted image found and removed: {file_path}")
    for file_path in corrupted_files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")
    print(f"Total corrupted images removed: {len(corrupted_files)}")

def train_model(model, train_loader, val_loader, epochs=5):
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

def ensemble_predict(models, images):
    outputs = [torch.softmax(m(images), dim=1) for m in models]
    avg_output = torch.stack(outputs).mean(dim=0)
    _, pred = torch.max(avg_output, 1)
    return pred

def predict_image(image_path, models, transform, classes):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_idx = ensemble_predict(models, input_tensor)
    return classes[pred_idx.item()]

def load_trained_model(model_class, weight_path, num_classes):
    if model_class == 'efficientnet':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_class == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("Unsupported model class")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

if __name__ == '__main__':
    main_folder = r'C:/Users/Rukmini/Desktop/PlantDiseaseDataset'  # Your dataset path
    batch_size = 32
    epochs = 5
    val_split = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial removal of corrupted images
    remove_corrupted_images(main_folder)

    # Data transforms and loading with custom SafeImageFolder
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = SafeImageFolder(main_folder, transform=transform)
    num_classes = len(dataset.classes)
    classes = dataset.classes

    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    efficientnet_path = "EfficientNet_agro_model.pkl"
    resnet_path = "ResNet_agro_model.pkl"

    if os.path.exists(efficientnet_path) and os.path.exists(resnet_path):
        efficientnet = load_trained_model('efficientnet', efficientnet_path, num_classes)
        resnet18 = load_trained_model('resnet18', resnet_path, num_classes)
        trained_models = [efficientnet, resnet18]
        print("Loaded saved models, skipping training.")
    else:
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, num_classes)
        efficientnet.to(device)

        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
        resnet18.to(device)

        models_list = [efficientnet, resnet18]

        trained_models = []
        for m in models_list:
            print(f"Training {m.__class__.__name__}")
            trained_model = train_model(m, train_loader, val_loader, epochs=epochs)
            trained_models.append(trained_model)

        torch.save(trained_models[0].state_dict(), efficientnet_path)
        torch.save(trained_models[1].state_dict(), resnet_path)
        print("Training complete, models saved.")

    correct = 0
    total = 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        preds = ensemble_predict(trained_models, images)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    print(f"Ensemble Validation Accuracy: {100 * correct / total:.2f}%")

    test_image_path = r'C:/Users/Rukmini/Desktop/code/b1.jpg'  # Change as needed
    predicted_class = predict_image(test_image_path, trained_models, transform, classes)
    print(f"Prediction for image '{test_image_path}': {predicted_class}")
