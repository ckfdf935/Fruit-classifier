import torch.nn as nn
import torchvision as tv
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

training = r"fruits-360_100x100\fruits-360\Training"
val = r"fruits-360_100x100\fruits-360\Test"

trainingTransforms = tv.transforms.Compose([
    tv.transforms.Resize((100, 100)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

valTransforms = tv.transforms.Compose([
    tv.transforms.Resize((100, 100)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


trainingDataset = tv.datasets.ImageFolder(training, transform=trainingTransforms)
valDataset = tv.datasets.ImageFolder(val, transform=valTransforms)

def ModelTraning(model, dataloader, criterion, optimizer, device):
    model.train() # Водим в режим обучения

    rloss = 0.0
    correct = total = 0

    loop = tqdm(dataloader, desc="Обучение", leave=False)

    for images, labels in loop: # цикл, который перебирает батчи

            # отправляем на GPU
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad() #Обнуляем градиенты

            with torch.cuda.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            rloss += loss.item()

            _, predicted = outputs.max(1)

            # Считаем accyracy
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)


    return rloss / len(dataloader), 100. * correct / total # Возвращаем средний loss (сумма / количество батчей) и итоговую accuracy



def validate(model, dataloader, criterion, device):
    model.eval()

    v_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Валидация", leave=False)

        for im, lab in loop:
            im = im.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)

            outputs = model(im)
            loss = criterion(outputs, lab)

            v_loss += loss.item()

            _, predicted = outputs.max(1)
            total += lab.size(0)
            correct += predicted.eq(lab).sum().item()

            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

    return v_loss / len(dataloader), 100. * correct / total


if __name__ == "__main__":
    trainingLoader  = DataLoader(
        trainingDataset,
        batch_size=64, # # по 64 картинки за раз
        shuffle=True, # перемешивает данные в разных эпохах
        num_workers=4, #  параллельных процесса загрузки данных

    )

    valLoader = DataLoader(valDataset, batch_size=64, shuffle=False, num_workers=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_classes = len(trainingDataset.classes)

    model = tv.models.resnet18(weights="IMAGENET1K_V1") # Готовая архитектура с обучеными весами с датасета ImageNet.
    model.fc = nn.Linear(model.fc.in_features, num_classes) # Замена последнего слоя и установка нужного количество класов
    model = model.to(device)
    model = torch.compile(model)# Компиляция модели

    criterion = nn.CrossEntropyLoss() # Функция потерь

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # Аптимезатор

    scaler = torch.cuda.amp.GradScaler()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_acc = 0.0
    patience = 3
    counter = 0


    for epoch in range(5):
        print(f"Эпоха [{epoch+1}]")

        train_loss, train_acc = ModelTraning(
            model, trainingLoader, criterion, optimizer, device
        )

        val_loss, val_acc = validate(model, valLoader, criterion, device)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Обучение: Потеря={train_loss:.4f}, ACC={train_acc:.2f}%")
        print(f"Валидация: Потеря={val_loss:.4f}, ACC={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            # Сохраняем веса именно лучшей эпохи в отдельный файл
            torch.save(model.state_dict(), "best_fruits_model.pth")
            print(f"!!! Найдена лучшая модель (Эпоха {epoch + 1}, Acc: {val_acc:.2f}%) !!!")


    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.legend()
    plt.title("Loss")
    plt.show()

    # Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_accuracies, label='Train Acc')
    plt.plot(epochs, val_accuracies, label='Val Acc')
    plt.legend()
    plt.title("Accuracy")
    plt.show()
