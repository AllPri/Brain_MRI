import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

def train_and_evaluate(model: torch.nn.Module, model_name: str, num_epochs: int, 
                       train_loader: torch.utils.data.DataLoader, 
                       test_loader: torch.utils.data.DataLoader, 
                       val_loader: torch.utils.data.DataLoader, 
                       optimizer: torch.optim.Optimizer, 
                       criterion: torch.nn.Module, device: torch.device) -> None:
    """
    Обучает модель и оценивает ее на тестовых и валидационных данных, рассчитывая метрику IoU.

    Параметры:
    - model (torch.nn.Module): Модель PyTorch.
    - model_name (str): Имя модели для сохранения.
    - num_epochs (int): Количество эпох обучения.
    - train_loader, test_loader, val_loader: DataLoader'ы для соответствующих данных.
    - optimizer (torch.optim.Optimizer): Оптимизатор.
    - criterion (torch.nn.Module): Функция потерь.
    - device (torch.device): Устройство для вычислений (CPU или GPU).
    """
    # Метрики для отслеживания
    train_losses, test_losses, val_losses, val_accuracies = [], [], [], []
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        # Обучение модели
        model.train()
        running_loss = 0.0
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            # меняем структуру предсказаний из модели
            if isinstance(outputs, dict):
                outputs = outputs['out']  # Для DeepLabV3, для другихх ничего не меняется

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Оценка модели
        model.eval()
        test_loss, val_loss = 0.0, 0.0
        intersection, union = 0, 0

        with torch.no_grad():
            for loader, losses in zip([test_loader, val_loader], [test_losses, val_losses]):
                running_loss = 0.0
                for images, masks, _ in loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)

                    # меняем структуру предсказаний из модели
                    if isinstance(outputs, dict):
                        outputs = outputs['out']  # Для DeepLabV3, для другихх ничего не меняется

                    loss = criterion(outputs, masks)
                    running_loss += loss.item()
                    if loader == val_loader:
                        predictions = (torch.sigmoid(outputs) > 0.5).int()
                        masks = masks.int()
                        intersection += (predictions & masks).sum().item()
                        union += (predictions | masks).sum().item()
                losses.append(running_loss / len(loader))

        val_IoU = intersection / union if union != 0 else 0.0
        val_accuracies.append(val_IoU)
        print(f'Эпоха {epoch + 1} из {num_epochs}: train_loss={round(train_losses[-1], 5)}, test_loss={round(test_losses[-1], 5)}, val_loss={round(val_losses[-1], 5)}, val_IoU={round(val_IoU, 5)},')

        # Сохранение лучшей модели
        if val_IoU > best_val_accuracy:
            best_val_accuracy = val_IoU
            torch.save(model.state_dict(), f'models/{model_name}_best_model.pth')

    print(f'Максимальная значение IoU для {model_name} = {best_val_accuracy}')
    # Графики
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 6))

    # Потери
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, test_losses, label='Test Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # IoU
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, label='Val IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.title('Validation IoU')
    plt.legend()

    plt.tight_layout()
    plt.show()
