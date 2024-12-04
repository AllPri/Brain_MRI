import torch

def train_torch(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, device: torch.device) -> float:
    """
    Обучает модель на одном проходе по тренировочному набору.
    Параметры:
    - model (torch.nn.Module): Модель PyTorch.
    - train_loader (torch.utils.data.DataLoader): DataLoader для тренировочного набора.
    - optimizer (torch.optim.Optimizer): Оптимизатор для обновления параметров модели.
    - criterion (torch.nn.Module): Функция потерь для вычисления ошибки.
    - device (torch.device): Устройство для выполнения вычислений (CPU или GPU).
    Возвращает:
    - float: Среднее значение функции потерь за эпоху.
    """
    running_loss = 0.0
    model.train()  # Устанавливаем модель в режим обучения

    for images, masks, _ in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)['out']  # Получаем предсказания модели
        loss = criterion(outputs, masks)  # Вычисляем функцию потерь
        loss.backward()  # Рассчитываем градиенты
        optimizer.step()  # Обновляем параметры модели
        running_loss += loss.item()  # Накопление потерь

    return running_loss / len(train_loader)


def evaluate_torch(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
                   criterion: torch.nn.Module, device: torch.device) -> float:
    """
    Оценивает модель на заданном наборе данных.
    Параметры:
    - model (torch.nn.Module): Модель PyTorch.
    - data_loader (torch.utils.data.DataLoader): DataLoader для тестового или валидационного набора.
    - criterion (torch.nn.Module): Функция потерь для вычисления ошибки.
    - device (torch.device): Устройство для выполнения вычислений (CPU или GPU).
    Возвращает:
    - float: Среднее значение функции потерь на наборе данных.
    """
    model.eval()  # Устанавливаем модель в режим оценки
    running_loss = 0.0

    with torch.no_grad():  # Отключаем вычисление градиентов
        for images, masks, _ in data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']  # Получаем предсказания модели
            loss = criterion(outputs, masks)  # Вычисляем функцию потерь
            running_loss += loss.item()  # Накопление потерь

    return running_loss / len(data_loader)


def calculate_iou_torch(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
                        device: torch.device) -> float:
    """
    Рассчитывает коэффициент пересечения и объединения (IoU) на заданном наборе данных.
    Параметры:
    - model (torch.nn.Module): Модель PyTorch.
    - data_loader (torch.utils.data.DataLoader): DataLoader для тестового или валидационного набора.
    - device (torch.device): Устройство для выполнения вычислений (CPU или GPU).
    Возвращает:
    - float: Значение IoU (пересечение/объединение).
    """
    model.eval()  # Устанавливаем модель в режим оценки
    intersection, union = 0, 0

    with torch.no_grad():  # Отключаем вычисление градиентов
        for images, masks, _ in data_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)['out']  # Получаем предсказания модели
            predictions = (torch.sigmoid(outputs) > 0.5).int()  # Бинаризация предсказаний
            masks = masks.int()  # Преобразуем маски в целые числа

            intersection += (predictions & masks).sum().item()  # Сумма пересечений
            union += (predictions | masks).sum().item()  # Сумма объединений

    return intersection / union if union != 0 else 0.0