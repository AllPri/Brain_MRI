import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image

import torch

def show_image_and_mask(row: pd.Series) -> None:
    """
    Отображает исходное изображение и наложенную на него маску.
    Параметры:
    - row (pd.Series): Строка DataFrame, содержащая пути к изображению и маске.
    Действия:
    - Загружает и отображает исходное изображение.
    - Загружает и накладывает маску на изображение с прозрачностью.
    """
    # Загружаем изображение и маску
    image = Image.open(row['IMAGE'])
    mask = Image.open(row['MASK'])
    
    # Конвертируем маску в черно-белый формат (если необходимо)
    mask = mask.convert('L')
    
    # Создаем фигуру для отображения изображений
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Отображаем изображение
    ax[0].imshow(image)
    ax[0].set_title('Снимок с патологией')
    ax[0].axis('off')
    
    # Отображаем изображение с наложенной маской
    ax[1].imshow(image)
    ax[1].imshow(mask, alpha=0.5, cmap='jet')  # Наложение маски с прозрачностью
    ax[1].set_title('Снимок с маской')
    ax[1].axis('off')
    
    # Показываем результат
    plt.show()


# Функция для денормализации изображения
def denormalize(image, mean, std):
    image = image.cpu().clone()
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)  # Обратное нормирование
    return image

def visualize_predictions(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
                          device: torch.device, num_images: int) -> None:
    """
    Визуализирует предсказания модели для заданного количества изображений.
    Параметры:
    - model (torch.nn.Module): Модель PyTorch для генерации предсказаний.
    - data_loader (torch.utils.data.DataLoader): DataLoader с изображениями и масками.
    - device (torch.device): Устройство для выполнения вычислений (CPU или GPU).
    - num_images (int): Количество изображений для визуализации.
    Действия:
    - Показывает исходное изображение, истинную маску и предсказанную маску.
    """
    model.eval()  # Переводим модель в режим оценки

    # Средние и стандартные отклонения для нормализации (для ImageNet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    with torch.no_grad():
        for i, (images, masks, _) in enumerate(data_loader):
            if i >= num_images:
                break

            images = images.to(device)
            outputs = model(images)['out']
            preds = torch.sigmoid(outputs) > 0.5

            # Денормализуем изображение для отображения
            img = denormalize(images[0], mean, std).permute(1, 2, 0)
            img = img.clamp(0, 1)  # Обеспечиваем диапазон [0, 1]

            _, ax = plt.subplots(1, 3, figsize=(15, 5))

            # Исходное изображение
            ax[0].imshow(img)
            ax[0].set_title('Изображение')
            ax[0].axis('off')

            # Истинная маска
            ax[1].imshow(masks[0].cpu().squeeze(), cmap='gray')
            ax[1].set_title('Истинная маска')
            ax[1].axis('off')

            # Предсказанная маска
            ax[2].imshow(preds[0].cpu().squeeze(), cmap='gray')
            ax[2].set_title('Предсказание')
            ax[2].axis('off')

            plt.show()