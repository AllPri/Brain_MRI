import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import torch
from torchvision import transforms

def get_pathology_label(mask_path):
    """
    Получает метку патологии на основе маски.
    Параметры:
    - mask_path (str): Путь к изображению маски.
    Возвращает:
    - int: 1, если маска содержит патологию, иначе 0.
    """
    mask = cv2.imread(mask_path)
    return 1 if np.max(mask) > 0 else 0

def create_datasets(path, test_size, val_size):
    """
    Создает тренировочный, тестовый и валидационный датасеты на основе пути к данным.
    Параметры:
    - path (str): Путь к папке с данными.
    - test_size (float): Доля тестового набора.
    - val_size (float): Доля валидационного набора.
    Возвращает:
    - tuple: DataFrame тренировочного, тестового и валидационного наборов.
    """
    # Получение путей к маскам и изображениям
    mask_files = glob.glob(path + '*/*_mask*')
    image_files = [file.replace('_mask', '') for file in mask_files]

    # Создание DataFrame с метками
    files_df = pd.DataFrame({
        "IMAGE": image_files,
        "MASK": mask_files,
        "PATHOLOGY_LABEL": [get_pathology_label(mask_path) for mask_path in mask_files]
    })

    # Разделение на тренировочный и временный набор (test + val)
    train_df, temp_df = train_test_split(
        files_df, 
        stratify=files_df['PATHOLOGY_LABEL'], 
        test_size=test_size + val_size, 
        random_state=37
    )

    # Вычисление пропорции для валидационного набора в оставшемся временном наборе
    val_ratio = val_size / (test_size + val_size)

    # Разделение временного набора на тестовый и валидационный
    test_df, val_df = train_test_split(
        temp_df, 
        stratify=temp_df['PATHOLOGY_LABEL'], 
        test_size=val_ratio, 
        random_state=37
    )

    # Печать размеров
    print(f'Всего записей: {len(files_df)}')
    print(f'Тренировочных: {len(train_df)}')
    print(f'Тестовых: {len(test_df)}')
    print(f'Валидационных: {len(val_df)}')

    # Возвращаем датасеты
    return train_df, test_df, val_df




def get_image_transforms():
    """
    Преобразовывает изображения для PyTorch.
    Возвращает:
    - transforms.Compose: Композиция преобразований для изображений.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_mask_transforms():
    """
    Преобразовывает маски для PyTorch.
    Возвращает:
    - transforms.Compose: Композиция преобразований для масок.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return transform

class CustomDataset(torch.utils.data.Dataset):
    """
    Кастомный датасет для загрузки изображений и масок.
    Атрибуты:
    - df (pd.DataFrame): DataFrame с путями к изображениям и маскам.
    - image_transform (callable, optional): Преобразования для изображений.
    - mask_transform (callable, optional): Преобразования для масок.
    """
    def __init__(self, df, image_transform=None, mask_transform=None):
        self.df = df
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Получает элемент по индексу.
        Параметры:
        - idx (int): Индекс элемента.
        Возвращает:
        - tuple: Кортеж (изображение, маска, метка патологии).
        """
        image_path = self.df.iloc[idx]["IMAGE"]
        mask_path = self.df.iloc[idx]["MASK"]
        
        # Загрузка изображений и масок
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Применение преобразований
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        pathology_label = self.df.iloc[idx]["PATHOLOGY_LABEL"]
        return image, mask, pathology_label
