from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator
from utils import preprocess_data


class QADataset:
    """
    Класс для работы с набором данных вопрос-ответ (QA), который включает в себя:
    - Загрузку данных
    - Предобработку данных
    - Создание DataLoader'ов для обучения и валидации
    """

    def __init__(self, dataset_name, model_name, batch_size=16):
        """
        Инициализирует объект QADataset, загружает и предобрабатывает данные, создает даталоудеры.

        Аргументы:
        dataset_name (str):
            Название набора данных, который будет загружен.
        model_name (str):
            Название предобученной модели для токенизации данных.
        batch_size (int):
            Размер батча для даталоудеров (по умолчанию 16).
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Инициализация токенизатора для выбранной модели
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Объявление переменных для хранения данных
        self.dataset = None
        self.dataset_train = None
        self.dataset_val = None
        self.dataloader_train = None
        self.dataloader_val = None

        # Вызов функции для загрузки и обработки данных и создания даталоудеров
        self.__load_and_preprocess_data()
        self.__create_dataloaders()

    def __load_and_preprocess_data(self):
        """
        Приватный метод для загрузки и предобработки данных.

        Загружает набор данных по названию датасета и применяет предобработку
        к тренировочным и валидационным данным.
        """
        # Загрузка датасета
        self.dataset = load_dataset(self.dataset_name)

        # Предобработка тренировочного набора
        self.dataset_train = self.dataset['train'].map(
            preprocess_data,
            batched=True,
            remove_columns=self.dataset['train'].column_names,
            fn_kwargs={'tokenizer': self.tokenizer},
        )

        # Предобработка валидационного набора
        self.dataset_val = self.dataset['validation'].map(
            preprocess_data,
            batched=True,
            remove_columns=self.dataset['validation'].column_names,
            fn_kwargs={'tokenizer': self.tokenizer, 'is_test': True},
        )

    def __create_dataloaders(self):
        """
        Приватный метод для создания даталоудеров для тренировочных и валидационных данных.
        
        Устанавливает формат данных как "torch" для удобного использования с PyTorch.
        Создает даталоудеры для тренировочного и валидационного наборов.
        """
        
        self.dataset_train.set_format("torch")
        # Форматирование валидационного набора
        dataset_val_formatted = self.dataset_val.remove_columns(["example_id", "offset_mapping"])
        dataset_val_formatted.set_format("torch")
        
        # Создание даталоудера для тренировочного набора
        self.dataloader_train = DataLoader(
            self.dataset_train,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=self.batch_size
        )

        # Создание даталоудера для валидационного набора
        self.dataloader_val = DataLoader(
            dataset_val_formatted,
            collate_fn=default_data_collator,
            batch_size=self.batch_size
        )
    
    def get_raw_dataset(self):
        """
        Возвращает исходный загруженный набор данных (содержит тренировочный и валидационный наборы).
        """
        return self.dataset

    def get_train_dataset(self):
        """
        Возвращает предобработанный тренировочный набор данных.
        """
        return self.dataset_train
    
    def get_val_dataset(self):
        """
        Возвращает предобработанный валидационный набор данных.
        """
        return self.dataset_val

    def get_train_dataloader(self):
        """
        Возвращает DataLoader для тренировочного набора данных.
        """
        return self.dataloader_train

    def get_val_dataloader(self):
        """
        Возвращает DataLoader для валидационного набора данных.
        """
        return self.dataloader_val
