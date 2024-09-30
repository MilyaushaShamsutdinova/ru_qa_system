import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AdamW, get_scheduler
from accelerate import Accelerator
from utils import compute_metrics
from dataset import QADataset


class QATrainer:
    """
    Класс для обучения модели вопрос-ответ (QA model). 
    Отвечает за обучение модели и оценку модели на валидационных данных.
    """

    def __init__(self, model_name, dataset: QADataset, model_save_dir, num_epochs=3, lr=3e-5):
        """
        Инициализация класса.

        Аргументы:
        model_name (str):
            Название модели для загрузки через Hugging Face.
        dataset (QADataset):
            Экземпляр датасета с тренировочными и валидационными данными.
        model_save_dir (str):
            Путь для сохранения обученной модели.
        num_epochs (int):
            Количество эпох для обучения (по умолчанию 3).
        lr (float):
            Темп обучения для оптимизатора (по умолчанию 3e-5).
        """
        self.model_name = model_name
        self.model_save_dir = model_save_dir

        # Выделение даталоудеров и датасетов для удобства нахождения нужной информации во время тренировки модели
        self.dataloader_train = dataset.get_train_dataloader()
        self.dataloader_val = dataset.get_val_dataloader()
        self.dataset_train = dataset.get_train_dataset()
        self.dataset_val = dataset.get_val_dataset()
        self.dataset = dataset.get_raw_dataset()

        # Загрузка токенизатора и модели с помощью Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)

        # Настройка оптимизатора (используем AdamW) и планировщика обучения и параметром обучения
        self.num_epochs = num_epochs
        self.lr = lr
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = self.__create_scheduler()

        # Подготовка модели, оптимизатора и даталоадеров
        self.accelerator = Accelerator(mixed_precision="fp16")
        self.model, self.optimizer, self.dataloader_train, self.dataloader_val = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader_train, self.dataloader_val
        )

    def __create_scheduler(self):
        """
        Создает планировщик для контроля скорости обучения на основе линейной убывающей стратегии.
        Возвращает объект планировщика. Приватный метод.
        """
        return get_scheduler(
            'linear',
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.dataloader_train)*self.num_epochs,
        )

    def train(self):
        """
        Основной метод для обучения модели. Обучает модель на тренировочном датасете
        и оценивает ее на валидационном после каждой эпохи. Сохраняет модель после обучения.
        """
        progress_bar = tqdm(range(len(self.dataloader_train)*self.num_epochs))

        for epoch in range(self.num_epochs):
            # Тренировка модели
            self.model.train()
            for _, batch in enumerate(self.dataloader_train):
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)

            # Оценка на валидационных данных
            self.evaluate(epoch)

        # Сохранение модели
        self.save_final_model()

    def evaluate(self, epoch):
        """
        Оценивает модель на валидационном наборе данных.

        Аргументы:
        epoch (int):
            Номер текущей эпохи.
        """
        # Перевод модели в режим оценки
        self.model.eval()
        start_logits = []
        end_logits = []

        for batch in tqdm(self.dataloader_val, desc='Оценка модели'):
            # Прямой проход модели
            with torch.no_grad():
                outputs = self.model(**batch)

            # Собираем start и end логиты для каждого батча
            start_logits.append(self.accelerator.gather(outputs.start_logits).cpu().numpy())
            end_logits.append(self.accelerator.gather(outputs.end_logits).cpu().numpy())

        # Преобразуем список логитов в один массив и обрезаем этот массив до размера валидационного набора
        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(self.dataset_val)]
        end_logits = end_logits[: len(self.dataset_val)]

        # Вычисление метрик
        metrics = compute_metrics(start_logits, end_logits, self.dataset_val, self.dataset["validation"])
        print(f"Эпоха {epoch + 1}:", metrics)

        # Сохранение модели
        self.accelerator.wait_for_everyone()  # Ожидание завершения всех операций
        unwrapped_model = self.accelerator.unwrap_model(self.model)  # Извлечение модели из оболочки accelerator
        unwrapped_model.save_pretrained(self.model_save_dir, save_function=self.accelerator.save)
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(self.model_save_dir)

    def save_final_model(self):
        """
        Сохраняет финальную версию модели и токенизатора в директорию для сохранения.
        """
        self.tokenizer.save_pretrained(self.model_save_dir)
        self.model.save_pretrained(self.model_save_dir)


if __name__ == "__main__":
    """
    Код для тренировки лучшей модели из рассматриваемых - RuBert на 1 эпохе с темпом обучения равнм 3e-5.
    """
    dataset_name = "kuznetsoffandrey/sberquad"
    model_name = "DeepPavlov/rubert-base-cased"
    model_save_dir = "trained_models/rubert-v3"

    dataset = QADataset(dataset_name=dataset_name, model_name=model_name)
    trainer = QATrainer(model_name=model_name, dataset=dataset, model_save_dir=model_save_dir, num_epochs=1)
    
    trainer.train()
