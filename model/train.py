import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AdamW, get_scheduler
from accelerate import Accelerator
from utils import compute_metrics
from dataset import QADataset


class QATrainer:
    def __init__(self, model_name, dataset: QADataset, model_save_dir, num_epochs=3, lr=3e-5):
        self.model_name = model_name
        self.dataloader_train = dataset.get_train_dataloader()
        self.dataloader_val = dataset.get_val_dataloader()
        self.dataset_train = dataset.get_train_dataset()
        self.dataset_val = dataset.get_val_dataset()
        self.dataset = dataset.get_raw_dataset()
        self.model_save_dir = model_save_dir
        self.num_epochs = num_epochs
        self.lr = lr

        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = self.__create_scheduler()
        self.accelerator = Accelerator(mixed_precision="fp16")
        
        # Подготовка модели и оптимизатора
        self.model, self.optimizer, self.dataloader_train, self.dataloader_val = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader_train, self.dataloader_val
        )

    def __create_scheduler(self):
        return get_scheduler(
            'linear',
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.dataloader_train)*self.num_epochs,
        )

    def train(self):
        progress_bar = tqdm(range(len(self.dataloader_train)*self.num_epochs))

        for epoch in range(self.num_epochs):
            self.model.train()
            for step, batch in enumerate(self.dataloader_train):
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)

            # Оценка на валидационных данных
            self.evaluate(epoch)

    def evaluate(self, epoch):
        self.model.eval()
        start_logits = []
        end_logits = []

        for batch in tqdm(self.dataloader_val, desc='Оценка модели'):
            with torch.no_grad():
                outputs = self.model(**batch)
            start_logits.append(self.accelerator.gather(outputs.start_logits).cpu().numpy())
            end_logits.append(self.accelerator.gather(outputs.end_logits).cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(self.dataset_val)]
        end_logits = end_logits[: len(self.dataset_val)]

        metrics = compute_metrics(start_logits, end_logits, self.dataset_val, self.dataset["validation"])
        print(f"Эпоха {epoch + 1}:", metrics)

        # Сохранение модели
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(self.model_save_dir, save_function=self.accelerator.save)
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(self.model_save_dir)

    def save_final_model(self):
        self.tokenizer.save_pretrained(self.model_save_dir)
        self.model.save_pretrained(self.model_save_dir)


if __name__ == "__main__":
    dataset_name = "kuznetsoffandrey/sberquad"
    model_name = "DeepPavlov/rubert-base-cased"
    model_save_dir = "trained_models/rubert-v3"

    dataset = QADataset(dataset_name=dataset_name, model_name=model_name)
    trainer = QATrainer(model_name=model_name, dataset=dataset, model_save_dir=model_save_dir, num_epochs=1)
    
    trainer.train()
    trainer.save_final_model()
