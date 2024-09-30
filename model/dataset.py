from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator
from utils import preprocess_data


class QADataset:
    def __init__(self, dataset_name, model_name, batch_size=16):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.dataset = None
        self.dataset_train = None
        self.dataset_val = None
        self.dataloader_train = None
        self.dataloader_val = None
        self.__load_and_preprocess_data()
        self.__create_dataloaders()

    def __load_and_preprocess_data(self):
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
        self.dataset_train.set_format("torch")
        dataset_val_formatted = self.dataset_val.remove_columns(["example_id", "offset_mapping"])
        dataset_val_formatted.set_format("torch")

        self.dataloader_train = DataLoader(
            self.dataset_train,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=self.batch_size
        )

        self.dataloader_val = DataLoader(
            dataset_val_formatted,
            collate_fn=default_data_collator,
            batch_size=self.batch_size
        )
    
    def get_raw_dataset(self):
        return self.dataset

    def get_train_dataset(self):
        return self.dataset_train
    
    def get_val_dataset(self):
        return self.dataset_val

    def get_train_dataloader(self):
        return self.dataloader_train

    def get_val_dataloader(self):
        return self.dataloader_val
