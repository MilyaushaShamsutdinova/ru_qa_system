from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import time


class QAModel:
    """
    Класс для работы с моделью вопрос-ответ (QA), загружаемой из директории.
    Предоставляет функционал для предсказания ответа на заданный вопрос в контексте.
    """

    def __init__(self, model_ref: str):
        """
        Инициализирует модель и токенизатор на основе предобученной модели.

        Аргументы:
        model_ref (str):
            Название модели на платформе Hugging Face или путь к директории
            с предобученной моделью и токенизатором.
        """
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_ref)
        self.tokenizer = AutoTokenizer.from_pretrained(model_ref)

    def predict(self, question, context):
        """
        Предсказывает ответ на вопрос на основе переданного контекста.

        Аргументы:
        question (str):
            Вопрос, на который нужно получить ответ.
        context (str):
            Текст контекста, в котором содержится ответ.

        Возвращаемые значения:
        answer (str):
            Строка с предсказанным ответом.
        """
        # Токенизация вопроса и контекста
        inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True, padding=True)

        # Прямой проход через модель
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Определение индекса начала и конца ответа
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)

        # Извлечение токенов ответа на основе индексов начала и конца и декодирование токенов обратно в текст
        answer_tokens = inputs['input_ids'][0][start_idx: end_idx + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

        return answer
    