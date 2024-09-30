import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Tuple


class TextSplitter:
    """
    Простой класс для разделения текста на вопросы и контекст. 
    """

    def __init__(self):
        """
        Инициализация класса.
        Автоматически загружает необходимый токенизатор NLTK для работы с русским языком.
        """
        # nltk.download('punkt', quiet=True)
        pass

    def split_question_context(self, input_text: str) -> Tuple[List[str], str]:
        """
        Разделяет входной текст на вопрос (или вопросы, если найдено несколько вопросов) и контекст.
        Считает предложения, заканчивающиеся на '?', вопросами, а остальные — контекстом.

        Аргументы:
        input_text (str):
            Входной текст, содержащий как вопрос, так и контекст.

        Возвращаемые значения:
        questions ([List[str]): 
            Список вопросов, извлеченных из текста.
        context (str):
            Контекст, собранный из предложений, которые не являются вопросами.
        """
        sentences = sent_tokenize(input_text, language='russian')
        questions = []
        context = []

        for sentence in sentences:
            if sentence.endswith('?'):
                questions.append(sentence.strip())
            else:
                context.append(sentence.strip())

        context = ' '.join(context)
        return questions, context
