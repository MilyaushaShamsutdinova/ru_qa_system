import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Tuple


class TextSplitter:
    def __init__(self):
        nltk.download('punkt', quiet=True)

    def split_question_context(self, input_text: str) -> Tuple[List[str], str]:
        sentences = sent_tokenize(input_text, language='russian')
        questions = []
        context = []
        for sentence in sentences:
            if sentence.endswith('?'):
                questions.append(sentence.strip())
            else:
                context.append(sentence.strip())
        return questions, ' '.join(context)
