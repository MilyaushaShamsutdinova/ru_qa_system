from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


class QAModel:
    def __init__(self, model_dir: str):
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def predict(self, question, context):
        inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)
        answer_tokens = inputs['input_ids'][0][start_idx: end_idx + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

        return answer
    