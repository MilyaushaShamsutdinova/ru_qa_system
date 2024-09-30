import numpy as np
import collections
import evaluate
from tqdm.auto import tqdm


def preprocess_data(examples, tokenizer, is_test=False, max_length=384, stride=128):
    """
    Предобработка и токенизация примеров для подготовки к инференсу
    
    Аргументы:
    ----------
    examples : datasets.Dataset
        Набор данных с примерами. Должен содержать столбцы:
        'id', 'question', 'context'
    tokenizer : transformers.AutoTokenizer
        Токенайзер для модели
    is_test : bool
        True, если это тестовый или валидационный набор
        False, если это тренировочный набор
    max_length : int
        Максимальная длина для усечения контекста
    stride : int
        Шаг для усечения контекста

    Возвращаемые значения:
    -----------
    inputs : dict
        Токенизированный и обработанный словарь данных с ключами
        'input_ids', 'attention_mask', 'start_positions', 'end_positions'
        Все значения — это списки длиной, равной количеству входных данных,
        которые вывел токенайзер:
            inputs['input_ids'][k] : list
                токен-идентификаторы, соответствующие токенам в признаке k
            inputs['attention_mask'][k] : list
                маска внимания для признака k
            inputs['start_positions'][k] : int
                начальные позиции токенов для ответа в последовательности k
            inputs['end_positions'][k] : int
                конечные позиции токенов для ответа в последовательности k
        Если is_test == True, добавляются ключи 'offset_mapping' и 'example_id':
            inputs['offset_mapping'][k] : list
                изменённые смещения для признака k (значение None, если это не токен контекста)
            inputs['example_id'][k] : int
                id примера, из которого взят признак k
    """
    # Удаление лишних пробелов в конце вопроса
    questions = [q.strip() for q in examples['question']]
    
    # Токенизация вопросов и контекста с усечением только контекста
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    answers = examples["answers"]
    if is_test:
        offset_mapping = inputs["offset_mapping"]
        example_ids = []
    else:
        offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    start_positions = []
    end_positions = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        offset = offset_mapping[i]
        sequence_ids = inputs.sequence_ids(i)

        # Если это тест/валидация, сохраняем example_ids и изменяем offset_mapping
        if is_test:
            example_ids.append(examples["id"][sample_idx])
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        # Некоторые примеры не содержат ответа, что указывает на отсутствие ответа в любом
        # из сегментов контекста, и все соответствующие сегменты контекста должны получить метки (0,0)
        if len(answer['answer_start'])==0:
            start_positions.append(0)
            end_positions.append(0)
            continue
        
        # Если ответ есть, записываем его начальный и конечный символьные индексы и находим sequence_ids
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        
        # Находим начало и конец сегмента контекста
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # Если ответ не полностью находится в сегменте контекста, метка (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Иначе находим начальные и конечные позиции ответа
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    if is_test:
        inputs["example_id"] = example_ids
    return inputs


def format_predictions(start_logits, end_logits, inputs, examples,
                       n_best=20, max_answer_length=30):
    """
    Постобработка логитов в данные с предсказаниями

    Аргументы:
    ----------
    start_logits, end_logits : list, list
        последовательности логитов, соответствующие возможным начальным
        и конечным индексам токенов для ответа
    inputs : dataset
        Токенизированный и предобработанный набор данных с колонками
        'example_id', 'offset_mapping' (другие колонки игнорируются)
    examples : datasets.Dataset
        Набор данных с примерами. Должен содержать столбцы:
        'id', 'question', 'context'
    n_best : int
        Количество лучших индексов начала/конца (по логитам) для рассмотрения
    max_answer_length : int
        Максимальная длина (в символах), допустимая для кандидата ответа
        
    Возвращаемые значения:
    -----------
    predicted_answers : list(dict)
        для каждой записи ключи: 'id', 'prediction_text'
    """
    assert n_best <= len(inputs['offset_mapping'][0]), 'n_best cannot be larger than max_length'
    
    # Словарь, где ключи - это id примеров, а значения - соответствующие индексы токенизированных признаков
    example_to_inputs = collections.defaultdict(list)
    for idx, feature in enumerate(inputs):
        example_to_inputs[str(feature["example_id"])].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = str(example["id"])
        context = example["context"]
        answers = []
        
        for feature_index in example_to_inputs[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            
            # Получаем изменённое offset_mapping; Индексы токенов контекста имеют реальные смещения,
            # все остальные индексы имеют значение None
            offsets = inputs[feature_index]['offset_mapping']
            
            # Получаем индексы n_best наиболее вероятных начальных и конечных токенов
            start_indices = np.argsort(start_logit)[-1:-n_best-1:-1].tolist()
            end_indices = np.argsort(end_logit)[-1 :-n_best-1: -1].tolist()

            for start_index in start_indices:
                for end_index in end_indices:
                    # Пропускаем пары, где длина ответа отрицательная или больше max_answer_length
                    if(end_index < start_index or end_index - start_index + 1 > max_answer_length):
                        continue
                    
                    # Пропускаем пары, где одна из смещений имеет значение None,
                    # что указывает на частичный ответ в контексте
                    if (offsets[start_index] is None)^(offsets[end_index] is None):
                        continue
                    
                    # Пары с None для обоих указывают на пустую строку как предсказание ответа
                    # Суммирование логитов эквивалентно умножению вероятностей
                    if (offsets[start_index] is None)&(offsets[end_index] is None):
                        answers.append(
                            {
                                "text": '',
                                "logit_score": start_logit[start_index] + end_logit[end_index],
                            }
                        )
                    # Если ни один из них не None и ответ имеет положительную длину менее
                    # max_answer_length, тогда это соответствует непустому кандидату на ответ
                    # в контексте, и мы хотим включить его в наш список
                    else:
                        answers.append(
                            {
                                "text": context[offsets[start_index][0] : offsets[end_index][1]],
                                "logit_score": start_logit[start_index] + end_logit[end_index],
                            }
                        )
            
        # Если есть кандидаты на ответ, выбираем кандидата с наибольшим логит-очком это может быть ''
        if len(answers)>0:
            best_answer = max(answers, key=lambda x:x['logit_score'])
            predicted_answers.append({'id':example_id, 'prediction_text':best_answer['text']})
        else:
            predicted_answers.append({'id':example_id, 'prediction_text':''})

    return predicted_answers


def compute_metrics(start_logits, end_logits, inputs, examples,
                    n_best = 20, max_answer_length=30):
    """
    Вычисление результатов метрики squad v2 на предсказаниях

    Аргументы:
    -----------
    start_logits, end_logits : list, list
        Последовательности логитов, соответствующие возможным индексам
        начальных и конечных токенов ответа
    inputs : dataset
        Токенизированный и предварительно обработанный набор данных, содержащий столбцы
        'example_id', 'offset_mapping' (другие столбцы игнорируются)
    examples : datasets.Dataset
        Набор данных примеров. Должен содержать столбцы:
        'id', 'question', 'context'
    n_best : int
        Количество лучших индексов начальных/конечных (по логиту) для рассмотрения
    max_answer_length : int
        Максимальная длина (в символах), допустимая для кандидата на ответ

    Возвращаемые значения:
    --------
    metrics : dict
        Словарь значений метрик
    """
    metric = evaluate.load('squad_v2')
    predicted_answers = format_predictions(start_logits, end_logits, inputs, examples,
                                                   n_best=n_best, max_answer_length=max_answer_length)
    for pred in predicted_answers:
        pred['no_answer_probability'] = 1.0 if pred['prediction_text'] == '' else 0.0

    correct_answers = []
    for example in examples:
        input_id = str(example["id"])
        answers = example["answers"]
        correct_answers.append({
            "id": input_id,
            "answers": {
                "text": answers["text"],
                "answer_start": answers["answer_start"]
            }
        })
    
    return metric.compute(predictions=predicted_answers, references=correct_answers)
