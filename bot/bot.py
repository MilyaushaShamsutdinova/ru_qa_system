import os
import telebot
from dotenv import load_dotenv
from model.input_preprocessing import TextSplitter
from model.inference import QAModel

# Загрузка переменных окружения
load_dotenv()

# Инициализация Telegram бота с токеном
BOT_TOKEN = os.getenv('BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)

# Инициализация необходимых классов и модели для получения ответа на запрос пользователя
splitter = TextSplitter()
qa_model = QAModel(r"trained_models\rubert-v3")


def handle_input(input_text):
    """
    Функция для обработки текста, содержащего вопрос и контекст.
    
    Разделяет входной текст на вопросы и контекст с помощью TextSplitter и
    выполняет предсказания для каждого вопроса с использованием QAModel.

    Аргументы:
    input_text (str):
        Строка, содержащая вопрос(-ы) и контекст.

    Возвращаемые значения:
    response (str):
        Ответы на вопросы или сообщение об ошибке, если вопросы или контекст не предоставлены.
    """
    # Разделение текста на список вопросов и контекст
    questions, context = splitter.split_question_context(input_text=input_text)

    # Если не удалось выделить ни одного вопроса, возвращаем сообщение об ошибке
    if not questions:
        response = "Задай вопрос в одном сообщении с контекстом, пожалуйста."
        return response
    # Если вопросов больше одного, готовим соответствующий ответ
    elif len(questions) > 1:
        response = "Я вижу у вас много вопросов :)\n\n"
    else:
        response = ""

    # Если контекст не предоставлен, возвращаем сообщение с просьбой предоставить контекст
    if not context:
        response = "Пожалуйста, предоставьте контекст для ответа на ваш вопрос."
        return response

    # Проход по каждому вопросу и выполнение предсказания с помощью модели
    for question in questions:
        answer = qa_model.predict(question=question, context=context)
        # Если модель не смогла дать ответ, отправляем сообщение об этом
        if not answer:
            answer = "Я не знаю :(\n\nПопробуйте предоставить больше релевантного контекста."
        # Формируем полный ответ с вопросами и их предсказанными ответами
        response += f"{question}\nОтвет: {answer}\n\n"

    return response


@bot.message_handler(commands=['start'])
def handle_start(message):
    """
    Функция бота, обрабатывающая команду '/start'.
    
    Отправляет приветственное сообщение, когда пользователь запускает бота.
    
    Аргументы:
    message:
        Объект сообщения от Telegram API.
    """
    bot.send_message(message.chat.id,
                     f"Здравствуй, {message.from_user.first_name}!\n\nЯ могу ответить на любой вопрос, основываясь на предоставленном контексте. Пожалуйста, напиши контекст и задайте свой вопрос.")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """
    Функция бота, обрабатывающая все текстовые сообщения, кроме команд.
    
    Получает текст сообщения от пользователя, затем передает текст
    для обработки функции handle_input и отправляет ответ пользователю через Telegram.

    Аргументы:
    message:
        Объект сообщения от Telegram API.
    """
    request = message.text
    response = handle_input(request)
    bot.send_message(message.chat.id, response)

def run_bot():
    """
    Функция для запуска бота в режиме постоянного опроса. Работает бесконечно.
    """
    bot.infinity_polling()
