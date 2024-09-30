import os
import telebot
from dotenv import load_dotenv
from model.input_preprocessing import TextSplitter
from model.inference import QAModel

load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)
splitter = TextSplitter()
qa_model = QAModel(r"model\trained_models\rubert-v3")


def handle_input(input_text):
    questions, context = splitter.split_question_context(input_text=input_text)
    response = "Я вижу у вас много вопросов :)\n\n"

    if not context:
        response = "Пожалуйста, предоставьте контекст для ответа на ваш вопрос."
        return response

    for question in questions:
        answer = qa_model.predict(question=question, context=context)
        if not answer:
            answer = "Я не знаю :(\n\Попробуйте предоставить больше контекста."
        response += f"{question}\Ответ: {answer}\n\n"

    return response


@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_message(message.chat.id, f"Здравствуй, {message.from_user.first_name}!\n\nЯ могу ответить на любой вопрос, основываясь на предоставленном контексте. Пожалуйста, напиши контекст и задайте свой вопрос.")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    request = message.text
    response = handle_input(request)
    bot.send_message(message.chat.id, response)

def run_bot():
    bot.infinity_polling()
