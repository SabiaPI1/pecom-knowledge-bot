import os
import asyncio
import concurrent.futures
from telebot.async_telebot import AsyncTeleBot
from dotenv import load_dotenv
from rag_pipeline import KnowledgeBaseRAG

load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ES_USER = os.getenv('ELASTIC_USER')
ES_PASSWORD = os.getenv('ELASTIC_PASSWORD')
ES_HOST = os.getenv('ELASTIC_HOST', 'https://localhost:9200')

# Инициализируем асинхронного бота
bot = AsyncTeleBot(TELEGRAM_TOKEN)

print("Запуск системы RAG...")
rag_system = KnowledgeBaseRAG(es_host=ES_HOST, es_user=ES_USER, es_password=ES_PASSWORD)

# Пул потоков для выполнения тяжелой LLM-задачи, чтобы не блокировать Telegram
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

@bot.message_handler(commands=['start', 'help'])
async def send_welcome(message):
    welcome_text = (
        "Привет! Я ИИ-помощник по базе знаний Confluence ПЭК.\n"
        "Задай мне любой вопрос о процессах (например, 'Как занести контакт в черный список?'), "
        "и я найду ответ в инструкциях, а также дам ссылку на источник."
    )
    await bot.reply_to(message, welcome_text)

@bot.message_handler(func=lambda message: True)
async def handle_message(message):
    question = message.text
    
    processing_msg = await bot.reply_to(message, "Ищу информацию в базе знаний и генерирую ответ. Пожалуйста, подождите ⏳...")
    
    try:
        # Запускаем синхронную LLM в фоне (executor), чтобы бот мог общаться с другими
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, rag_system.ask, question)
        
        final_answer = f"{result['answer']}\n\n🔗 Источник: {result['link']}"
        
        await bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=processing_msg.message_id,
            text=final_answer
        )
    except Exception as e:
        await bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=processing_msg.message_id,
            text=f"Произошла ошибка при обработке: {e}"
        )

if __name__ == '__main__':
    print("Бот успешно запущен и готов к работе!")
    # Запуск асинхронного бота
    asyncio.run(bot.polling())