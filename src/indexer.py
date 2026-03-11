import os
import json
import warnings
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_core.documents import Document
from langchain_community.vectorstores import ElasticsearchStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")
load_dotenv()

ES_HOST = os.getenv('ELASTIC_HOST', 'https://localhost:9200')
ES_USER = os.getenv('ELASTIC_USER')
ES_PASSWORD = os.getenv('ELASTIC_PASSWORD')

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json_path = os.path.join(base_dir, 'data', 'processed', 'ConfluencePages.json')

def main():
    print("1. Проверка связи с Elasticsearch...")
    es_client = Elasticsearch([ES_HOST], basic_auth=(ES_USER, ES_PASSWORD), verify_certs=False, request_timeout=120)

    if not es_client.ping():
        print("❌ ОШИБКА: Не удалось подключиться к Elasticsearch!")
        return
    print("✅ Соединение установлено!")

    print("2. Чтение базы знаний...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    docs = []
    for page in data['pages']:
        if not page['text'].strip():
            continue
            
        chunks = text_splitter.split_text(page['text'])
        
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "title": page['title'],
                    "link": page['link'],
                    "date": page['date'],
                    "author": page['author']
                }
            ))

    print(f"✅ Найдено {len(data['pages'])} статей, разбито на {len(docs)} смысловых чанков.")

    print("3. Загрузка модели ИИ для вычисления векторов...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используемое устройство для векторов: {device.upper()}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/distiluse-base-multilingual-cased-v2',
        model_kwargs={'device': device}
    )

    print("4. Инициализация таблицы в Elasticsearch...")
    vector_store = ElasticsearchStore(
        index_name="articles",
        embedding=embeddings,
        es_connection=es_client,
        vector_query_field='vector',
        query_field='text',
        distance_strategy='COSINE'
    )

    print("5. Вычисление векторов и отправка данных партиями (батчами)...")
    batch_size = 400  # Отправляем по 400 чанков за раз
    
    for i in tqdm(range(0, len(docs), batch_size), desc="Индексация документов"):
        batch = docs[i:i + batch_size]
        try:
            vector_store.add_documents(batch)
        except Exception as e:
            print(f"\n❌ Ошибка при отправке батча {i}-{i+batch_size}: {e}")

    print("\n🎉 УРА! База знаний успешно загружена и проиндексирована!")

if __name__ == "__main__":
    main()