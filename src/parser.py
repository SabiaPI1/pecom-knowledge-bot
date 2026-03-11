import os
import json
import re
from bs4 import BeautifulSoup

def load_links_mapping(raw_dirpath):
    """
    Проходит по всем .txt файлам (DRK.txt, FTL.txt и т.д.) и собирает ссылки.
    Возвращает список всех найденных ссылок.
    """
    links =[]
    for root, dirs, files in os.walk(raw_dirpath):
        for file in files:
            if file.endswith('.txt'):
                txt_path = os.path.join(root, file)
                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        # Ищем все URL в строке с помощью регулярного выражения
                        urls_in_line = re.findall(r'(https?://[^\s]+)', line)
                        links.extend(urls_in_line)
    return links

def parse_html_to_dict(html_filepath, all_links):
    """Парсит один HTML файл и возвращает словарь с данными."""
    with open(html_filepath, 'r', encoding='utf-8') as html:
        soup = BeautifulSoup(html, 'html.parser')

    # Находим заголовок
    title_tag = soup.find('title')
    title = title_tag.text.strip() if title_tag else "Без заголовка"

    # Безопасное извлечение текста (чтобы слова не склеивались)
    # Ищем блок wiki-content. Если его нет — берем весь body
    content_div = soup.find('div', {'class': 'wiki-content'})
    if content_div:
        text = content_div.get_text(separator=' ', strip=True)
    else:
        body = soup.find('body')
        text = body.get_text(separator=' ', strip=True) if body else soup.get_text(separator=' ', strip=True)
    
    # Убираем лишние множественные пробелы
    text = re.sub(r'\s+', ' ', text)

    # Логика подбора правильной ссылки из .txt
    # Пытаемся найти ссылку из .txt файлов, которая содержит ID страницы или название
    link = ""
    
    # Ищем pageId в самом HTML, чтобы потом найти его среди ссылок из txt
    page_id_meta = soup.find('meta', {'name': 'ajs-page-id'})
    page_id = page_id_meta['content'] if page_id_meta else None

    for txt_url in all_links:
        if page_id and f"pageId={page_id}" in txt_url:
            link = txt_url
            break
            
    # Если в txt не нашли, используем запасной вариант 
    if not link:
        soup_link = soup.find('link', {'rel': 'canonical'})
        link = soup_link.get('href') if soup_link else "Ссылка не найдена"

    # Находим дату изменения
    soup_date = soup.find('a', {'class': 'last-modified'})
    date = soup_date.text.strip() if soup_date else ""

    # Находим авторов
    rows_author = soup.find_all('a', {'class': 'url fn'})
    author = ", ".join([row.text.strip() for row in rows_author if len(row.text.strip()) > 0])

    return {'title': title, 'text': text, 'link': link, 'date': date, 'author': author}

def build_knowledge_base(raw_dirpath, output_json_path):
    pages =[]
    print(f"1. Читаем ссылки из .txt файлов в директории: {raw_dirpath}")
    all_links = load_links_mapping(raw_dirpath)
    
    print(f"2. Начинаем парсинг HTML директории: {raw_dirpath}")
    for root, dirs, files in os.walk(raw_dirpath):
        for file in files:
            if file.endswith('.html'):
                html_filepath = os.path.join(root, file)
                try:
                    page_data = parse_html_to_dict(html_filepath, all_links)
                    pages.append(page_data)
                except Exception as e:
                    print(f"Ошибка при обработке {file}: {e}")

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump({'pages': pages}, f, indent=4, ensure_ascii=False)

    print(f"Парсинг завершен. Сохранено {len(pages)} статей в {output_json_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    out_file = os.path.join(base_dir, 'data', 'processed', 'ConfluencePages.json')

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    build_knowledge_base(raw_dir, out_file)