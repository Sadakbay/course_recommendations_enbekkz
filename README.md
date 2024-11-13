# Система рекомендации курсов на основе навыков пользователя

## Описание проекта
Проект представляет собой рекомендательную систему, которая подбирает образовательные курсы на основе текущих навыков пользователя. Система использует семантический анализ и векторные представления текста для определения релевантных курсов.

## Технологический стек
- Python 3.x
- pandas - для обработки данных
- transformers (rubert-tiny) - для векторного представления текста
- sklearn - для вычисления косинусной близости
- numpy - для работы с векторами

## Архитектура системы

### 1. Подготовка данных
```python
import pandas as pd
import re

course_data = pd.read_csv('course_data3.csv')
course_data = course_data[['Course Title', 'Unnamed: 1']]
course_data = course_data.drop_duplicates()
course_data = course_data.dropna()
course_data['Unnamed: 1'] = course_data['Unnamed: 1'].str.lower()
course_data = course_data.rename(columns = {'Unnamed: 1': 'Tags'})
```

### 2. Векторизация текста
Система использует модель rubert-tiny для создания векторных представлений текста:
```python
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "cointegrated/rubert-tiny"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
```

### 3. Поиск похожих курсов
```python
def find_similar_courses(input_embedding, tag_embeddings, course_titles, top_n=3):
    if input_embedding.ndim == 1:
        input_embedding = input_embedding.reshape(1, -1)
    
    similarities = cosine_similarity(input_embedding, tag_embeddings).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    results = [(course_titles[i], float(similarities[i])) for i in top_indices]
    return results
```

### 4. Предварительная обработка текста
```python
def clean_text(text):
    text = text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    return text
```

## Использование системы

### Пример использования:
```python
# Подготовка данных
tag_embeddings = np.array([get_embedding(tags) for tags in course_data['Tags']])
course_titles = course_data['Course Title'].tolist()

# Поиск курсов по запросу
query = "бухгалтер"
query = clean_text(query)
query_embedding = get_embedding(query)

similar_courses = find_similar_courses(
    query_embedding,
    tag_embeddings,
    course_titles
)

# Вывод результатов
for title, score in similar_courses:
    print(f"Course: {title}")
    print(f"Similarity: {score:.4f}\n")
```

## Особенности и преимущества
1. Использование современной языковой модели rubert-tiny, оптимизированной для русского языка
2. Семантический поиск вместо простого текстового сопоставления
3. Возможность находить релевантные курсы даже при различных формулировках навыков
4. Масштабируемость решения для большого количества курсов

## Возможные улучшения
1. Добавление фильтрации по дополнительным параметрам (длительность курса, сложность и т.д.)
2. Реализация учета истории просмотров пользователя
3. Внедрение механизма обратной связи для улучшения рекомендаций
4. Оптимизация производительности для больших наборов данных
5. Добавление многоязычной поддержки

## Установка и настройка

### Требования
```
pandas
transformers
torch
scikit-learn
numpy
```

### Установка зависимостей
```bash
pip install pandas transformers torch scikit-learn numpy
```

## Контрибьюция
Мы приветствуем вклад в развитие проекта. Пожалуйста, создавайте pull requests или открывайте issues для обсуждения предлагаемых изменений.
