# Імпорт бібліотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from surprise import SVD, NMF, KNNBasic, Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import accuracy

# Завантаження датасетів
@st.cache
def load_data():
    movies = pd.read_csv('ml-latest-small/movies.csv')
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    tags = pd.read_csv('ml-latest-small/tags.csv')
    links = pd.read_csv('ml-latest-small/links.csv')
    return movies, ratings, tags, links

movies, ratings, tags, links = load_data()

# 1. Огляд сирих даних
st.title("Movie Data Overview")

# Виведення датасетів
st.header("Movies Dataset")
st.write(movies.head())
st.header("Ratings Dataset")
st.write(ratings.head())
st.header("Tags Dataset")
st.write(tags.head())
st.header("Links Dataset")
st.write(links.head())

# 2. Підготовка даних для кращої візуалізації
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))
movies['genre_count'] = movies['genres'].apply(len)
ratings['rating'] = ratings['rating'].astype(float)

# Створення колонки середнього рейтингу фільму
movie_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
movie_ratings.columns = ['movieId', 'avg_rating']
movies_with_ratings = pd.merge(movies, movie_ratings, on='movieId', how='left')

# Перевірка результату
st.subheader('Movies with Ratings')
st.write(movies_with_ratings[['movieId', 'title', 'avg_rating', 'genre_count']].head())

# 3. Експлораційний аналіз даних

# Розподіл оцінок
st.header("Distribution of Ratings")
fig, ax = plt.subplots(figsize=(10, 6))  # Створення фігури і осей
sns.histplot(ratings['rating'], bins=10, kde=False, color='blue', ax=ax)  # Передача осей
ax.set_title('Distribution of Ratings')
ax.set_xlabel('Rating')
ax.set_ylabel('Frequency')
ax.grid(True)
st.pyplot(fig)  # Передача фігури

# Розподіл середніх оцінок фільмів
st.header("Distribution of Average Ratings for Movies")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(movies_with_ratings['avg_rating'], bins=10, kde=False, color='green', ax=ax)
ax.set_title('Distribution of Average Ratings for Movies')
ax.set_xlabel('Average Rating')
ax.set_ylabel('Frequency')
ax.grid(True)
st.pyplot(fig)

# Найпопулярніші фільми
st.header("Top 10 Movies by Rating Count")
movie_counts = ratings.groupby('movieId').size().reset_index(name='rating_count')
movies_with_counts = pd.merge(movies_with_ratings, movie_counts, on='movieId')
top_movies = movies_with_counts.sort_values(by='rating_count', ascending=False).head(10)
st.write(top_movies[['title', 'rating_count']])

# Розподіл оцінок для Toy Story
st.header("Rating Distribution for Toy Story")

# Розподіл оцінок для обраного фільму
st.header("Rating Distribution for Selected Movie")

# Отримання унікальних назв фільмів
movie_titles = movies['title'].unique()

# Додання вибору фільму через Streamlit SelectBox
selected_movie = st.selectbox("Choose a movie to view its rating distribution:", movie_titles)

# Перевірка, чи існує обраний фільм у даних
selected_movie_data = movies[movies['title'] == selected_movie]
if not selected_movie_data.empty:
    selected_movie_id = selected_movie_data.iloc[0]['movieId']
    selected_movie_ratings = ratings[ratings['movieId'] == selected_movie_id]
    
    # Візуалізація розподілу оцінок
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(selected_movie_ratings['rating'], bins=10, kde=False, color='orange', ax=ax)
    ax.set_title(f'Rating Distribution for {selected_movie}')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    st.pyplot(fig)
else:
    st.write("The selected movie is not found in the dataset.")


# Жанри фільмів
st.header("Distribution of Movie Genres")
genre_counts = movies['genres'].explode().value_counts().reset_index(name='count')
genre_counts.columns = ['Genre', 'Count']
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='Count', y='Genre', data=genre_counts, palette='viridis', ax=ax)
ax.set_title('Distribution of Movie Genres')
ax.set_xlabel('Number of Movies')
ax.set_ylabel('Genre')
st.pyplot(fig)

# 4. Візуалізація ключових слів (Plot keywords)
st.header("Distribution of Movie Keywords")

# Залишаємо лише ті рядки, де є ключові слова
tags_non_empty = tags[tags['tag'].notna()]

# Обчислюємо кількість кожного унікального ключового слова
keyword_counts = tags_non_empty['tag'].value_counts().reset_index(name='count')
keyword_counts.columns = ['Keyword', 'Count']

# Вибір топ-10 найбільш поширених ключових слів
top_keywords = keyword_counts.head(10)

# Створення барового графіка
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='Count', y='Keyword', data=top_keywords, palette='plasma', ax=ax)
ax.set_title('Top 10 Keywords in Movie Tags')
ax.set_xlabel('Count')
ax.set_ylabel('Keyword')
st.pyplot(fig)

# 5. Створення рекомендательной системи

st.title("Building the Recommender System")

# Побудова рекомендательной системи на основі факторизації матриці (SVD)
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Розділення даних на тренувальну та тестову вибірку
trainset, testset = train_test_split(data, test_size=0.2)

# 5.1 СVD
model_svd = SVD()
model_svd.fit(trainset)
predictions_svd = model_svd.test(testset)
rmse_svd = accuracy.rmse(predictions_svd)

# 5.2 NMF
model_nmf = NMF()
model_nmf.fit(trainset)
predictions_nmf = model_nmf.test(testset)
rmse_nmf = accuracy.rmse(predictions_nmf)

# 5.3 KNN
model_knn = KNNBasic()
model_knn.fit(trainset)
predictions_knn = model_knn.test(testset)
rmse_knn = accuracy.rmse(predictions_knn)

# Виведення результатів
st.write(f"RMSE for SVD: {rmse_svd}")
st.write(f"RMSE for NMF: {rmse_nmf}")
st.write(f"RMSE for KNN: {rmse_knn}")

# Порівняння результатів
st.header("Comparison of Matrix Factorization Models")
fig, ax = plt.subplots(figsize=(10, 6))
model_names = ['SVD', 'NMF', 'KNN']
rmse_values = [rmse_svd, rmse_nmf, rmse_knn]
ax.bar(model_names, rmse_values, color='skyblue')
ax.set_title('RMSE Comparison for Different Matrix Factorization Models')
ax.set_ylabel('RMSE')
st.pyplot(fig)

# Функція для отримання рекомендацій для певного користувача
def get_movie_recommendations(user_id, top_n=5):
    # Отримуємо список усіх фільмів
    all_movie_ids = movies['movieId'].tolist()
    
    # Прогнозуємо оцінки для всіх фільмів для користувача
    predictions = [model_svd.predict(user_id, movie_id) for movie_id in all_movie_ids]
    
    # Сортуємо фільми за оцінками
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Отримуємо top N рекомендованих фільмів
    recommended_movie_ids = [pred.iid for pred in predictions[:top_n]]
    
    # Отримуємо інформацію про рекомендовані фільми
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]
    return recommended_movies[['movieId', 'title']]

# Виведення рекомендацій
user_id = 1  # Тестовий користувач
st.subheader(f"Top 5 Recommendations for User {user_id}")
recommended_movies = get_movie_recommendations(user_id)
st.write(recommended_movies)
