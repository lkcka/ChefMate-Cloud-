import pandas as pd
import sqlite3
import streamlit as st
import time

from langchain.chains.llm import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

class DatabaseHandler:
    def __init__(self, db_path):
        self.db_path = db_path

    def load_data(self, query):
        connection = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(query, connection)
        connection.close()
        return df

class RecipeLoader:
    def __init__(self, df):
        self.df = df
        self.documents = self.load_documents()

    def load_documents(self):
        loader = DataFrameLoader(self.df, page_content_column='recipe_name')
        return loader.load()

class RecipeGenerator:
    def __init__(self, api_key, prompt_template, max_tokens=1500):
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.prompt = PromptTemplate(template=prompt_template, input_variables=["algorithm", "ingredients"])
        self.llm = OpenAI(temperature=0, max_tokens=max_tokens, openai_api_key=api_key)
        self.chain = LLMChain(prompt=self.prompt, llm=self.llm)
        self.db = None

    def create_vectorstore(self, documents):
        texts = self.text_splitter.split_documents(documents)
        self.db = FAISS.from_documents(texts, self.embeddings)

    def filter_recipes_by_ingredients(self, recipes, required_ingredients):
        filtered_recipes = []
        for recipe in recipes:
            ingredients = recipe.metadata.get('ingredients', '').lower().split(', ')
            if set(ingredients) == set(required_ingredients):
                filtered_recipes.append(recipe)
        return filtered_recipes

    def generate_recipe_description(self, query):
        # Компонент извлечения: ищем релевантные документы
        relevants = self.db.similarity_search(query)
        if not relevants:
            return "Не удалось найти информацию о данном рецепте.", None

        # Извлекаем данные рецепта из первого найденного документа
        recipe_data = relevants[0].metadata
        algorithm = recipe_data.get('algorithm', '')
        ingredients = recipe_data.get('ingredients', '')
        photo_url = recipe_data.get('photo', None)

        # Генеративный компонент: создаем описание рецепта
        response = self.chain.run(algorithm=algorithm, ingredients=ingredients)
        return response, photo_url

    def search_recipes_by_ingredients(self, query):
        # Разделяем запрос пользователя на отдельные ингредиенты
        required_ingredients = [ingredient.strip().lower() for ingredient in query.split('и')]

        # Ищем релевантные рецепты
        relevants = self.db.similarity_search(query)
        if not relevants:
            return "Не удалось найти информацию о данном рецепте.", None

        # Фильтрация рецептов по ингредиентам
        filtered_recipes = self.filter_recipes_by_ingredients(relevants, required_ingredients)
        if not filtered_recipes:
            return "Не удалось найти рецепт, который содержит только указанные ингредиенты.", None

        # Извлекаем данные рецепта из первого найденного и подходящего документа
        recipe_data = filtered_recipes[0].metadata
        algorithm = recipe_data.get('algorithm', '')
        ingredients = recipe_data.get('ingredients', '')
        photo_url = recipe_data.get('photo', None)

        # Генеративный компонент: создаем описание рецепта
        response = self.chain.run(algorithm=algorithm, ingredients=ingredients)
        return response, photo_url

class StreamlitApp:
    def __init__(self, recipe_generator):
        self.recipe_generator = recipe_generator

    def clean_url(self, url):
        url = "https:" + url
        return url.replace('"', '')

    def run(self):
        tab1, tab2, tab3 = st.tabs(["Поиск", "Поиск по ингредиентам", "О ChefMate"])

        with tab1:
            self.run_search_tab()

        with tab2:
            self.run_search_by_ingredients_tab()

        with tab3:
            self.run_info_tab()

    def run_search_tab(self):
        st.title("Поиск рецептов ChefMate")

        query = st.text_input("Введите ваш запрос:")

        if st.button("Получить рецепт"):
            response, photo_url = self.recipe_generator.generate_recipe_description(query)
            avatar_url = "https://i.yapx.ru/XZyvC.png"
            st.image(avatar_url, width=50)
            st.write("ChefMate:")
            st.write(response)
            if photo_url:
                clean_url = self.clean_url(photo_url)
                st.image(clean_url, width=400)

    def run_search_by_ingredients_tab(self):
        st.title("Поиск рецептов по ингредиентам")

        query = st.text_input("Введите ингредиенты которые у вас есть(например, 'майонез и хлеб'):")

        if st.button("Поиск рецептов по ингредиентам"):
            response, photo_url = self.recipe_generator.search_recipes_by_ingredients(query)
            avatar_url = "https://i.yapx.ru/XZyvC.png"
            st.image(avatar_url, width=50)
            st.write("ChefMate:")
            st.write(response)
            if photo_url:
                clean_url = self.clean_url(photo_url)
                st.image(clean_url, width=400)

    def run_info_tab(self):
        st.markdown("""
        ## Обзор ChefMate
        ChefMate разработан как кулинарный ассистент, предназначенный для предоставления персонализированных рекомендаций по блюдам и помощи на кухне. Его главная задача — сделать готовку и планирование меню приятным и разнообразным процессом, учитывая личные предпочтения пользователя.
        """)
        st.write("")
        st.markdown("""
        ## Основные функции ChefMate

        #### Персонализированные предложения блюд:
        - Для гурмана, предпочитающего легкие и низкокалорийные блюда, ChefMate может предложить воздушный салат из морепродуктов с лимонной заправкой и зеленью, который станет идеальным выбором для летнего обеда.

        #### Поддержка и помощь в приготовлении блюд:
        - Для начинающего повара, желающего освоить основы итальянской кухни, ChefMate предоставляет пошаговые инструкции по приготовлению классического ризотто с грибами и пармезаном.
        """)
        st.write("")
        st.markdown("""
        ## Целевая аудитория ChefMate

        #### Занятые профессионалы
        - Люди, испытывающие нехватку времени для планирования и приготовления пищи. ChefMate помогает им быстро выбирать блюда, соответствующие их графику и диетическим предпочтениям.

        #### Пользователи, заботящиеся о здоровье
        - Те, кто следит за своим здоровьем и придерживается специальных диет. ChefMate помогает выбирать блюда с учетом их пищевых предпочтений и ограничений, обеспечивая сбалансированное и полезное питание.

        #### Кулинарные энтузиасты
        - Любители готовки, стремящиеся к новым вкусовым открытиям. ChefMate предлагает обширную коллекцию уникальных рецептов, позволяя пользователям расширять свои кулинарные горизонты.

        #### Семьи
        - Семьи, где каждый член имеет свои вкусовые предпочтения и диетические требования. ChefMate предлагает универсальные рецепты, которые удовлетворят всех, создавая атмосферу гармонии за обеденным столом.
        """)

# Инициализация компонентов
db_handler = DatabaseHandler("Russian_cuisine.db")
query = "SELECT * FROM 'Russian cuisine'"
df = db_handler.load_data(query)

recipe_loader = RecipeLoader(df)
api_key = st.secrets["api_keys"]["openai"]
recipe_generator = RecipeGenerator(api_key=api_key,
                                   prompt_template="""
Используя данные из рецепта, включающие алгоритм: {algorithm} и ингредиенты: {ingredients}, опишите подробно, как приготовить блюдо, чтобы читатель мог легко следовать вашему описанию.
""")
recipe_generator.create_vectorstore(recipe_loader.documents)

app = StreamlitApp(recipe_generator)
app.run()
