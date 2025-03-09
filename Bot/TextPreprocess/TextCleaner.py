### необходимо удалить стоп слова и знаки препинания
import nltk
import re
import emoji
import pymorphy2

from num2words import num2words
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm


class TextCleaner:
    '''
        TextCleaner - класс, позволяющий настроить собственный пайплайн предобработки текста для задачи
        распознавания в тексте спама
    '''

    def __init__(self):
        nltk.download('stopwords')
        nltk.download('punkt')
        self.stopwords = set(list(stopwords.words('russian')) + list(stopwords.words('english')))
        self.morph = pymorphy2.MorphAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.ru_stemmer = SnowballStemmer('russian')
        self.en_stemmer = SnowballStemmer('english')

    def text_clean(self,
                   texts,
                   stopwords=True,
                   nums2txt=True,
                   punkt=False,
                   del_nums=False,
                   ret='tokens',
                   lemmatization=False,
                   stemming=False,
                   max_digits=10):
        '''
            texts - список текстов для предобработки
            stopwords = True - удаление стоп слов
            punkt = False - удаление знаков пунктуации
            nums2txt = True - перевод числе вида 100 в вид сто
            del_nums = False - удаление цифр
            ret = tokens - формат в котором нужно вернуть данные
            max_digits - параметр для nums2txt, определяющий какой максимальной длины числа преобразовывваем
        '''
        cleaned_text = []
        for i, text in tqdm(enumerate(texts)):
            try:
                if nums2txt:
                    text = self.replace_numbers_with_words(text, max_digits=max_digits)
                text = self.emojis_words(text)

                text = text.lower()

                tokens = word_tokenize(text)

                if stopwords:
                    tokens = [word for word in tokens if word not in self.stopwords]

                text = ' '.join(tokens)
                if punkt:
                    text = re.sub(r'[^\w\s]', '', text)
                if del_nums:
                    text = re.sub(r'\d+', '', text)

                if lemmatization:
                    text = self.lemmatize_text(text)
                    text = ' '.join(text)
                if stemming:
                    text = self.stem_text(text)
                    text = ' '.join(text)

                if ret == 'tokens':
                    cleaned_text.append(text.split())
                else:
                    cleaned_text.append(' '.join(text))

            except Exception as e:
                raise e

        return cleaned_text

    def stem_text(self, text: str):
        def detect_language(word):
            return 'ru' if re.search(r'[а-яА-Я]', word) else 'en'

        stemmed_tokens = []

        tokens = text.split()

        for token in tokens:
            lang = detect_language(token)
            if lang == 'ru':
                stemmed_tokens.append(self.russian_stemmer.stem(token))
            else:
                stemmed_tokens.append(self.english_stemmer.stem(token))

        return stemmed_tokens

    def replace_numbers_with_words(self, text: str, language='ru', max_digits=10):
        def replace_match(match):
            number = match.group(0)
            if len(number) > max_digits:
                return '[LONG_NUMBER]'
            return num2words(int(number), lang=language)

        return re.sub(r'\d+', replace_match, text)

    def emojis_words(self, text: str):
        # Модуль emoji: преобразование эмоджи в их словесные описания
        clean_text = emoji.demojize(text, delimiters=(" ", " "))
        # Редактирование текста путём замены ":" и" _", а так же - путём добавления пробела между отдельными словами
        clean_text = clean_text.replace(":", "").replace("_", " ")
        return clean_text

    def lemmatize_text(self, text):
        def detect_language(word):
            return 'ru' if re.search(r'[а-яА-Я]', word) else 'en'

        tokens = text.split()

        lemmatized_tokens = []
        for token in tokens:
            lang = detect_language(token)
            if lang == 'ru':
                lemmatized_tokens.append(self.morph.parse(token)[0].normal_form)
            else:
                lemmatized_tokens.append(self.lemmatizer.lemmatize(token))

        return lemmatized_tokens