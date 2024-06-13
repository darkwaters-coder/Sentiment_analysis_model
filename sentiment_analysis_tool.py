import nltk
from nltk.corpus import  movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report

#download necessary  NLTK data if not already downloaded
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#Load movie reviews dataset
reviews = [(list(movie_reviews.words(fileid)),category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]

#shuffle the reviews
import random
random.seed(42)
random.shuffle(reviews)

#separate feature(text) and labels(sentiment)
documents = [' '.join(review) for review ,category in reviews]
labels = [category for review ,category in reviews]

#text preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens =word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

#preprocess all documents
processed_documents = [preprocess_text(doc) for doc in documents]

#split the data into training and testing units
x_train ,x_test ,y_train ,y_test =train_test_split(processed_documents,labels,test_size=0.2,random_state = 42)

#TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features= 2000)
x_train_tfidf =vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.fit_transform(x_test)

#initialize logistic regression model
model = LogisticRegression(max_iter=1000)

#train the model
model.fit(x_train_tfidf,y_train)

#predict on the test set
y_pred =model.predict(x_test_tfidf)

#Evaluate model perfomance
accuracy = accuracy_score(y_test,y_pred)
print(f'Accuracy : {accuracy:.2f}')
print(classification_report(y_test,y_pred))
