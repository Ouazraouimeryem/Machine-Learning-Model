import nltk
import json
import pickle
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

# Charger les données
data_file = open('/content/intents.json').read()
intents = json.loads(data_file)

# Initialiser les variables
words = []
classes = []
documents = []
ignore_words = ['?', '!']
lemmatizer = WordNetLemmatizer()

# Prétraitement des données
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)  # add each element into list
        #combination between patterns and intents
        documents.append((w, intent['tag']))  # add single element to end of list
        # add to tag in our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lower each word, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save the results in pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
