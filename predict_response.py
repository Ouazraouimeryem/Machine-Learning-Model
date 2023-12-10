from keras.models import load_model
import json
import pickle
import nltk
import random
import sys

# Charger le modèle
model = load_model('chatbot_model.h5')

# Charger les données
intents = json.loads(open('/app/intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
lemmatizer = nltk.stem.WordNetLemmatizer()

# Définir des méthodes utilitaires
#Utility Methods

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array

    sentence_words = nltk.word_tokenize(sentence)
    #print(sentence_words)
    # stem each word - create short form for word

    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    #print(sentence_words)

    return sentence_words
#return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern

    sentence_words = clean_up_sentence(sentence)
    #print(sentence_words)

    # bag of words - matrix of N words, vocabulary matrix

    bag = [0]*len(words)
    #print(bag)

    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
                #print ("found in bag: %s" % w)
    #print(bag)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold

    p = bow(sentence, words,show_details=False)
    #print(p)

    res = model.predict(np.array([p]))[0]
    #print(res)

    ERROR_THRESHOLD = 0.25

    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    #print(results)
    # sort by strength of probability

    results.sort(key=lambda x: x[1], reverse=True)
    #print(results)

    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list
    #print(return_list)



def getResponse(ints, intents_json):

    tag = ints[0]['intent']
    #print(tag)

    list_of_intents = intents_json['intents']
    #print(list_of_intents)

    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    #print(ints)

    res = getResponse(ints, intents)
    #print(res)
    return res


#start = True

#while start:

    #query = input('Enter Message:') if sys.stdin.isatty() else sys.stdin.readline().rstrip()
    #if query in ['quit','exit','bye']:
        #start = False
        #continue
    #try:
    #else{    
        #res = chatbot_response(query)
        #print(res)}
    #except:
        #print('You may need to rephrase your question.')
