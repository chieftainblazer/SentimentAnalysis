import string
import re


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = io.open(filename, mode='r', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# turn a doc into clean tokens
def clean_doc(doc):
    # replace '--' with a space ' '
    doc = doc.replace('--', ' ')
    # split into tokens by white space
    doc = re.sub(r'@[A-Za-z0-9]+', "", doc)  # Removed mentions
    doc = re.sub(r"RT[\S]+", '', doc)  # Removing retweets
    doc = re.sub(r"https?:\/\/\S+", '', doc)  # Removes the hyper link
    doc = doc.encode('ascii', "ignore")
    text = text.decode()
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokes that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]
    return tokens


# save tokens to file, one dialog per line
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# load document
in_filename = 'twitter.txt'
doc = load_doc(in_filename)
print(doc[:200])
# clean document
tokens = clean_doc(doc)
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))
# organize into sequences of tokens
length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
    # select sequence of tokens
    seq = tokens[i-length:i]
    # convert into a line
    line = ' '.join(seq)
    # store
    sequences.append(line)
print('Total Sequences: %d' % len(sequences))
# save sequences to file
out_filename = 'twitter_sequences2.txt'
save_doc(sequences, out_filename)

from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = io.open(filename, mode='r', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# define the model
def define_model(vocab_size, seq_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    # compile network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    plot_model(model, to_file='model2.png', show_shapes=True)
    return model


# load
in_filename = 'twitter_sequences2.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]
# define model
model = define_model(vocab_size, seq_length)
# fit model
model.fit(X, y, batch_size=128, epochs=100)
# save the model to file
model.save('model2.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer2.pkl', 'wb'))

from random import randint
from pickle import load
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import io


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = io.open(filename, mode='r', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = np.argmax(model.predict(encoded, verbose=0), axis=-1)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
        if len(result) > 145:
            break
    return ' '.join(result)


# load cleaned text sequences
in_filename = 'twitter_sequences2.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1
# load the model
model = load_model('model2.h5')
# load the tokenizer
tokenizer = load(open('tokenizer.pkl2', 'rb'))
# generate new text
generated = generate_seq(model, tokenizer, seq_length, lines[randint(0,len(lines))], 18)
generated1 = generate_seq(model, tokenizer, seq_length, lines[randint(0,len(lines))], 18)
generated2 = generate_seq(model, tokenizer, seq_length, lines[randint(0,len(lines))], 18)
generated3 = generate_seq(model, tokenizer, seq_length, lines[randint(0,len(lines))], 18)
generated4 = generate_seq(model, tokenizer, seq_length, lines[randint(0,len(lines))], 18)
generated5 = generate_seq(model, tokenizer, seq_length, lines[randint(0,len(lines))], 18)
generated6 = generate_seq(model, tokenizer, seq_length, lines[randint(0,len(lines))], 18)
generated7 = generate_seq(model, tokenizer, seq_length, lines[randint(0,len(lines))], 18)
generated8 = generate_seq(model, tokenizer, seq_length, lines[randint(0,len(lines))], 18)
generated9 = generate_seq(model, tokenizer, seq_length, lines[randint(0,len(lines))], 18)
generated10 = generate_seq(model, tokenizer, seq_length, lines[randint(0,len(lines))], 18)
generated11 = generate_seq(model, tokenizer, seq_length, lines[randint(0,len(lines))], 18)
generated12 = generate_seq(model, tokenizer, seq_length, lines[randint(0,len(lines))], 18)
generated13 = generate_seq(model, tokenizer, seq_length, lines[randint(0,len(lines))], 18)
generated14 = generate_seq(model, tokenizer, seq_length, lines[randint(0,len(lines))], 18)
generated15 = generate_seq(model, tokenizer, seq_length, lines[randint(0,len(lines))], 18)
generated16 = generate_seq(model, tokenizer, seq_length, lines[randint(0,len(lines))], 18)
generated17 = generate_seq(model, tokenizer, seq_length, lines[randint(0,len(lines))], 18)
generated18 = generate_seq(model, tokenizer, seq_length, lines[randint(0,len(lines))], 18)
generated19 = generate_seq(model, tokenizer, seq_length, lines[randint(0,len(lines))], 18)

print(generated)
print(generated1)
print(generated2)
print(generated3)
print(generated4)
print(generated5)
print(generated6)
print(generated7)
print(generated8)
print(generated9)
print(generated10)
print(generated11)
print(generated12)
print(generated13)
print(generated14)
print(generated15)
print(generated16)
print(generated17)
print(generated18)
print(generated19)