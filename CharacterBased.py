import re
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


# cleans the text
def cleantext(text):
    text = re.sub(r'@[A-Za-z0-9]+', "", text)  # Removed mentions
    text = re.sub(r"RT[\S]+", '', text)  # Removing retweets
    text = re.sub(r"https?:\/\/\S+", '', text)  # Removes the hyper link
    text = text.encode('ascii', "ignore")
    text = text.decode()
    return text


# save tokens to file, one dialog per line
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, "w")
    file.write(data)
    file.close()


# load text
raw_text = load_doc("twitter.txt")

# clean
tokens = cleantext(raw_text).split()
raw_text = ' '.join(tokens)
print(raw_text)
# organize into sequences of characters
length = 10
sequences = list()
for i in range(length, len(raw_text)):
    # select sequence of tokens
    seq = raw_text[i - length:i + 1]
    # store
    sequences.append(seq)

print("Total Sequences: %d" % len(sequences))
# save sequences to file
out_filename = "twitter_sequences.txt"
save_doc(sequences, out_filename)


from numpy import array
from pickle import dump
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


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
def define_model(X):
    model = Sequential()
    model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(vocab_size, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

# load
in_filename = 'twitter_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')
# integer encode sequences of characters
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))
sequences = list()

for line in lines:
    # integer encode line
    encoded_seq = [mapping[char] for char in line]
    # store
    sequences.append(encoded_seq)

# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)
# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)
# define model
model = define_model(X)
# fit model
model.fit(X, y, epochs=100, verbose=2)
# save the model to file
model.save('model.h5')
# save the mapping
dump(mapping, open('mapping.pkl', 'wb'))


from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
import numpy


# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    # generate a fixed number of characters
    for _ in range(n_chars):
        # encode the characters as integers
        encoded = [mapping[char] for char in in_text]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # one hot encode
        encoded = to_categorical(encoded, num_classes=len(mapping))
        # predict character
        yhat = model.predict(encoded, verbose=0)
        number = numpy.argmax(yhat)
        # reverse map integer to character
        out_char = ''
        for char, index in mapping.items():
            if index == number:
                out_char = char
                break
        # append to input
        in_text += out_char
    return in_text


# load the model
model = load_model('model.h5')
# load the mapping
mapping = load(open('mapping.pkl', 'rb'))
print(generate_seq(model, mapping, 10, 'Another po', 140))
print(generate_seq(model, mapping, 10, 'Tip for to', 140))
print(generate_seq(model, mapping, 10, 'Both parti', 140))
print(generate_seq(model, mapping, 10, 'Great game', 140))
print(generate_seq(model, mapping, 10, 'Luca Pelle', 140))
print(generate_seq(model, mapping, 10, 'Done deal ', 140))
print(generate_seq(model, mapping, 10, 'So after a', 140))
print(generate_seq(model, mapping, 10, 'Juventus a', 140))
print(generate_seq(model, mapping, 10, 'Why do you', 140))
print(generate_seq(model, mapping, 10, 'Mauro Icar', 140))
print(generate_seq(model, mapping, 10, 'We have to', 140))
print(generate_seq(model, mapping, 10, 'It all beg', 140))
print(generate_seq(model, mapping, 10, 'I want Con', 140))
print(generate_seq(model, mapping, 10, 'Zidane com', 140))
print(generate_seq(model, mapping, 10, 'No more of', 140))
print(generate_seq(model, mapping, 10, 'I have hop', 140))
print(generate_seq(model, mapping, 10, 'Talks over', 140))
print(generate_seq(model, mapping, 10, 'Manchester', 140))
print(generate_seq(model, mapping, 10, 'After obta', 140))
print(generate_seq(model, mapping, 10, "I'm immens", 140))

