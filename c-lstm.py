import keras
from keras.layers import LSTM, Dense, Embedding, MaxPooling1D, Conv1D
from keras.models import Sequential


class LstmModel:
    def __init__(self, max_words, embedding_vector_len, max_len):
        self.max_words = max_words
        self.embedding_vector_length = embedding_vector_len
        self.max_len = max_len
        self.model = self.get_model()

    def get_model(self):
        self.model = Sequential()

        self.model.add(Embedding(self.max_words, self.embedding_vector_length, input_length=self.max_len))
        self.model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(LSTM(100))
        self.model.add(Dense(1, name='out_layer', activation="sigmoid"))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return self.model
