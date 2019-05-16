from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D

from ..dictionary import Dictionary


def build_network(n_features=100000, embedding_dimensions=300, input_length=20):
    """
    """
    model = Sequential()
    model.add(Embedding(n_features,
                        embedding_dimensions,
                        input_length=input_length))

    model.add(GlobalAveragePooling1D())

    model.add(Dense(1, activation='sigmoid'))

    return model
