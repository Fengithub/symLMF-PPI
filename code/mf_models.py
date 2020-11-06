from keras.models import Model
from keras.layers import Input, Reshape, Dot, Activation, Add
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras import constraints, initializers
import numpy as np

class symLMF:

    def __init__(self, k = 30, l2_p = 1e-6, batch_size = 64, epochs=20, c = 1):
        self.k = k
        self.l2_p = l2_p
        self.batch_size = batch_size
        self.epochs = epochs
        self.c = c

    def build_model(self, n_users, n_factors, l2_p):
        user = Input(shape=(1,))
        u_emb = Embedding(n_users, n_factors, embeddings_initializer='he_normal', embeddings_regularizer=l2(l2_p))
        u = u_emb(user)
        u = Reshape((n_factors,))(u)
        u_bias = Embedding(n_users, 1, embeddings_initializer='he_normal', embeddings_regularizer=l2(l2_p))
        ub = u_bias(user)
        ub = Reshape((1,))(ub)

        item = Input(shape=(1,))
        m = u_emb(item)
        m = Reshape((n_factors,))(m)
        mb = u_bias(item)
        mb = Reshape((1,))(mb)

        x = Dot(axes=1)([u, m])
        x = Add()([x, ub, mb])
        x = Activation('sigmoid')(x)

        model = Model(inputs=[user, item], outputs=x)
        opt = Adam(lr=0.001)
        model.compile(loss='mean_squared_error', optimizer=opt)

        user_embedding_model = Model(inputs=user, outputs=u)
        user_bias_model = Model(inputs=user, outputs=ub)

        return [model, user_embedding_model, user_bias_model]

    def train_model(self, m, X_train, y_train, X_test, y_test):
        [model, user_embedding_model, user_bias_model] = self.build_model(m, self.k, self.l2_p)
        sample_weight = np.ones(shape=(len(y_train), ))
        sample_weight[y_train == 1] = self.c # fit with sample weight
        history = model.fit(X_train, y_train, sample_weight = sample_weight, batch_size = self.batch_size, epochs = self.epochs, verbose = 0, \
                    validation_data = (X_test, y_test))
        return [model, user_embedding_model, user_bias_model]

    def get_test_prediction(self, model, X_test):
        return model.predict(X_test)

    def extract_embedding_matrix(self, user_embedding_model, m):
        userIndex = np.array(list(range(m)))
        user_embedding_matrix = user_embedding_model.predict(userIndex)

        return user_embedding_matrix


class symPMF:

    def __init__(self, k = 30, l2_p = 1e-6, batch_size = 64, epochs=20, c = 1):
        self.k = k
        self.l2_p = l2_p
        self.batch_size = batch_size
        self.epochs = epochs
        self.c = c

    def build_model(self, n_users, n_factors, l2_p):
        user = Input(shape=(1,))
        u_emb = Embedding(n_users, n_factors, embeddings_initializer='he_normal', embeddings_regularizer=l2(l2_p))
        u = u_emb(user)
        u = Reshape((n_factors,))(u)
        u_bias = Embedding(n_users, 1, embeddings_initializer='he_normal', embeddings_regularizer=l2(l2_p))
        ub = u_bias(user)
        ub = Reshape((1,))(ub)

        item = Input(shape=(1,))
        m = u_emb(item)
        m = Reshape((n_factors,))(m)
        mb = u_bias(item)
        mb = Reshape((1,))(mb)

        x = Dot(axes=1)([u, m])
        x = Add()([x, ub, mb])

        model = Model(inputs=[user, item], outputs=x)
        opt = Adam(lr=0.001)
        model.compile(loss='mean_squared_error', optimizer=opt)

        user_embedding_model = Model(inputs = user, outputs = u)
        user_bias_model = Model(inputs = user, outputs = ub)

        return [model, user_embedding_model, user_bias_model]

    def train_model(self, m, X_train, y_train, X_test, y_test):
        [model, user_embedding_model, user_bias_model] = self.build_model(m, self.k, self.l2_p)
        sample_weight = np.ones(shape=(len(y_train), ))
        sample_weight[y_train == 1] = self.c # fit with sample weight
        history = model.fit(X_train, y_train, sample_weight = sample_weight, batch_size = self.batch_size, epochs = self.epochs, verbose = 0, \
                    validation_data = (X_test, y_test))
        return [model, user_embedding_model, user_bias_model]

    def get_test_prediction(self, model, X_test):
        return model.predict(X_test)

    def extract_embedding_matrix(self, user_embedding_model, m):
        userIndex = np.array(list(range(m)))
        user_embedding_matrix = user_embedding_model.predict(userIndex)

        return user_embedding_matrix


class symNMF:

    def __init__(self, k = 30, l1_p = 1e-3, batch_size = 64, epochs=20, c = 1):
        self.k = k
        self.l1_p = l1_p
        self.batch_size = batch_size
        self.epochs = epochs
        self.c = c

    def build_model(self, n_users, n_factors, l1_p):
        user = Input(shape=(1,))
        u_emb = Embedding(n_users, n_factors, embeddings_constraint=constraints.NonNeg(), embeddings_initializer=initializers.RandomUniform(minval=0, maxval=1), embeddings_regularizer=l1(l1_p))
        u = u_emb(user)
        u = Reshape((n_factors,))(u)
        u_bias = Embedding(n_users, 1, embeddings_constraint=constraints.NonNeg(), embeddings_initializer=initializers.RandomUniform(minval=0, maxval=1), embeddings_regularizer=l1(l1_p))
        ub = u_bias(user)
        ub = Reshape((1,))(ub)

        item = Input(shape=(1,))
        m = u_emb(item)
        m = Reshape((n_factors,))(m)
        mb = u_bias(item)
        mb = Reshape((1,))(mb)

        x = Dot(axes=1)([u, m])
        x = Add()([x, ub, mb])

        model = Model(inputs=[user, item], outputs=x)
        opt = Adam(lr=0.001)
        model.compile(loss='mean_squared_error', optimizer=opt)

        user_embedding_model = Model(inputs = user, outputs = u)
        user_bias_model = Model(inputs = user, outputs = ub)

        return [model, user_embedding_model, user_bias_model]

    def train_model(self, m, X_train, y_train, X_test, y_test):
        [model, user_embedding_model, user_bias_model] = self.build_model(m, self.k, self.l1_p)
        sample_weight = np.ones(shape=(len(y_train), ))
        sample_weight[y_train == 1] = self.c # fit with sample weight
        history = model.fit(X_train, y_train, sample_weight = sample_weight, batch_size = self.batch_size, epochs = self.epochs, verbose = 0, \
                    validation_data = (X_test, y_test))
        return [model, user_embedding_model, user_bias_model]

    def get_test_prediction(self, model, X_test):
        return model.predict(X_test)

    def extract_embedding_matrix(self, user_embedding_model, m):
        userIndex = np.array(list(range(m)))
        user_embedding_matrix = user_embedding_model.predict(userIndex)

        return user_embedding_matrix
