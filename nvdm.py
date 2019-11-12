import time
import numpy as np
import tensorflow as tf

#from tensorflow.layers import dense
from tensorflow.keras.layers import Dense

class NVDM:
    def __init__(self, sess, train_data, test_data, num_classes, num_samples,
                 batch_size, max_seq_len, initial_lr, decay_rate, decay_step,
                 hidden_dim, latent_dim, epochs, checkpoint_dir, vocab_size):

        self.sess = sess
        self.train_data = train_data
        self.test_data = test_data
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.vocab_size = vocab_size

        self.global_step = tf.Variable(0, trainable=False)

        self.build_model()

    def build_model(self):
        self.build_inputs()
        self.build_encoder()
        self.build_latent()
        self.build_posterior()
        self.build_decoder()
        self.build_loss()
        self.build_training_step()

    def build_inputs(self):
        train_dataset = tf.data.Dataset().from_tensor_slices(self.train_data)
        train_dataset = train_dataset.batch(self.batch_size, drop_remainder=True)
        train_dataset = train_dataset.prefetch(1)
        val_dataset = tf.data.Dataset().from_tensor_slices(self.test_data)
        val_dataset = val_dataset.batch(self.batch_size, drop_remainder=True)
        val_dataset = val_dataset.prefetch(1)
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                   train_dataset.output_shapes)

        # This is an op that gets the next element from the iterator
        self.bow = iterator.get_next()
        self.batch_word_count = tf.reduce_sum(tf.reduce_sum(self.bow, -1), -1)

        # These ops let us switch and reinitialize every time we finish an epoch
        self.training_init_op = iterator.make_initializer(train_dataset)
        self.validation_init_op = iterator.make_initializer(val_dataset)

    def build_encoder(self):
        with tf.variable_scope("encoder"):
            self.dense1 = Dense(units=self.hidden_dim,
                                activation=tf.nn.relu).apply(self.bow)

            self.dense2 = Dense(units=self.hidden_dim,
                                activation=tf.nn.relu).apply(self.dense1)

    def build_latent(self):
        with tf.variable_scope("latent"):
            self.mu = Dense(units=self.latent_dim).apply(self.dense2)

            self.log_sigma_sq = Dense(units=self.latent_dim).apply(self.dense2)

            self.sigma_sq = tf.exp(self.log_sigma_sq)

    def build_posterior(self):
        with tf.variable_scope("posterior"):
            self.posterior = []
            for i in range(self.num_samples):
                epsilon = tf.random_normal([self.batch_size, self.latent_dim])
                self.posterior.append(self.mu + epsilon * self.sigma_sq)

    def build_decoder(self):
        with tf.variable_scope("decoder"):
            self.logits = []
            self.dense3 = Dense(units=self.vocab_size)
            for i in range(self.num_samples):
                self.logits.append(self.dense3.apply(self.posterior[i]))

    def build_loss(self):
        self.build_neg_log_likelihood_loss()
        self.build_kl_loss()
        self.loss = self.neg_log_likelihood_loss + self.kl_loss

    def build_neg_log_likelihood_loss(self):
        self.neg_log_likelihood_loss = 0.0
        for i in range(self.num_samples):
            log_softmax = tf.nn.log_softmax(self.logits[i])
            self.neg_log_likelihood_loss += -tf.reduce_sum(log_softmax * self.bow, 1) / self.num_samples
        self.neg_log_likelihood_loss = tf.reduce_sum(self.neg_log_likelihood_loss, axis=0)

    def build_kl_loss(self):
        self.kl_loss = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.exp(self.log_sigma_sq) - self.log_sigma_sq - 1, axis=1)
        self.kl_loss = tf.reduce_sum(self.kl_loss, axis=0)

    def build_training_step(self):
        self.lr = tf.train.exponential_decay(
                self.initial_lr,
                self.global_step,
                self.decay_step,
                self.decay_rate,
                staircase=True,
                name="lr")

        optimizer = tf.train.AdamOptimizer(self.lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(self.loss, var_list=tf.trainable_variables())
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients)

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(1000):
            # Initialize the iterator to consume training data
            self.sess.run(self.training_init_op)
            train_loss = 0
            perplexity = 0
            iter = 0
            while True:
                # As long as the iterator is not empty
                try:
                    _, loss, kl, log = self.sess.run([self.train_op, self.loss, self.kl_loss, self.neg_log_likelihood_loss])
                    iter += 1;
                    train_loss += loss
#                    print(kl, log)
                except tf.errors.OutOfRangeError:
                    train_loss /= iter
                    break

            # We'll store the losses from each batch to get an average
            iter = 0
            test_loss = 0
            log_loss = 0
            word_count = 0
            doc_count = 0
            batch_perplexity = 0
            for i in range(20):
                # Intiialize the iterator to provide validation data
                self.sess.run(self.validation_init_op)
                while True:
                    # As long as the iterator is not empty
                    iter += 1
                    try:
                        loss, batch_log_loss, batch_word_count = self.sess.run([self.loss, self.neg_log_likelihood_loss, self.batch_word_count])
                        test_loss += loss
                        log_loss += self.batch_size
                        word_count += batch_word_count
                        doc_count += self.batch_size
                        batch_perplexity += batch_log_loss * self.batch_size / batch_word_count
                    except tf.errors.OutOfRangeError:
                        break
            test_loss = test_loss / iter
            perplexity = np.exp(batch_perplexity / doc_count)

            print("epoch_{}, train_loss = {}, test_loss = {}, perplexity = {}".format(epoch, train_loss, test_loss, perplexity))
