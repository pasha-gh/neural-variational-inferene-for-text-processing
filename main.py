import os
import tensorflow as tf

from nvdm import NVDM
from utils import load_20newgroups

flags = tf.app.flags
flags.DEFINE_float("initial_lr", 0.001, "Initial learning rate of adam optimizer")
flags.DEFINE_float("decay_rate", 0.96, "Decay rate of learning rate")
flags.DEFINE_float("decay_step", 10000, "Number of decay step for learning rate decaying")
flags.DEFINE_integer("epochs", 1000, "Number of training epochs")
flags.DEFINE_integer("num_classes", 5, "Number of classes, this will creat N guassian distributions for the prior")
flags.DEFINE_integer("num_samples", 20, "Number of samples for generation")
flags.DEFINE_integer("max_seq_len", 1000, "Maximum document length")
flags.DEFINE_integer("batch_size", 64, "Training and validation batch size")
flags.DEFINE_integer("latent_dim", 50, "The dimension of latent space")
flags.DEFINE_integer("hidden_dim", 500, "The dimension of dence hidden layers")
flags.DEFINE_integer("vocab_size", 2000, "Vocabulary size")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints")
flags.DEFINE_string("device", "1", "GPU device to use")
FLAGS = flags.FLAGS

def main(_):
    train_data, test_data = load_20newgroups(FLAGS.vocab_size, FLAGS.max_seq_len)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.device

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:

        model = NVDM(sess, train_data, test_data,
                     num_classes=FLAGS.num_classes,
                     num_samples=FLAGS.num_samples,
                     batch_size=FLAGS.batch_size,
                     max_seq_len=FLAGS.max_seq_len,
                     hidden_dim=FLAGS.hidden_dim,
                     latent_dim=FLAGS.latent_dim,
                     initial_lr=FLAGS.initial_lr,
                     decay_rate=FLAGS.decay_rate,
                     decay_step=FLAGS.decay_step,
                     epochs=FLAGS.epochs,
                     checkpoint_dir=FLAGS.checkpoint_dir,
                     vocab_size=FLAGS.vocab_size)
        print('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
        model.train()

if __name__ == '__main__':
    tf.app.run()
