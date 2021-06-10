import argparse
import tensorflow as tf
from deepfrier.DeepFRI import DeepFRI


class DeepFRIpb(tf.keras.Model):
    def __init__(self, model_hdf5_fn, output_dim):
        super(DeepFRIpb, self).__init__()
        self.output_dim = output_dim
        input_cmap = tf.keras.layers.Input(shape=(None, None), name='cmap')
        input_seq = tf.keras.layers.Input(shape=(None, 26), name='seq')

        self.model = DeepFRI(output_dim=output_dim, n_channels=26, gc_dims=[512, 512, 512], fc_dims=[1024],
                             lr=0.0002, drop=0.3, l2_reg=1e-4, gc_layer='GraphConv',
                             lm_model_name='trained_models/lstm_lm_tf.hdf5', model_name_prefix='example').model

        # build the model
        self.model([input_cmap, input_seq], False)

        # gradCAM model
        self.grad_model = tf.keras.Model([self.model.inputs], [self.model.get_layer("GCNN_concatenate").output, self.model.output])

        aa_vocab = tf.constant(['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
                                'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M'])
        aa_indx = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
        init_aa2indx = tf.lookup.KeyValueTensorInitializer(aa_vocab, aa_indx)
        self.aa2indx = tf.lookup.StaticHashTable(init_aa2indx, default_value=0)
        self.one_hot = tf.eye(len(aa_vocab), dtype=tf.float32)

        # build the model
        self([input_cmap, input_seq], False)

    def call(self, inputs, training):
        out = tf.squeeze(self.model(inputs, training)[:, :, 0])
        return out

    def load_model_weights(self, fn):
        self.model.load_weights(fn)

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, None), dtype=tf.string)])
    def encode(self, sequence):
        idx = self.aa2indx.lookup(sequence)
        return tf.gather(self.one_hot, idx)

    def _get_gradients_and_filters(self, inputs, class_idx):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(inputs)
            grads = tape.gradient(predictions[:, class_idx, 0], conv_outputs)
        grads = tf.cast(conv_outputs > 0, "float32")*tf.cast(grads > 0, "float32")*grads
        return conv_outputs, grads, predictions

    def _compute_cam(self, output, grad):
        weights = tf.reduce_mean(grad, axis=1)
        # perform weighted sum
        cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
        return cam

    @tf.function(input_signature=[[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.float32)],
                                  tf.TensorSpec(shape=None, dtype=tf.int32)], experimental_compile=True)
    def gradCAM(self, inputs, class_idx):
        output, grad, preds = self._get_gradients_and_filters(inputs, class_idx)
        cam = self._compute_cam(output, grad)
        saliency = (cam - tf.reduce_min(cam))/(tf.reduce_max(cam) - tf.reduce_min(cam))
        return tf.squeeze(saliency)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
        }
        return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--inp_model_fn', type=str, required=True, help="DeepFRI model in *.hdf5 format.")
    parser.add_argument('-n', '--goterms_number', type=int, required=True, help="Number of GO/EC terms in the last layer.")
    parser.add_argument('-o', '--out_model_fn', type=str, default='tf_model', help="Output model (*.hdf5).")

    args = parser.parse_args()

    # load model
    model = DeepFRIpb(args.inp_model_fn, args.goterms_number)
    model.load_model_weights(args.inp_model_fn)

    inp_seq = 'RDSGTVWGALGHGINLNIPNFQMTDDIDEVRWERGSTLVAEFKRKMKPFLKSGAFEILANGDLKIKNLTRDDSGTYNVTVYSTNGTRILNKALDLRIL'

    S = model.encode(tf.constant([list(inp_seq)]))
    A = tf.eye(len(inp_seq), dtype=tf.float32)
    A = tf.reshape(A, [1, A.shape[0], A.shape[1]])
    print (model([A, S]))

    # save model (*.pb)
    # model.save(args.out_model_fn)
    tf.saved_model.save(model, args.out_model_fn)
