import numpy as np
import os


class TensorboardUtil(object):
    def __init__(self, writer, enable=True, **options):
        """
        Args:
            writer(tf.summary.FileWriter): instance of summary writer
            options: other options
        """
        self.enable = enable
        self._options = options
        self._writer = writer
        self._child_writers = dict()
        self.count = 0
        self.flush_steps = 1

    def set_flush_steps(self, steps):
        assert steps >= 1, 'invalid steps < 1.'
        self.flush_steps = steps

    def add_summary(self, summary, step, writer=None):
        if not self.enable:
            return
        if writer is None:
            writer = self._writer
        if writer is not None:
            writer.add_summary(summary, step)
            self.count += 1
            if self.count % self.flush_steps == 0:
                self.flush()
                self.count = 0

    def add_scalar(self, name, value, step):
        if not self.enable:
            return
        import tensorflow as tf
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
        self.add_summary(summary, step)

    def add_scalars(self, name, value_dict, step):
        if not self.enable:
            return
        import tensorflow as tf
        for tag, scalar_value in value_dict.items():
            if tag not in self._child_writers:
                self._child_writers[tag] = tf.summary.FileWriter(self.writer.get_logdir() + f'/{tag}')
            summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=scalar_value)])
            self.add_summary(summary, step, self._child_writers[tag])

    def add_numpy(self, name, value, step):
        if not self.enable:
            return
        import tensorflow as tf
        import numpy as np
        value = np.asarray(value)
        assert isinstance(value, np.ndarray)
        try:
            from tensorflow.python.framework.tensor_util import make_tensor_proto
        except:
            from tensorflow.contrib.util import make_tensor_proto
        tensor_proto = make_tensor_proto(value, value.dtype, value.shape)
        summary = tf.Summary(value=[tf.Summary.Value(tensor=tensor_proto, tag=name)])
        self.add_summary(summary, step)

    def add_histogram(self, name, values, step, bins=10000):
        if not self.enable:
            return
        import tensorflow as tf
        values = np.asarray(values)
        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        bin_edges = bin_edges[1:]
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, histo=hist)])
        self.add_summary(summary, step)

    @property
    def writer(self):
        return self._writer

    @writer.setter
    def writer(self, writer):
        self._writer = writer

    def flush(self):
        if self._writer is not None:
            self._writer.flush()
        for cwriter in self._child_writers.values():
            cwriter.flush()

    def __enter__(self, ):
        return self

    def __exit__(self, type, value, traceback):
        self.flush()


def get_pai_tensorboard_path():
    env_dict = os.environ
    if 'VISUAL_DATA_PATH' in env_dict:
        if 'EXECUTIONRECORD_ID' in env_dict:
            tensorboard_url1 = 'https://aistudio.alipay.com/tensorboard/remote?' \
                               f'executionId={env_dict["EXECUTIONRECORD_ID"]}' \
                               '&nodeName=ais-tmpl-pytorch-1&disk=5120&memory=1024&cpu=1&cluster=et15' \
                               f'&userNumber={env_dict["USERNUMBER"]}'
            print('TENSORBOARD URL1:\n', tensorboard_url1)
            tensorboard_url2 = 'https://aistudio.alipay.com/project/job/tensorboard?' \
                               f'recordId={env_dict["EXECUTIONRECORD_ID"]}' \
                               f'&nodeName={env_dict["kubemaker"]}'
            print('TENSORBOARD URL2:\n', tensorboard_url2)
        return env_dict['VISUAL_DATA_PATH']
    else:
        app_id = env_dict.get('APP_ID')
        return 'pangu://localcluster/pai/optflow/tensorboard/%s/' % app_id
