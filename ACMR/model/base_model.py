'''
Created on Nov 18, 2020
@author: nakaizura
'''

import os
import json
import tensorflow as tf

class BaseDataIter(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def train_data(self):
        raise NotImplemented

    def test_data(self):
        raise NotImplemented
    

class BaseModelParams(object):
    """
    Base class for model parameters
    Any model that takes parameters should derive this class to provide parameters
    """
    def __init__(self):
        """
        Common parameters
        Derived classes should override these parameters
        """
        # Checkpoint root directory; it may contain multiple directories for
        # different models
        self.checkpoint_dir = None

        # Sample directory
        self.sample_dir = None

        # Log directory
        self.log_dir = None

        # Dataset directory; this is the root directory of all datasets.
        # E.g., if dataset coco is located at /mnt/data/coco, then this
        # value should be /mnt/data
        self.dataset_dir = None

        # Name of the dataset; it should be the same as the directory
        # name containing this dataset.
        # E.g., if dataset coco is located at /mnt/data/coco, then this
        # value should be coco
        self.dataset_name = None

        # Name of this model; it is used as the base name for checkpoint files
        self.model_name = None

        # Name of the directory containing the checkpoint files.
        # This can be the same as the model name; however, it can also be encoded
        # to contain certain details of a particular model.
        # This directory will be a subdirectory under checkpoint directory.
        self.model_dir = None

        # Checkpoint file to load
        self.ckpt_file = None

    def load(self, f):
        """
        Load parameters from specified json file.
        The loaded parameters override those with the same name defined in this subclasses
        :param f:
        :return:
        """
        self.__dict__ = json.load(f)

    def loads(self, s):
        """
        Load parameters from json string
        The loaded parameters override those with the same name defined in this subclasses
        :param s:
        :return:
        """
        self.__dict__ = json.loads(s)

    def update(self):
        """
        Update the params
        :return:
        """
        raise Exception('Not implemented')


class BaseModel(object):
    """
    Base class for models
    """
    def __init__(self, model_params=None):
        """
        """
        self.model_params = model_params
        self.saver = None

    def get_checkpoint_dir(self):
        """
        Get the dir for all checkpoints.
        Implemented by the derived classes.
        :return:
        """
        if self.model_params is not None and self.model_params.checkpoint_dir is not None:
            return self.model_params.checkpoint_dir
        else:
            raise Exception('get_checkpoint_dir must be implemented by derived classes')

    def get_model_dir(self):
        """
        Get the model dir for the checkpoint
        :return:
        """
        if self.model_params is not None and self.model_params.model_dir is not None:
            return self.model_params.model_dir
        else:
            raise Exception('get_model_dir must be implemented by derived classes')

    def get_model_name(self):
        """
        Get the base model name.
        Implemented by the derived classes.
        :return:
        """
        if self.model_params is not None and self.model_params.model_name is not None:
            return self.model_params.model_name
        else:
            raise Exception('get_model_name must be implemented by derived classes')

    def get_sample_dir(self):
        """
        Get the dir for samples.
        Implemented by the derived classes.
        :return:
        """
        if self.model_params is not None and self.model_params.sample_dir is not None:
            return self.model_params.sample_dir
        else:
            raise Exception('get_sample_dir must be implemented by derived classes')

    def get_dataset_dir(self):
        """
        Get the dataset dir.
        Implemented by the derived classes.
        :return:
        """
        if self.model_params is not None and self.model_params.dataset_dir is not None:
            return self.model_params.dataset_dir
        else:
            raise Exception('get_dataset_dir must be implemented by derived classes')

    def check_dirs(self):
        if not os.path.exists(self.get_sample_dir()):
            os.mkdir(self.get_sample_dir())

        # sanity check for dataset
        if not os.path.exists(self.get_dataset_dir()):
            raise Exception('Dataset dir %s does not exist' % self.get_dataset_dir())

    def save(self, step, sess):
        checkpoint_dir = os.path.join(self.get_checkpoint_dir(), self.get_model_dir())

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(sess,
                        os.path.join(checkpoint_dir, self.get_model_name()),
                        global_step=step)

    def load(self, sess):
        """
        Load from a specified directory.
        This is for resuming training from a previous snapshot and is called from train(),
        therefore, a saver is created in train()
        Args:
            sess: tf session
        """
        print(' [*] Reading checkpoints...')

        checkpoint_dir = os.path.join(self.get_checkpoint_dir(), self.get_model_dir())

        ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
        if ckpt_path is not None:
            self.saver.restore(sess, ckpt_path)
            return True
        else:
            return False

    def load_for_testing(self, ckpt_path, sess):
        """
        Load from specified checkpoint file.
        This is for testing the model, a saver will be created here to restore the variables
        Args:
            ckpt_path: path to the checkpoint file
            sess: tf session
        """
        print(' [*] Reading checkpoints...')

        if not os.path.exists(ckpt_path):
            return False

        self.saver = tf.train.Saver()
        self.saver.restore(sess, ckpt_path)
        return True
