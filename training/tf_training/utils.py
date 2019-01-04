import math
import time

import numpy as np
import tensorflow as tf


# Class holding statistics
class Stats:
    def __init__(self):
        self.s = {}

    def add(self, stat_dict):
        for (k, v) in stat_dict.items():
            if k not in self.s:
                self.s[k] = []
            self.s[k].append(v)

    def n(self, name):
        return len(self.s[name] or [])

    def mean(self, name):
        return np.mean(self.s[name] or [0])

    def stddev_mean(self, name):
        # standard deviation in the sample mean.
        return math.sqrt(
            np.var(self.s[name] or [0]) / max(0.0001, (len(self.s[name]) - 1)))

    def str(self):
        return ', '.join(
            ["{}={:g}".format(k, np.mean(v or [0])) for k, v in self.s.items()])

    def clear(self):
        self.s = {}

    def summaries(self, tags):
        return [tf.Summary.Value(
            tag=k, simple_value=self.mean(v)) for k, v in tags.items()]


# Simple timer
class Timer:
    def __init__(self):
        self.last = time.time()

    def elapsed(self):
        # Return time since last call to 'elapsed()'
        t = time.time()
        e = t - self.last
        self.last = t
        return e

    def reset(self):
        t = time.time()
        self.last = t
