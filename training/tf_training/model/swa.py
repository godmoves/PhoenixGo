#!/use/bin/env python3
import tensorflow as tf


# SWA (stochastic weight averaging) is a technique we use to help the neural
# network converge faster and better.


class SWA:
    def __init__(self, session, model, c=1, max_n=16, recalc_bn=True):
        self.model = model
        self.session = session

        # Net sampling rate (e.g 2 == every 2nd network).
        self.swa_c = c

        # Take an exponentially weighted moving average over this
        # many networks. Under the SWA assumptions, this will reduce
        # the distance to the optimal value by a factor of 1/sqrt(n)
        self.swa_max_n = max_n

        # Recalculate SWA weight batchnorm means and variances
        self.swa_recalc_bn = recalc_bn

        # Count of networks accumulated into SWA
        self.swa_count = tf.Variable(0., name='swa_count', trainable=False)

        # Count of networks to skip
        self.swa_skip = tf.Variable(self.swa_c, name='swa_skip', trainable=False)

    def construct_net(self, x):
        return self.model.construct_net(x)

    @property
    def training(self):
        return self.model.training

    def init_swa(self):
        # Build the SWA variables and accumulators
        accum = []
        load = []
        n = self.swa_count
        for w in self.model.weights:
            name = w.name.split(':')[0]
            var = tf.Variable(tf.zeros(shape=w.shape), name='swa/' + name, trainable=False)
            accum.append(tf.assign(var, var * (n / (n + 1.)) + w * (1. / (n + 1.))))
            load.append(tf.assign(w, var))
        with tf.control_dependencies(accum):
            self.swa_accum_op = tf.assign_add(n, 1.)
        self.swa_load_op = tf.group(*load)

    def save_swa_network(self, steps, path, data):
        # Sample 1 in self.swa_c of the networks. Compute in this way so
        # that it's safe to change the value of self.swa_c
        rem = self.session.run(tf.assign_add(self.swa_skip, -1))
        if rem > 0:
            return
        self.swa_skip.load(self.swa_c, self.session)

        # Add the current weight vars to the running average.
        num = self.session.run(self.swa_accum_op)

        if self.swa_max_n is not None:
            num = min(num, self.swa_max_n)
            self.swa_count.load(float(num), self.session)

        # save the current network.
        self.snap_save()
        # Copy the swa weights into the current network.
        self.session.run(self.swa_load_op)
        if self.swa_recalc_bn:
            for _ in range(200):
                batch = next(data)
                self.session.run(
                    [self.loss, self.update_ops],
                    feed_dict={self.model.training: True,
                               self.planes: batch[0], self.probs: batch[1],
                               self.winner: batch[2]})

        # Deprecated: do not save lz weight to speed up training
        # swa_path = path + "-swa-" + str(int(num)) + "-" + str(steps) + ".txt"
        # self.save_leelaz_weights(swa_path)
        # self.logger.info("Wrote averaged network to {}".format(swa_path))

        # restore the saved network.
        self.snap_restore()

    def snap_restore(self):
        # Restore variables in the current graph from the snapshot.
        self.session.run(self.restore_op)

    def snap_save(self):
        # Save a snapshot of all the variables in the current graph.
        if not hasattr(self, 'save_op'):
            save_ops = []
            rest_ops = []
            for var in self.model.weights:
                if isinstance(var, str):
                    var = tf.get_default_graph().get_tensor_by_name(var)
                name = var.name.split(':')[0]
                v = tf.Variable(var, name='save/' + name, trainable=False)
                save_ops.append(tf.assign(v, var))
                rest_ops.append(tf.assign(var, v))
            self.save_op = tf.group(*save_ops)
            self.restore_op = tf.group(*rest_ops)
        self.session.run(self.save_op)
