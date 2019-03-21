#!/usr/bin/env python3
import os
import argparse

from tfprocess import TFProcess
from utils.logger import DefaultLogger


def net_to_model(file_path):
    with open(file_path, 'r') as f:
        weights = []
        for e, line in enumerate(f):
            if e == 0:
                # Version
                logger.info("Version {}".format(line.strip()))
                if line != '1\n':
                    raise ValueError("Unknown version {}".format(line.strip()))
            else:
                weights.append(list(map(float, line.split(' '))))
            if e == 2:
                channels = len(line.split(' '))
                logger.info("Channels {}".format(channels))

        blocks = e - (4 + 14)
        if blocks % 8 != 0:
            raise ValueError("Inconsistent number of weights in the file")
        blocks //= 8
        logger.info("Blocks {}".format(blocks))

    tfprocess = TFProcess()
    # set the blocks and filters as target weight
    tfprocess.RESIDUAL_BLOCKS = blocks
    tfprocess.RESIDUAL_FILTERS = channels

    tfprocess.init(batch_size=1, gpus_num=1)
    tfprocess.replace_weights(weights)
    path = os.path.join(os.getcwd(), "leelaz-model")
    save_path = tfprocess.saver.save(tfprocess.session, path, global_step=0)
    logger.info("TF model saved in {}".format(save_path))


def model_to_net(file_path):
    tfprocess = TFProcess()
    # TODO: detect model filter and block size automatically
    tfprocess.RESIDUAL_BLOCKS = 10
    tfprocess.RESIDUAL_FILTERS = 128
    tfprocess.init(batch_size=1, gpus_num=1)
    tfprocess.restore(file_path)
    tfprocess.save_leelaz_weights(file_path + "_lz.txt")
    logger.info("Leelaz weight saved in {}".format("leelaz-weight"))


if __name__ == '__main__':
    # usage: python -m tools.net_to_model [options]
    parser = argparse.ArgumentParser(description="lz net and tf model conversion")
    parser.add_argument("-f", "--file", help="path to lz weight or tf model")
    parser.add_argument("-r", "--reverse", default=False, action="store_true",
                        help="switch from net_to_model to model_to_net")
    args = parser.parse_args()
    logger = DefaultLogger
    if args.reverse:
        logger.info("convert tf model to lz net...")
        model_to_net(args.file)
    else:
        logger.info("convert lz net to tf model...")
        net_to_model(args.file)
