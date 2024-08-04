import logging
import os
import sys

logger_initialized = []


def setup_logger(name='default', output=None):
    logger = logging.getLogger(name)
    if logger in logger_initialized:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if output is not None:
        if output.endswith('.txt') or output.endswith('.log'):
            filename = output
        else:
            filename = os.path.join(output, 'log.txt')
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter())
        logger.addHandler(fh)
    logger_initialized.append(logger)
    return logger
