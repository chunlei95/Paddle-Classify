import logging
import os
import sys

import colorlog

logger_initialized = []

log_colors_config = {
    'DEBUG': 'white',  # cyan white
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}


def setup_logger(name='Default', output=None):
    logger = logging.getLogger(name)
    if logger in logger_initialized:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    color_formatter = colorlog.ColoredFormatter("%(log_color)s[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                                                datefmt="%m/%d %H:%M:%S",
                                                log_colors=log_colors_config)
    ch.setFormatter(color_formatter)
    # ch.setFormatter(formatter)
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
