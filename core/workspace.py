import importlib


def from_config(config_path):
    return importlib.import_module(config_path)


if __name__ == '__main__':
    cfg = from_config('D:/PycharmProjects/Paddle-Classify/Paddle-Classify/configs/VAN.py')
