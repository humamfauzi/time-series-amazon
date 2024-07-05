from enum import Enum
class DataPart(Enum):
    train = 1
    valid = 2
    test = 3

    @classmethod
    def all(cls):
        return [
            cls.train.name,
            cls.valid.name,
            cls.test.name,
        ]

    @classmethod
    def train_eval(cls):
        return [
            cls.train.name,
            cls.valid.name,
        ]