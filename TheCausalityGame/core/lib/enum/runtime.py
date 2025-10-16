from enum import Enum


class RuntimeMode(str, Enum):
    PROD = "prod"
    DEV = "dev"


class RuntimeDebugLevel(int, Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
