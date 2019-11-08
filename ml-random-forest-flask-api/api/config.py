import pathlib
import os

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent

class Config:
    DEBUG = False
    TESTING = False
    # CSRF_ENABLED = True
    # SECRET_KEY = 'this-really-needs-to-be-changed'
    SERVER_PORT = 6000


class ProductionConfig(Config):
    DEBUG = False
    SERVER_ADDRESS: os.environ.get('SERVER_ADDRESS', '0.0.0.0')
    SERVER_PORT: os.environ.get('SERVER_PORT', '6000')


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True

class TestingConfig(Config):
    TESTING = True