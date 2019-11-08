from flask import Flask
from flasgger import Swagger

def create_app(*, config_object) -> Flask:
    """Create a flask app instance."""

    app = Flask(__name__)
    swagger = Swagger(app)
    app.config.from_object(config_object)

    return app