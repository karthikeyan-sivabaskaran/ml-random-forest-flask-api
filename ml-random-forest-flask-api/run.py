from api.app import create_app
from api.config import DevelopmentConfig, ProductionConfig
from api.routes import api_routes

app = create_app(
    config_object=DevelopmentConfig)

app.register_blueprint(api_routes)

app.config["JSON_SORT_KEYS"] = False

# if __name__  == '__main__':
#     app.debug = True
#     app.run(host="0.0.0.0",port=5777)