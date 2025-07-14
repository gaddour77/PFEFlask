from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from .api.routes import register_routes
from .utils.scheduler import init_scheduler
from .utils.logger import logger
from .core.yolo_manager import YoloManager
from .core.yolo import Yolotrainer
def create_app():
    try:
        app = Flask(__name__, static_folder='../static')
        socketio = SocketIO(app, cors_allowed_origins="*")
        
        # Initialize core components
        #data_fetcher = YFinanceDataFetcher()
       # news_fetcher = FinnhubNewsFetcher(Config.FINNHUB_API_KEY)
        #asset_manager = AssetManager(data_fetcher, socketio)
        yolo_manager = YoloManager()
        yolo = Yolotrainer()
        # Register routes and start scheduler
        
        register_routes(app,socketio,yolo_manager,Yolotrainer)
        logger.info("Application initialized successfully")
        return app, socketio
    except Exception as e:
        logger.error(f"Failed to create application: {str(e)}")
        raise