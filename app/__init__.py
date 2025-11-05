from flask import Flask
from .routes import main

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    
    # Register blueprints
    app.register_blueprint(main)
    
    return app
