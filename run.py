# run.py
import os
import webbrowser
import time
from app import app as flask_app
from threading import Thread
import warnings

# Define color constants
class Colors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'

# Suppress TensorFlow and Keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show errors
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow.keras')

# Print statements to trace execution
print(f"{Colors.BLUE}Starting the Flask application...{Colors.RESET}")

# Ensure the model file is trained and saved
if not os.path.exists('app/sentiment_model.h5'):
    print(f"{Colors.RED}Error: The model file 'app/sentiment_model.h5' was not found. Please run 'train_model.py' first.{Colors.RESET}")
    raise FileNotFoundError("The model file 'app/sentiment_model.h5' was not found. Please run 'train_model.py' first.")
else:
    print(f"{Colors.GREEN}Model file 'app/sentiment_model.h5' found.{Colors.RESET}")

# Import main after the model file is confirmed to exist
print(f"{Colors.YELLOW}Importing the main application logic from 'app.main'...{Colors.RESET}")
import app.main

def open_browser():
    """
    Open the default web browser to the Flask app URL.
    """
    url = "http://127.0.0.1:5000"  # The URL of the Flask app
    webbrowser.open(url)  # Open the URL in the default web browser

if __name__ == '__main__':
    """
    Run the Flask application and open it in the default web browser.
    """
    
    # Function to start the Flask server
    def run_server():
        """
        Start the Flask application server.
        """
        flask_app.run(debug=True, use_reloader=False)
    
    # Start the Flask app in a separate thread to avoid blocking the main thread
    server_thread = Thread(target=run_server)
    server_thread.start()  # Start the server thread

    # Wait a few seconds to ensure the Flask server is up and running
    time.sleep(2)

    # Open the default web browser to the Flask app URL
    open_browser()
