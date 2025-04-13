import tkinter as tk
from gui import CrowdAnalysisApp
import os
import sys
import logging

def setup_logging():
    """Configure logging for the application"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'crowd_analysis.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main entry point for the application"""
    logger = setup_logging()
    logger.info("Starting Crowd Analysis Application")
    
    try:
        # Check if running on Raspberry Pi
        is_pi = False
        try:
            with open('/proc/device-tree/model', 'r') as f:
                is_pi = 'raspberry pi' in f.read().lower()
        except:
            is_pi = os.uname().machine in ('armv7l', 'aarch64')
        
        if is_pi:
            logger.info("Running on Raspberry Pi")
            # Set display for SSH
            os.environ['DISPLAY'] = ':0'
            
            # Configure for Pi's display
            root = tk.Tk()
            root.attributes('-fullscreen', True)  # Fullscreen mode for Pi
        else:
            logger.info("Running on non-Pi system")
            root = tk.Tk()
            root.geometry("1200x800")
        
        # Create and run the application
        app = CrowdAnalysisApp(root)
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()