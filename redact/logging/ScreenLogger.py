from pynput import mouse
from PIL import ImageGrab

class ScreenLogger:
    def __init__(self):
        self.ctr = 0  # counter for naming
        self.listener = mouse.Listener(on_click=self.log)
        print("listener created")

    def log(self, x, y, button, pressed):
        print(str(button) + ", at " + str((x, y)))
        if pressed: # do nothing on release
            bbox = (1920, 0, 3840, 1080) # for linux
            screenshot = ImageGrab.grab(bbox)
            screenshot.save("ScreenLog/screenshot" + str(self.ctr) + ".png")
            self.ctr += 1

    def start(self):
        self.listener.start()
        self.listener.join()

# Test logger
logger = ScreenLogger()
logger.start()
