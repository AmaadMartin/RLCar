import asyncio
import websockets
from picarx import Picarx
from time import sleep, time
import picamera
from picamera.array import PiRGBArray
import threading
from tqdm import tqdm
import collections

"""
Some improvements to make:
    - could combine process that is capturing image and greyscale data into one.
    - encode the string checking.
    - Consider compressing image before sending
"""

RESOLUTION = (224, 224)
FRAMERATE = 24
MAX_MOVES = 3
action_map = {0: "left", 1: "right", 2: "forward", 3: "backward", 4: "stop"}


class ImageCapture:
    def __init__(self, camera, resolution=RESOLUTION, framerate=FRAMERATE):
        self.last_image = None
        self.camera = camera
        self.camera.resolution = resolution
        self.capture_thread = threading.Thread(target=self.capture_images)
        self.capture_thread.start()

    def capture_images(self):
        rawCapture = PiRGBArray(self.camera, size=self.camera.resolution)
        for frame in self.camera.capture_continuous(
            rawCapture, format="bgr", use_video_port=True
        ):
            image = frame.array
            image = image.tobytes()
            self.last_image = image
            rawCapture.truncate(0)


class Car:
    def __init__(self, picar):
        self.px = picar
        self.reset_flag = False
        self.gs_data = None
        self.move_list = collections.deque(maxlen=MAX_MOVES)
        self.reset_thread = threading.Thread(target=self.check_reset)
        self.reset_thread.start()

    def check_reset(self):
        # Needs to be fixed
        while True:
            self.gs_data = self.px.get_grayscale_data()
            if not self.reset_flag and any(self.gs_data):
                self.reset_flag = True
                self.stop()
            sleep(0.1)

    def stop(self):
        self.px.stop()

    def move_reverse_for_seconds(self, operate, speed, seconds):
        if operate == 2:  # forward
            px.set_dir_servo_angle(0)
            px.backward(speed)
        elif operate == 3:  # backward
            px.set_dir_servo_angle(0)
            px.forward(speed)
        elif operate == 0:  # left
            px.set_dir_servo_angle(-30)
            px.backward(speed)
        elif operate == 1:  # right
            px.set_dir_servo_angle(30)
            px.backward(speed)
        sleep(seconds)
        self.stop()

    def move(self, operate, speed):
        if operate == 2:  # forward
            px.set_dir_servo_angle(0)
            px.forward(speed)
        elif operate == 3:  # backward
            px.set_dir_servo_angle(0)
            px.backward(speed)
        elif operate == 0:  # left
            px.set_dir_servo_angle(-30)
            px.forward(speed)
        elif operate == 1:  # right
            px.set_dir_servo_angle(30)
            px.forward(speed)

    def reset(self):
        assert self.reset_flag
        self.stop()

        anchor = time()

        if len(self.move_list) > 0:
            status, speed, move_time = self.pop()
            self.move_reverse_for_seconds(status, speed, anchor - move_time)

        for i in range(len(self.move_list) - 2, -1, -1):
            status, speed, move_time = self.move_list[i]
            _, _, next_move_time = self.move_list[i + 1]
            self.move_reverse_for_seconds(status, speed, next_move_time - move_time)

        self.move_list.clear()
        self.reset_flag = False


px = Picarx()
camera = picamera.PiCamera()
image_capture = ImageCapture(camera)
car = Car(px)


async def start(websocket, path):
    async for message in websocket:
        print(f"Received message: {message}")
        if message == "start":
            # Send back the current image.
            image = image_capture.last_image
            await websocket.send(image)
        elif message == "reset":
            # Execute the reset on the car. Stop. Then send current state.
            car.reset()
            image = image_capture.last_image
            await websocket.send(image)
        else:
            # Execute the action on the car. Then send current state.
            car.move(message)
            res = image_capture.last_image
            res += car.gs_data
            res += str(car.reset_flag)
            await websocket.send(res)


start_server = websockets.serve(start, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
