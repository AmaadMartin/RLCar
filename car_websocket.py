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
    - Going to do a force stop, but might be doing updates to model we don't want to do fix!
    - Error catching, saving model etc.
    - Fix the way we are resetting
    - ISSUE: changes at last second!!!
"""

# RESOLUTION = (224, 224)
RESOLUTION = (640, 480)
FRAMERATE = 60
THRESHOLD = 50
MAX_MOVES = 3
SPEED = 10
MOVE_TIME = 0.25
DEBUG = False
action_map = {0: "left", 1: "right", 2: "forward", 3: "backward", 4: "stop"}


class ImageCapture:
    def __init__(self, camera, resolution=RESOLUTION, framerate=FRAMERATE):
        self.last_image = None
        self.camera = camera
        self.camera.resolution = resolution
        self.camera.framerate = framerate

    def start_capture(self):
        self.capture_thread = threading.Thread(target=self.capture_images)
        self.capture_thread.start()

    def capture_images(self):
        rawCapture = PiRGBArray(self.camera, size=self.camera.resolution)
        for frame in self.camera.capture_continuous(
            rawCapture, format="rgb", use_video_port=True
        ):
            image = frame.array
            image = image.tobytes()
            self.last_image = image
            rawCapture.truncate(0)


class Car:
    def __init__(self, picar):
        self.px = picar
        self.reset_flag = 0
        self.gs_data = None
        self.move_list = collections.deque(maxlen=MAX_MOVES)

    def start_reset_thread(self):
        self.reset_thread = threading.Thread(target=self.check_reset)
        self.reset_thread.start()

    def check_reset(self):
        # Needs to be fixed
        while True:
            self.gs_data = self.px.get_grayscale_data()
            if self.reset_flag == 0 and any(r < THRESHOLD for r in self.gs_data):
                self.reset_flag = 1
                self.stop()
            sleep(0.1)

    def stop(self):
        # print("Stopping")
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
        operate = int(operate)
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
        self.move_list.append((operate, speed, time()))
        sleep(MOVE_TIME)

    def reset(self):
        self.stop()
        # print("starting reset")

        anchor = time()

        while len(self.move_list) > 0:
            status, speed, move_time = self.move_list.pop()
            # if anchor - move_time < 0.15:
            #     anchor = time()
            #     continue
            self.move_reverse_for_seconds(status, speed, anchor - move_time)
            anchor = move_time

        # if len(self.move_list) > 0:
        #     status, speed, move_time = self.move_list.pop()
        #     self.move_reverse_for_seconds(status, speed, anchor - move_time)

        # for i in range(len(self.move_list) - 2, -1, -1):
        #     status, speed, move_time = self.move_list[i]
        #     _, _, next_move_time = self.move_list[i + 1]
        #     self.move_reverse_for_seconds(status, speed, next_move_time - move_time)

        # self.move_list.clear()
        self.reset_flag = 0
        # print("done resetting")


px = Picarx()
camera = picamera.PiCamera()
image_capture = ImageCapture(camera)
car = Car(px)
# move camera down
px.set_camera_servo2_angle(-27)


async def start(websocket, path):
    if DEBUG: print("Connected")
    image_capture.start_capture()
    car.start_reset_thread()

    async for message in websocket:
        message = message.decode("utf-8")
        if DEBUG: print(f"Received message: {message}")
        if str(message) == "start":
            # Send back the current image.
            image = image_capture.last_image
            await websocket.send(image)
        elif str(message) == "reset":
            # Execute the reset on the car. Stop. Then send current state.
            car.reset()
            image = image_capture.last_image
            await websocket.send(image)
        elif car.reset_flag != 1:
            # Execute the action on the car. Then send current state.
            car.move(message, speed=SPEED)
            res = image_capture.last_image

            # print(car.gs_data)
            for data in car.gs_data:
                # data_bytes = data.to_bytes(
                #     (data.bit_length() + 7) // 8, "big"
                # )  # Convert integer to bytes
                data_byte = data.to_bytes(1, "big")
                res += data_byte
            # print(car.reset_flag)
            reset_flag_byte = car.reset_flag.to_bytes(1, "big")
            await websocket.send(res + reset_flag_byte)
        else: 
            reset_flag_byte = car.reset_flag.to_bytes(1, "big")
            await websocket.send(res + reset_flag_byte)

start_server = websockets.serve(start, "0.0.0.0", 8765)
try:
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
finally:
    px.stop()
    camera.close()
