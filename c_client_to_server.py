import socket
from picarx import Picarx
import io
import socket
import picamera
from picamera.array import PiRGBArray
import sys
from tqdm import tqdm
from agent import Agent

IP = '0.0.0.0'
PORT = 48622
RESOLUTION = (224, 224)

class ClientSender:
    def __init__(self, agent, port):
        self.px = agent.px
        self.lightThreshold = agent.lightThreshold

        self.ip = '0.0.0.0'
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1)
        self.client_socket.bind((self.ip, self.port))
        self.client_socket.listen(1)
        self.connection, _ = self.client_socket.accept()

        self.camera = picamera.PiCamera()
        self.camera.resolution = RESOLUTION
        self.camera.framerate = 24

    
    def sendImageReward(self, image, reward):
        totalSent = 0
        while totalSent < RESOLUTION[0] * RESOLUTION[1] * 3:
            sent = self.client_socket.send(image[totalSent:])
            if sent == 0:
                raise RuntimeError("Socket connection broken")
            totalSent += sent
        print(self.client_socket.send(reward.encode()))

    def startSendProcess(self):
        rawCapture = PiRGBArray(self.camera, size=self.camera.resolution)  
        for frame in self.camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            image = frame.array
            image = image.tobytes()

            gs_data = self.px.get_grayscale_data()
            reward = 0 if any(r < self.lightThreshold for r in gs_data) else 1

            self.sendImageReward(image, str(reward))
            rawCapture.truncate(0)

def main():
    px = Picarx()
    lightThreshold = 20
    agent = Agent(px, lightThreshold)
    sender = ClientSender(agent, PORT)
    sender.startSendProcess()
    
if __name__ == '__main__': 
    main()