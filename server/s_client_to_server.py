import socket
from time import sleep
from tqdm import tqdm
import cv2
import numpy as np
from math import sqrt

IP = '172.26.177.26'
PORT = 48622
RESOLTUION = (224, 224)

class ServerReciever:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.ip, self.port))
    
    def recieveBytes(self):
        expected_data_size = RESOLTUION[0] * RESOLTUION[1] * 3
        image = b''
        while len(image) < expected_data_size:
            chunk = self.socket.recv(expected_data_size - len(image))
            if not chunk:
                break
            image += chunk
        # recieve reward
        
        reward = self.socket.recv(1024).decode()

        return image, reward
    
    def convertBytesToImage(self, image):
        image_array = np.frombuffer(image, dtype=np.uint8).reshape(RESOLTUION[0], RESOLTUION[1], 3)
        return image_array
    
    def showImage(self, image):
        cv2.imshow('BGR Image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
        return False
    
    def close(self):
        self.socket.close()


def main():
    connection = ServerReciever(IP, PORT)
    while True:
        image, reward = connection.recieveBytes()
        print(reward)
        image = connection.convertBytesToImage(image)
        connection.showImage(image)

if __name__ == '__main__':
    main()