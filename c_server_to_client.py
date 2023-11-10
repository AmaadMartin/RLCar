# #!/usr/bin/env python3

from robot_hat.utils import reset_mcu
from picarx import Picarx
from time import sleep, time, strftime, localtime
import readchar
import socket

IP = '0.0.0.0'  
PORT = 48621

class ClientReciever:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_address = (self.ip, self.port)  # Replace 'server_ip_address' with the actual IP address of the server
        self.client_socket.bind(server_address)
        self.client_socket.listen(1)
        self.connection, _ = self.client_socket.accept()
        print(f"Accepted connection from the server: {server_address}")
    
    def recieveAction(self):
        try:
            action = self.connection.recv(1024).decode()
            print(f"Received data from the server: {action}")
        except:
            action = 'stop'
        return action
    
    def close(self):
        self.client_socket.close()

reset_mcu()
sleep(0.2)

px = Picarx()
SPEED = 10

def move(operate:str):
    speed = SPEED

    if operate == 'stop':
        px.stop()  
    else:
        if operate == 'forward':
            px.set_dir_servo_angle(0)
            px.forward(speed)
        elif operate == 'backward':
            px.set_dir_servo_angle(0)
            px.backward(speed)
        elif operate == 'turn left':
            px.set_dir_servo_angle(-30)
            px.forward(speed)
        elif operate == 'turn right':
            px.set_dir_servo_angle(30)
            px.forward(speed)

def recieveAction():
    
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Server address and port
    server_address = ('0.0.0.0', 48621)  # Replace 'server_ip_address' with the actual IP address of the server
    client_socket.bind(server_address)

    client_socket.listen(1)

    print(f"Waiting for connection from the server: {server_address}")

    # Accept a connection from the server
    connection, client_address = client_socket.accept()
    print(f"Accepted connection from the server: {client_address}")

    # Receive data from the server
    action = connection.recv(1024).decode()

    if len(action) > 0: print(f"Received data from the server: {action}")

    # print(f"Connecting to the server: {server_address}")
    # # Connect to the server 
    # client_socket.connect(server_address)
    # print(f"Connected to the server: {server_address}")

    # # Receive data from the server
    # action = client_socket.recv(1024).decode()
    # print(f"Received data from the server: {action}")

    # # Close the connection
    # client_socket.close()
    
    return action


def main():
    status = 'stop'
    reciever = ClientReciever(IP, PORT)

    while True:
        # recieve action from server
        action = reciever.recieveAction()
        move(action)
        sleep(0.1)

if __name__ == '__main__':
    main()
