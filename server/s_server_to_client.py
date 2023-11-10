import socket
import readchar
from time import sleep

IP = '172.26.177.26'
PORT = 48621

canSend = True

class ServerSender:
    def __init__(self):
        self.ip = IP
        self.port = PORT
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.ip, self.port))
        print(f"Connected to the server: {self.ip}:{self.port}")

    def sendAction(self, action):
        try:
            self.socket.sendall(action.encode())
        except:
            print("Error sending action")

    def close(self):
        self.socket.shutdown(socket.SHUT_WR)
        self.socket.close()
        
def send_action(action, sender):
    global canSend
    canSend = False
    sender.sendAction(action)
    canSend = True

def main():
    sender = ServerSender()
    global canSend
    while True:
        if not canSend:
            continue
        # readkey
        key = readchar.readkey().lower()
        # direction
        if key in ('wsad'):
            if key == 'w':  
                # launch thread to send forward action
                send_action('forward', sender)
            elif key == 'a':
                send_action('turn left', sender)
            elif key == 'd':
                send_action('turn right', sender)
            elif key == 's':
                send_action('stop', sender)
        sleep(0.1)
        

if __name__ == '__main__':
    main()