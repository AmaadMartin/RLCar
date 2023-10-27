import socket
import readchar
from time import sleep

canSend = True

def send_action(action):
    global canSend
    canSend = False
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to a specific address and port
    server_address = ('172.26.177.26', 48621)  # 0.0.0.0 means listen on all available interfaces
    server_socket.connect(server_address)
    server_socket.sendall(action.encode())
    server_socket.shutdown(socket.SHUT_WR)
    server_socket.close()
    print("Sent action: ", action)
    canSend = True

def main():
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
                send_action('forward')
            elif key == 'a':
                send_action('turn left')
            elif key == 'd':
                send_action('turn right')
            elif key == 's':
                send_action('stop')
        sleep(0.1)
        

if __name__ == '__main__':
    main()