import socket
from time import sleep
from tqdm import tqdm

def main():
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to a specific address and port
    server_address = ('172.26.177.26', 48621)  # 0.0.0.0 means listen on all available interfaces
    server_socket.connect(server_address)
    for i in tqdm(range(100)):
        # Get image data
        image = server_socket.recv(10000)

        # Show image
        # with open('captured_image.png', 'wb') as f:
        #     f.write(image)

if __name__ == '__main__':
    main()