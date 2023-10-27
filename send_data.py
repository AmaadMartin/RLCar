import socket
from picarx import Picarx
import io
import socket
import picamera
from picamera.array import PiRGBArray
import sys
from tqdm import tqdm


def get_image():

    # Create a stream to hold the image data
    print("Capturing image")
    stream = io.BytesIO()

    # Initialize Picamera and capture an image to the stream
    with picamera.PiCamera() as camera:
        camera.resolution = (100, 100)  # Set your desired resolution
        camera.capture(stream, format='png')

    # Get the image data as a byte array
    image_data = stream.getvalue()

    return image_data


def main():
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Server address and port
    server_address = ('0.0.0.0', 48621)  # Replace 'server_ip_address' with the actual IP address of the server
    client_socket.bind(server_address)
    client_socket.listen(1)
    connection, _ = client_socket.accept()

    camera = picamera.PiCamera()
    camera.resolution = (100, 100)
    camera.framerate = 24
    rawCapture = PiRGBArray(camera, size=camera.resolution)  

    while True:
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            # Get image data
            image = frame.array

            # turn image into bytes
            image = image.tobytes()

            print(f"Sending image data to the server")
            # Send image data
            
            connection.sendall(image)
            # send_image(image, client_socket)
    
if __name__ == '__main__': 
    main()