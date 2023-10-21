'''
    Environment Reset 

'''
from picarx import Picarx
from time import sleep
import readchar
import threading

px = Picarx()

reset_flag = False  # Initialize a flag to control loop restart

def check_reset():
    gs_data = px.get_grayscale_data()
    if(any(r < 50 for r in gs_data)):
        print('found black!')
        # px.backward(5)
        # sleep(1)
        # px.stop()
        return True
    return False


def check_reset_loop():
    global reset_flag
    while True:
        if check_reset():
            reset_flag = True  # Set the flag to True if reset condition is detected
            px.backward(5)
            sleep(1)
            px.stop()
            reset_flag = False
        sleep(0.1)  # Adjust the sleep interval as needed



            
def move(operate:str, speed):
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


def main():
    global reset_flag
    reset_thread = threading.Thread(target=check_reset_loop)
    reset_thread.daemon = True  # The thread will exit when the main program exits
    reset_thread.start()
    while True:
        if reset_flag:
            continue  # Restart the loop
        key = readchar.readkey().lower()
        if reset_flag:
            continue
        speed = 0
        status='stop'
        if key in ('wsadfop'):
            # throttle
            if key == 'o':
                if speed <=90:
                    speed += 10           
            elif key == 'p':
                if speed >=10:
                    speed -= 10
                if speed == 0:
                    status = 'stop'
            # direction
            elif key in ('wsad'):
                if speed == 0:
                    speed = 10
                if key == 'w':
                    # Speed limit when reversing,avoid instantaneous current too large
                    if status != 'forward' and speed > 60:  
                        speed = 60
                    status = 'forward'
                elif key == 'a':
                    status = 'turn left'
                elif key == 's':
                    if status != 'backward' and speed > 60: # Speed limit when reversing
                        speed = 60
                    status = 'backward'
                elif key == 'd':
                    status = 'turn right' 
            # stop
            elif key == 'f':
                status = 'stop'
            # move 
            move(status, speed)  
        # quit
        elif key == readchar.key.CTRL_C:
            print('\nquit ...')
            px.stop()
            break 

        sleep(0.1)

if __name__=='__main__':
    try:
        # reset_thread = threading.Thread(target=check_reset_loop)
        # reset_thread.daemon = True  # The thread will exit when the main program exits
        # reset_thread.start()
        # main_thread = threading.Thread(target=main)
        # main_thread.daemon = True  # The thread will exit when the main program exits
        # main_thread.start()
        main()
    finally:
        px.stop()