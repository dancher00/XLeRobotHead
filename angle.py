from phone_server import PhoneServer
import time

server = PhoneServer()
server.run()


while True:
    time.sleep(0.5)
    angles = server.get_angles()
    print(f"Roll: {angles['roll']:.2f}°")
    print(f"Pitch: {angles['pitch']:.2f}°")
    print(f"Yaw: {angles['yaw']:.2f}°")
