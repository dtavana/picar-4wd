import picar_4wd as fc
import sys


def main(servo_offset=0):
    print("Servo angle tester")
    print("Enter an angle between -90 and 90")
    print("Use Ctrl+C to stop at any time")
    print("------------------------------")

    fc.servo.offset = servo_offset

    while True:
        value = input("Angle: ").strip()
        try:
            angle = int(value)
        except ValueError:
            print("Please enter a valid number")
            continue

        print(f"Setting servo to {angle} degrees")
        fc.servo.set_angle(angle)

if __name__ == "__main__":
    servo_offset = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    try:
        main(servo_offset)
    except KeyboardInterrupt:
        print("\nStopping")
    finally:
        fc.stop()
        CENTER_SERVO_POSITION.set_angle()