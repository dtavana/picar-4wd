"""
Helper for tuning the calibration constants in movement.py.

  forward: drives forward the given cm using FORWARD_CM_PER_SECOND. Measuring the actual distance
           and scale FORWARD_CM_PER_SECOND by (actual / requested).
  reverse: same as forward but for REVERSE_CM_PER_SECOND.
  turn:    makes the given number of 90 degree right turns (negative for left) using TURN_90_TIME.
           Using 4 to check the car ends up facing the way it started, and adjusting TURN_90_TIME.
"""
import argparse
import picar_4wd as fc
from . import movement

def main():
    parser = argparse.ArgumentParser(description="Calibrate navigation movement constants")
    parser.add_argument("action", choices=["forward", "reverse", "turn"])
    parser.add_argument("amount", type=float, help="cm for forward/reverse, number of 90 degree turns for turn")
    args = parser.parse_args()

    fc.servo.set_angle(0)
    try:
        if args.action == "forward":
            moved = movement.forward(args.amount)
            print(f"Requested {args.amount:.1f} cm, estimated {moved:.1f} cm (less means an emergency stop)")
        elif args.action == "reverse":
            movement.reverse(args.amount)
        else:
            # Turn one quarter at a time, movement.turn() treats a full circle as no turn at all
            quarter_turns = int(args.amount)
            for _ in range(abs(quarter_turns)):
                movement.turn(1 if quarter_turns > 0 else -1)
    finally:
        fc.stop()

if __name__ == "__main__":
    main()
