import sys

from encoderfile import run_cli  # type: ignore

if __name__ == "__main__":
    args = ["encoderfile"] + sys.argv[1:]

    try:
        run_cli(args)
    except RuntimeError as e:
        print(e, file=sys.stderr)
