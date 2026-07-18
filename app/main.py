"""
This file is deprecated.
Please run `main_ui.py` to start the AirControl application.
"""
import sys


def main():
    print("=====================================================", file=sys.stderr)
    print("WARNING: main.py is deprecated and no longer maintained.", file=sys.stderr)
    print("Please run `main_ui.py` to start the application with ", file=sys.stderr)
    print("the full UI and Orchestrator architecture.", file=sys.stderr)
    print("=====================================================", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
