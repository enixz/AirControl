"""Shared resource and writable-data paths for source and packaged builds."""

import os
import sys

APP_NAME = "AirControl"


def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource_root():
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return project_root()


def resource_path(*parts):
    return os.path.join(resource_root(), *parts)


def writable_data_dir():
    override = os.environ.get("AIRCONTROL_DATA_DIR")
    if override:
        root = os.path.abspath(os.path.expanduser(override))
    elif getattr(sys, "frozen", False):
        appdata = (
            os.environ.get("APPDATA")
            or os.environ.get("LOCALAPPDATA")
            or os.path.expanduser("~")
        )
        root = os.path.join(appdata, APP_NAME)
    else:
        root = project_root()
    os.makedirs(root, exist_ok=True)
    return root


def data_path(*parts):
    path = os.path.join(writable_data_dir(), *parts)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path

