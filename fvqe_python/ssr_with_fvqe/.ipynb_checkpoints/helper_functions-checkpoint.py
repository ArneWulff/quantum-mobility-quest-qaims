"""
helper_function.py

Helper functions for time formatting in the F-VQE module.

This module provides utility functions to format elapsed time into a
standardized string format (HH:MM:SS.ss). It is primarily used to convert
timestamps or time differences into human-readable strings.
"""

import time

def timestr(x: float) -> str:
    return f"{int(x)//3600:02}:{(int(x)%3600)//60:02}:{int(x)%60:02}.{int(round((x%1)*100))}"


def timediffstr(t0) -> str:
    return timestr(time.time() - t0)