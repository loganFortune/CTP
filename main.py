"""
    Name Design: main.py

    Author: Logan Fortune

    Email: logan.fortune@orange.fr

    License: Open Source

    Client: Wintics

    Date: October 2020
"""
import os
import sys
from database_vision import DatabaseVision
from wvideo import process_video, post_process

SETUP_FILE = "./setup.json"


def check_system():
    """
    This function checks that the system is correctly prepared before doing anything.

    :return: Bool
    """
    # Check Python Version
    if sys.version_info.major != 3:
        # Python 2
        print("You are using 'python2' which is outside the scope of the test provided.")
        return False
    # Check "setup.json" validity
    assert os.path.isfile(SETUP_FILE)
    return True


def main():
    """
    This function is the backbone of the detection program:

    - Get the database (videos + color/rectangle info)
    - Process the database (NN model + computer vision)

    :return: Bool
    """
    # Get the input from the client
    database_vision = DatabaseVision(SETUP_FILE)
    # Process the videos
    process_video(database_vision)
    # Post-processing of the videos
    post_process(database_vision)
    return True


if __name__ == '__main__':
    print('Wintics (Case Study) by Logan Fortune.')
    assert check_system()
    main()
