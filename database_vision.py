"""
    Name Design: database_vision.py

    Author: Logan Fortune

    Email: logan.fortune@orange.fr

    License: Open Source

    Client: Wintics

    Date: October 2020
"""
import json
import os


class DatabaseVision:
    """
        This class stores the data from the "setup.json" completed by the client

        Class DatabaseVision:

            src_path_videos : file path for videos (mp4)
            src_path_params : file path for parameters (JSON)
            name_files_video : all the names of the videos
            name_files_params_color : the name of the JSON that stores the parameters for color
            name_files_params_zone : the name of the JSON that stores the parameters for the zones
            object_to_detect : the names of all objects like ["car", "truck", "motorcycle", "bus"]
            id_object_to_detect : list with all the id objects that must be detected
            video_class : list of the different localization ["highway_1", "highway_2"]
            rect : dict self.rect[video_class] = rect
            id_color : dict self.id_color[id_object] => color_id (car = blue, bus = yellow, truck = black)
    """

    def __init__(self, setup_file):
        # Read the json file
        with open(setup_file) as json_file:
            data = json.load(json_file)
        # Store files names
        # Store the labels to color
        # Store the zones
        assert data.get("database_vision", None) is not None
        self.src_path_videos = data["database_vision"].get("src-path-videos", None)
        assert self.src_path_videos is not None
        self.src_path_params = data["database_vision"].get("src-path-params", None)
        assert self.src_path_params is not None
        self.name_files_video = data["database_vision"].get("name-files-video", None)
        assert self.name_files_video is not None
        self.name_files_params_color = data["database_vision"].get("name-file-param-color", None)
        assert self.name_files_params_color is not None
        self.name_files_params_zone = data["database_vision"].get("name-file-param-zone", None)
        assert self.name_files_params_zone is not None
        self.check_files_consistency()
        # Computer Vision Parameters
        # What to detect ?
        self.object_to_detect = data["database_vision"].get("object-to-detect", None)
        assert self.object_to_detect is not None
        self.id_object_to_detect = data["database_vision"].get("id-object-to-detect", None)
        assert self.id_object_to_detect is not None
        # Where to detect ?
        self.video_class = data["database_vision"].get("video-class", None)
        assert self.video_class is not None
        self.rect = dict()
        self.get_rectangle_info()
        # Color Setup
        self.id_color = dict()
        self.get_color_info()

    def check_files_consistency(self):
        """ Check if the files exist """
        for video_file_names in self.name_files_video:
            assert os.path.isfile(self.src_path_videos + video_file_names)
        assert os.path.isfile(self.src_path_params + self.name_files_params_color)
        assert os.path.isfile(self.src_path_params + self.name_files_params_zone)

    def get_rectangle_info(self):
        """ Get rectangle info according to the class of the video
        (example of possible class of the video: highway_1, highway_2) """
        with open(self.src_path_params + self.name_files_params_zone) as f_color:
            data = json.load(f_color)
        for video_class_type in self.video_class:
            rect = data.get(video_class_type, None)
            if rect is None:
                print("Program Error (class DatabaseVision)(get_rectangle_info): One video class has not a rectangle "
                      "defined.")
                exit(-1)
            else:
                self.rect[video_class_type] = rect

    def get_color_info(self):
        """ Get the color that must be used according to the id of the object detected. """
        with open(self.src_path_params + self.name_files_params_color) as f_color:
            data = json.load(f_color)
        for id_object in self.id_object_to_detect:
            color_id = data.get(str(id_object), None)
            if color_id is None:
                print("Program Error (class DatabaseVision)(get_color_info): One id has not a color.")
                exit(-1)
            else:
                self.id_color[id_object] = color_id
