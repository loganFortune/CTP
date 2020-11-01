"""
    Name Design: wvideo.py

    Author: Logan Fortune

    Email: logan.fortune@orange.fr

    License: Open Source

    Client: Wintics

    Date: October 2020
"""
# ML
import torchvision
import torch
# Image Processing
import cv2
import numpy as np
# Files management
import os
import json


# Download Model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# Training dataset : https://github.com/nightrome/cocostuff
# Evaluation Mode
model.eval()
# Use CPU (constraint)
device = torch.device('cpu')


def process_video(database_vision, debug=False):
    """
        This function aims to get the all the detections in a certain zone provided by database_vision.

        Detection via fasterrcnn_resnet50_fpn:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values of ``x``
                            between ``0`` and ``W`` and values of ``y`` between ``0`` and ``H``

        - labels (``Int64Tensor[N]``): the predicted labels for each image

        - scores (``Tensor[N]``): the scores or each prediction

    :param database_vision: class DatabaseVision

    :return:
    """
    # Get the rectangle via database_vision.video_class and database_vision.rect
    # Get the color via database_vision.id_color
    for video_file_name in database_vision.name_files_video:

        video_to_process = cv2.VideoCapture(database_vision.src_path_videos + video_file_name)

        current_rect = get_video_type_by_collision(database_vision.rect, video_file_name)

        assert current_rect is not None

        height_rect = current_rect[0][0][1] - current_rect[0][2][1]
        width_rect = current_rect[0][1][0] - current_rect[0][0][0]
        assert height_rect > 0
        assert width_rect > 0

        # Case Study Custom Parameters !!

        # TODO(logan): check that the absolute_height_marge is not too large according to the image size...
        marge_rect_for_detection = 0.20  # 30% marge to get a rectangle that is enough to get an accurate detection
        absolute_height_marge = int(marge_rect_for_detection * height_rect / 2)
        absolute_width_marge = int(marge_rect_for_detection * width_rect / 2)

        # Debug : print(height_rect, width_rect)

        buffer_size = 10  # How many frames we skipped ?
        it_buffer = -1  # iterator to count the number of frames since the last processed.

        do_resizing_method = False

        # Files Management setup for screenshots
        time_stamp = -1
        lane, hour = filter_file_name(video_file_name)
        assert lane.isnumeric() and hour.isnumeric()

        # "screenshots/highway_"+lane+"/"+hour+"h/"+time_stamp+".png"
        try:
            base_file_name = "./screenshots/highway_"+lane+"/"+hour+"h/"
            os.makedirs(base_file_name)
        except FileExistsError:
            # directory already exists
            pass

        screenshots_results = {'screenshots': []}

        while video_to_process.isOpened():

            ret, frame = video_to_process.read()
            time_stamp += 1

            # Skip some frames
            it_buffer += 1
            if it_buffer == buffer_size:
                it_buffer = -1

            if ret is True:
                if it_buffer == -1:

                    # Take the region of interest with some margins
                    crop_img = frame[
                               current_rect[0][2][1] - absolute_height_marge:
                               current_rect[0][2][1] + height_rect + absolute_height_marge,
                               current_rect[0][0][0] - absolute_width_marge:
                               current_rect[0][0][0] + width_rect + absolute_width_marge
                               ]

                    # Debug :
                    # cv2.imshow("cropped", crop_img)

                    img_process = process_image(crop_img, do_resizing_method)

                    # run inference on the model and get detections
                    tensor_img = torch.FloatTensor([img_process])  # 32-bit floating point
                    with torch.no_grad():
                        tensor_img = tensor_img.to(device)
                        detections = model(tensor_img)[0]

                    boxes = detections["boxes"]  # boxes
                    labels = detections["labels"]  # labels
                    scores = detections["scores"]  # scores

                    tensor_filter = torchvision.ops.nms(boxes, scores, 0.5)  # IOU = AREA of Overlap / AREA of the Union

                    one_valid_box_found = False
                    for i in range(0, 3):
                        if labels[tensor_filter[i]] in database_vision.id_object_to_detect \
                                and scores[tensor_filter[i]] > 0.5:
                            one_valid_box_found = True
                            screenshots_results['screenshots'].append([
                                                    time_stamp,
                                                    tuple([int(boxes[tensor_filter[i]][0]), int(boxes[tensor_filter[i]][1])]),
                                                    tuple([int(boxes[tensor_filter[i]][2]), int(boxes[tensor_filter[i]][3])]),
                                                    database_vision.id_color[int(labels[tensor_filter[i]])]
                                                    ])
                            if debug is True:
                                resized = cv2.rectangle(img_process.transpose(1, 2, 0),
                                                    tuple([int(boxes[tensor_filter[i]][0]), int(boxes[tensor_filter[i]][1])]),
                                                    tuple([int(boxes[tensor_filter[i]][2]), int(boxes[tensor_filter[i]][3])]),
                                                    color=database_vision.id_color[int(labels[tensor_filter[i]])], thickness=2)
                    if one_valid_box_found and debug:
                        cv2.imshow("Filter", resized)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        with open(base_file_name + "screenshots_result.txt", 'w') as outfile:
            json.dump(screenshots_results, outfile)

        video_to_process.release()
        cv2.destroyAllWindows()


def get_video_type_by_collision(video_type_rect, video_name):
    """
        This function aims to find what is the right rectangle instantiation according to the name of the video.

    :param video_type_rect: class DatabaseVision (self.rect)

    :param video_name: str

    :return: points of the rectangle or None
    """
    for video_type in video_type_rect:
        if video_name.find(video_type) != -1:
            return video_type_rect.get(video_type)
    return None


def filter_file_name(file_name):
    """
        This function aims to deliver an accurate decomposition of the video file name.
        This function is highly dependent of the format given for the example.

    :param file_name: str

    :return: str, str
    """
    it_video_format = file_name.find(".")
    it_video_lane = file_name.find("_")
    it_video_hour = file_name.find("_", it_video_lane+1)
    return file_name[it_video_lane+1:it_video_hour], file_name[it_video_hour+1:it_video_format-1]


def process_image(image, img_nn_size=224, normalize=False):
    """
        This functions aims to prepare the image for the NN.

        If img_nn_size <= 0 : do not resize !
        Else img_nn_size > 0 : resize !

        All pre-trained models expect input images normalized in the same way,

        i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.

        The images have to be loaded in to a range of [0, 1]

        and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

        Theory :

    https://stackoverflow.com/questions/4674623/why-do-we-have-to-normalize-the-input-for-an-artificial-neural-network/

        When you use unnormalized input features, the loss function is likely to have very elongated valleys.
        When optimizing with gradient descent, this becomes an issue because the gradient will be steep with respect
        some of the parameters.
        That leads to large oscillations in the search space, as you are bouncing between steep slopes.
        To compensate, you have to stabilize optimization with small learning rates.

    :param normalize: Bool

    :param image: OpenCV Image

    :param img_nn_size: int

    :return: OpenCV Image
    """

    if img_nn_size > 0:
        # rows, columns and channels
        height_img = image.shape[0]
        width_img = image.shape[1]

        # Find the shorter side and resize it to img_nn_size keeping aspect ratio
        # if the height_img > width_img
        if width_img > height_img:
            scale = img_nn_size / width_img
            dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            # Constrain the height to be img_nn_size
            resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        else:
            scale = img_nn_size / height_img
            dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            # Constrain the width to be img_nn_size
            resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    else:
        resized = image

    # DEBUG
    # print(resized.shape, image.shape)
    # cv2.imshow("cropped", resized)

    # Convert values to range of 0 to 1 instead of 0-255
    image_filtered = np.array(resized)
    image_filtered = image_filtered / 255

    if normalize is True:
        # Normalize the image according to the spec
        image_filtered -= image_filtered.mean()
        image_filtered /= image_filtered.std()

    # Move color channels to first dimension as expected by PyTorch

    image_filtered = image_filtered.transpose(2, 0, 1)

    return image_filtered
