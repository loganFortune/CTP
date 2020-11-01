"""
    Name Design: wvideo.py

    Author: Logan Fortune

    Email: logan.fortune@orange.fr

    License: Open Source

    Client: Wintics

    Date: October 2020
"""

# Test Library
import unittest
# Computer Vision
import cv2

# Python procedures to check
from wvideo import process_image, filter_file_name

# ML
import torchvision
import torch

# Download Model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# Training dataset : https://github.com/nightrome/cocostuff
# Evaluation Mode
model.eval()
# Use CPU (constraint)
device = torch.device('cpu')


class TestWVideo(unittest.TestCase):
    """
        This test provides some simple tests for some tools given in the wvideo.py file
    """

    def setUp(self):
        """
            Initialisation tests.

            Warning: This test takes wintics_image.jpg as the source of the test !!

        """
        self.image = cv2.imread("./test/wintics_image.jpg")
        self.image_detection = cv2.imread("./test/embouteillages_peripherique.jpg")
        self.assertIsNotNone(self.image)
        self.assertIsNotNone(self.image_detection)

    def test_process_image(self):
        """

            This test checks if the process_image function from wvideo is working as expected !

        :return:
        """
        image_filtered = process_image(self.image, img_nn_size=224, normalize=False)
        self.assertEqual(image_filtered.shape[0], 3)  # check transpose is working correctly
        self.assertEqual(image_filtered.shape[1], 224)  # check dimensions of the image with a simple example
        self.assertEqual(image_filtered.shape[2], 224)
        self.assertEqual(image_filtered.dtype, "float64")  # check that we have float values

    def test_file_name(self):
        """

            This test checks of the function filter_file_name function from wvideo.py is working as expected !

        :return:
        """
        inputs = ["highway_2_9h.mp4", "highway_13_18h.mp4"]
        for i in range(0, len(inputs)):
            output_filter = filter_file_name(inputs[i])
            if i == 0:
                self.assertEqual(output_filter, tuple(["2", "9"]))
            else:
                self.assertEqual(output_filter, tuple(["13", "18"]))

    def test_video_detection(self):

        resizing_dim = 256

        img_process = process_image(self.image_detection, resizing_dim)

        # run inference on the model and get detections
        tensor_img = torch.FloatTensor([img_process])  # 32-bit floating point
        with torch.no_grad():
            tensor_img = tensor_img.to(device)
            detections = model(tensor_img)[0]

        boxes = detections["boxes"]  # boxes
        labels = detections["labels"]  # labels
        scores = detections["scores"]  # scores

        tensor_filter = torchvision.ops.nms(boxes, scores, 0.5)  # IOU = AREA of Overlap / AREA of the Union
        for i in range(0, len(tensor_filter)):
            resized = cv2.rectangle(img_process.transpose(1, 2, 0),
                                    tuple([int(boxes[tensor_filter[i]][0]), int(boxes[tensor_filter[i]][1])]),
                                    tuple([int(boxes[tensor_filter[i]][2]), int(boxes[tensor_filter[i]][3])]),
                                    color=(255, 0, 0), thickness=2)
        """cv2.imshow("Filter", resized)
        cv2.waitKey(0)"""
        # TODO: We can talk about it !!


if __name__ == '__main__':
    unittest.main()