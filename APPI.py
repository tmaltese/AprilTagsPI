# April Tag Software
import copy
import numpy as np
import cv2 as cv
import math as m
from pupil_apriltags import Detector
import os
import time
from networktables import NetworkTablesInstance, NetworkTables
import sys

RED = '\033[0;31m'


def Main():
    NetworkTables.getDefault()
    NetworkTables.initialize(server="10.18.07.2")

    print(RED + "")
    # value of input device
    inputDevice = 0
    inputWidth = 600
    inputHeight = 1024

    # Family of april tags being detected
    families = 'tag16h5'
    # sensitivity for detection
    nthreads = 15
    # low resolution input help
    quadDecimate = 4.0
    # blurring for easier processing in noisy images
    quadSigma = 3.85
    # edges snap to gradiants
    refineEdges = 1.5
    # helps with strange lighting
    decodeSharpening = -0.85
    debug = 0

    # set video capture
    input = cv.VideoCapture(inputDevice)
    # set frame
    input.set(cv.CAP_PROP_FRAME_HEIGHT, inputHeight)
    input.set(cv.CAP_PROP_FRAME_WIDTH, inputWidth)

    # set detector settings
    at_detector = Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quadDecimate,
        quad_sigma=quadSigma,
        refine_edges=refineEdges,
        decode_sharpening=decodeSharpening,
        debug=debug,
    )

    while True:
        # ret is bool saying if input can be read
        # image is set to input
        ret, image = input.read()
        if not ret:
            # print("Input can not be accessed")
            break
        # make copy of image
        debugImage = copy.deepcopy(image)

        # convert image to black and white
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        tags = at_detector.detect(
            image,
            estimate_tag_pose=False,
            camera_params=None,
            tag_size=None,
        )

        # add tags to shown image

        debugImage = drawTags(debugImage, tags)

        key = cv.waitKey(1)
        if key == 27:
            break

        cv.imshow('april tags', debugImage)
    input.release()
    cv.destroyAllWindows()


def drawTags(image, tags, ):
    for tag in tags:
        degree = int
        # set variables of tags
        tagID = tag.tag_id
        center = tag.center
        corners = tag.corners
        cross = 317, 235

        focalLength = 850.5
        realWidth = 15.0

        table = NetworkTables.getTable('SmartDashboard')
        VisionTable = NetworkTables.getTable('Vision')

        # define corners and center of tag
        center = (int(center[0]), int(center[1]))
        corner1 = (int(corners[0][0]), int(corners[0][1]))
        corner2 = (int(corners[1][0]), int(corners[1][1]))
        corner3 = (int(corners[2][0]), int(corners[2][1]))
        corner4 = (int(corners[3][0]), int(corners[3][1]))

        # define the center of the top and bottom of tag
        detx1 = (corner2[0] + corner3[0]) / 2
        detx2 = (corner3[0] + corner4[0]) / 2
        dety1 = (corner2[1] + corner1[1]) / 2
        dety2 = (corner3[1] + corner4[1]) / 2
        dett = (0, int(dety1))
        detb = (0, int(dety2))

        detx1s = (corner2[0] + corner3[0]) / 2
        detx2s = (corner3[0] + corner4[0]) / 2
        dety1s = (corner2[1] + corner3[1]) / 2
        dety2s = (corner3[1] + corner4[1]) / 2

        # distances x and y for sides
        y = abs((dety1s - dety2s))
        x = abs((detx1s - detx2s))

        cv.circle(image, (cross[0], cross[1]), 5, (0, 255, 0), 5)

        # use distance formula on top and botom of tag
        if 400 > abs((dety1 - dety2)) > 35:
            d = m.dist(dett, detb)
        else:
            d = -1

        # make loop for calculating distance when possible
        if d > 0:
            # use distance formula to use pixel width, real width and focal length to find distance
            cv.circle(image, (center[0], center[1]), 5, (255, 0, 255), 2)

            cv.circle(image, (corner1[0], corner1[1]), 5, (255, 0, 0), 2)
            cv.circle(image, (corner2[0], corner2[1]), 5, (0, 255, 0), 2)
            cv.circle(image, (corner3[0], corner3[1]), 5, (255, 0, 255), 2)

            xd = abs((cross[0] - center[0]))
            yd = abs((cross[1] - center[1]))
            if x >= xd >= 0 and y >= yd >= 0:
                aligned = 1
                allignment = "alligned fully"
            elif x >= xd >= 0:
                aligned = 3
                allignment = "alligned horizontally"
            elif y >= yd >= 0:
                aligned = 4
                allignment = "alligned vertically"
            else:
                aligned = 2

            cv.line(image, (cross[0], cross[1]),
                    (center[0], center[1]), (255, 255, 0,), 2)

            if cross[0] > center[0]:
                dx = cross[0] - center[0]
                left = 1
            elif cross[0] < center[0]:
                dx = center[0] - cross[0]
                left = 2
            else:
                dx = 0
                left = 3

            if aligned == 2:
                cv.line(image, (corner1[0], corner1[1]),
                        (corner2[0], corner2[1]), (255, 0, 0), 2)
                cv.line(image, (corner2[0], corner2[1]),
                        (corner3[0], corner3[1]), (255, 0, 0), 2)
                cv.line(image, (corner3[0], corner3[1]),
                        (corner4[0], corner4[1]), (0, 0, 255), 2)
                cv.line(image, (corner4[0], corner4[1]),
                        (corner1[0], corner1[1]), (0, 0, 255), 2)
                distance = (realWidth * focalLength) / d


            elif aligned == 1:
                cv.line(image, (corner1[0], corner1[1]),
                        (corner2[0], corner2[1]), (0, 255, 0), 2)
                cv.line(image, (corner2[0], corner2[1]),
                        (corner3[0], corner3[1]), (0, 255, 0), 2)
                cv.line(image, (corner3[0], corner3[1]),
                        (corner4[0], corner4[1]), (0, 255, 0), 2)
                cv.line(image, (corner4[0], corner4[1]),
                        (corner1[0], corner1[1]), (0, 255, 0), 2)
                cv.putText(image, "ALIGNED", (0, 200),
                           cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv.LINE_AA)
                distance = (realWidth * focalLength) / d

            elif aligned == 3:
                cv.line(image, (corner1[0], corner1[1]),
                        (corner2[0], corner2[1]), (255, 0, 0), 2)
                cv.line(image, (corner2[0], corner2[1]),
                        (corner3[0], corner3[1]), (0, 255, 0), 2)
                cv.line(image, (corner3[0], corner3[1]),
                        (corner4[0], corner4[1]), (0, 0, 255), 2)
                cv.line(image, (corner4[0], corner4[1]),
                        (corner1[0], corner1[1]), (0, 255, 0), 2)
                cv.putText(image, "ALIGNED HORIZONTAL", (0, 200),
                           cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv.LINE_AA)
                distance = (realWidth * focalLength) / d

            elif aligned == 4:
                cv.line(image, (corner1[0], corner1[1]),
                        (corner2[0], corner2[1]), (0, 255, 0), 2)
                cv.line(image, (corner2[0], corner2[1]),
                        (corner3[0], corner3[1]), (0, 0, 255), 2)
                cv.line(image, (corner3[0], corner3[1]),
                        (corner4[0], corner4[1]), (0, 255, 0), 2)
                cv.line(image, (corner4[0], corner4[1]),
                        (corner1[0], corner1[1]), (0, 0, 255), 2)
                cv.putText(image, "ALIGNED VERTICAL", (0, 200),
                           cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv.LINE_AA)
                distance = (realWidth * focalLength) / d

            else:
                distance = 0

            side1 = m.dist(corner1, corner4)
            side2 = m.dist(corner2, corner3)
            side3 = m.dist(corner1, corner2)
            side4 = abs(m.dist(corner4, corner3))
            hyp1 = np.sqrt(np.square(side1) + np.square(side3))
            hyp2 = np.sqrt(np.square(side2) + np.square(side4))

            if side2 > 0:
                scale = 15 / side2
                distancex = dx * scale

            dx = cross[0] - center[0]

            if side1 and side2 > 4:
                if side1 > side2 and side4 > 0:
                    '''
                    angle_per_pix = 73/1080
                    degree = (center[0] - 540) * angle_per_x
                    '''
                    # print("side", (side1 + side2)/2)

                    degree = ((side2 / side1) / 2) * 360 - 180

                    # print("Side 1: ", side1)
                    # print("Side 2: ", side2)
                    # print("bottom", side4)

                    # print(degree)

                elif side2 > side1 and side4 > 0:
                    '''
                    angle_per_pix = 73/1080 
                    '''
                    degree = ((side1 / side2) / 2) * -360 + 180

                    # print("side", (side2+side1)/2)

                    # print("Side 1: ", side1)
                    # print("Side 2: ", side2)
                    # print("bottom", side4)

                    # print(degree)
                else:
                    degree = 0
            '''
            plt.plot([distance, distancex], 'ro')
            plt.plot([d, d],[dx, dx], '-')
            plt.grid()
            plt.xlim(-200, 200)
            plt.ylim(-200, 200)
            plt.show()
            '''

            # make distane that is printed to screen rounded
            dishow = round(distance)
            # show distance from tags
            if left == 1 and distancex > 0:
                cv.putText(image, str(np.around(distancex, 2)), (0, 500),
                           cv.FONT_HERSHEY_SIMPLEX, 0.95, (255, 0, 255), 4, cv.LINE_AA)

                cv.putText(image, ("cm left"), (0, 520),
                           cv.FONT_HERSHEY_SIMPLEX, 0.95, (255, 0, 255), 4, cv.LINE_AA)
            elif left == 2 and distancex > 0:
                cv.putText(image, str(np.around(distancex, 2)), (0, 500),
                           cv.FONT_HERSHEY_SIMPLEX, 0.95, (255, 0, 255), 4, cv.LINE_AA)
                cv.putText(image, ("cm right"), (0, 520),
                           cv.FONT_HERSHEY_SIMPLEX, 0.95, (255, 0, 255), 4, cv.LINE_AA)
                '''
            if  -1.5 < degree < 1.5:
                cv.putText(image, "alligned", (0, 30),
                cv.FONT_HERSHEY_SIMPLEX, 0.95, (255, 0, 255), 4, cv.LINE_AA)  
                '''

            if degree >= 0:
                cv.putText(image, str(round(degree)), (0, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.95, (255, 0, 255), 4, cv.LINE_AA)
                cv.putText(image, "degrees right", (110, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.95, (255, 0, 255), 4, cv.LINE_AA)
            elif degree <= 0:
                cv.putText(image, str(round(degree)), (0, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.95, (255, 0, 255), 4, cv.LINE_AA)
                cv.putText(image, "degrees left", (110, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.95, (255, 0, 255), 4, cv.LINE_AA)
            else:
                cv.putText(image, str(np.around(degree, 0), ), (0, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.95, (255, 0, 255), 4, cv.LINE_AA)

            cv.putText(image, str(dishow), (center[0] - 80, center[1] - 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2, cv.LINE_AA)
            cv.putText(image, ("cm"), (center[0] - 80, center[1] - 0),
                       cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2, cv.LINE_AA)
            # show exact measurement in console
            cv.putText(image, str(tagID), (center[0] - 10, center[1] - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2, cv.LINE_AA)
            # print(distance)

            filename = ("data.txt");
            with open(filename, "a") as f:
                f.write("distance : " + str(distance) + "\n")
                f.write("distancex : " + str(distancex) + "\n")
                f.write("degrees : " + str(degree) + "\n")

            with open("distance.txt", "a") as f:
                f.write(str(distance) + "\n")

            with open("distancex.txt", "a") as f:
                f.write(str(distancex) + "\n")

            table.putNumber("Distnace x:", distancex)
            table.putNumber("Distnace:", distance)
            table.putNumber("Degrees", degree)
            p = table.getEntry("Degrees")
            print(str(p))

        else:
            break
        # put tag id on tags

    return image
    return distance
    return degree
    return distancex


Main()