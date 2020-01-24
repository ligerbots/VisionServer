#!/usr/bin/python3

# Use the Hough line algorithm to fix a contour

import numpy
import cv2
import math

from codetimer import CodeTimer


def hough_fit(contour, shape, nsides=4):
    contour_plot = numpy.zeros(shape=shape[:2], dtype=numpy.uint8)
    cv2.drawContours(contour_plot, [contour, ], -1, 255, 1)

    # the binning does affect the speed, so tune it....
    with CodeTimer("HoughLines"):
        lines = cv2.HoughLines(contour_plot, 1, numpy.pi / 180, threshold=10)
    if lines is None or len(lines) < nsides:
        print("Hough found too few lines")
        return None

    # The returned lines from HoughLines() are ordered by confidence, but there may/will be
    #  many variants of the best lines. Loop through the lines and pick the best from
    #  each cluster.

    x, y, w, h = cv2.boundingRect(contour)
    contour_center = (x + int(w / 2), y + int(h / 2))
    boundaries = (x-5, x+w+5, y-5, y+h+5)

    dist_thres = 10
    theta_thres = numpy.pi / 36  # 5 degrees
    best_lines = []
    for linelist in lines:
        line = linelist[0]
        if line[0] < 0:
            line[0] *= -1
            line[1] -= numpy.pi

        coord_near_ref = compute_line_near_reference(line, contour_center)

        if not best_lines or not is_close(best_lines, line, coord_near_ref, dist_thres, theta_thres):
            # print('best line:', line[0], math.degrees(line[1]))
            best_lines.append((line, coord_near_ref))

        if len(best_lines) == nsides:
            break

    if len(best_lines) != nsides:
        print("hough_fit: found %s lines" % len(best_lines))
        return None

    # Find the nsides vertices which are inside the bounding box (with a little slop).
    # There will be extra intersections. Assume the right ones (and only those) are within the bounding box.
    vertices = []
    iline1 = 0
    used = set()
    used.add(iline1)
    while len(used) < nsides:
        found = False
        for iline2 in range(nsides):
            if iline2 in used:
                continue

            inter = intersection(best_lines[iline1][0], best_lines[iline2][0])
            if inter is not None and \
               inter[0] >= boundaries[0] and inter[0] <= boundaries[1] and \
               inter[1] >= boundaries[2] and inter[1] <= boundaries[3]:
                vertices.append(inter)
                used.add(iline2)
                iline1 = iline2
                found = True
                break
        if not found:
            print("No intersection with %s and available lines" % iline1)
            return None

    # add in the last pair
    inter = intersection(best_lines[0][0], best_lines[iline1][0])
    if inter is not None and \
       inter[0] >= boundaries[0] and inter[0] <= boundaries[1] and \
       inter[1] >= boundaries[2] and inter[1] <= boundaries[3]:
        vertices.append(inter)

    if len(vertices) != nsides:
        print('Not correct number of vertices:', len(vertices))

    return numpy.array(vertices)


def compute_line_near_reference(line, ref_point):
    with CodeTimer("compute_line_near_reference"):
        rho, theta = line

        # remember: theta is actually perpendicular to the line, so there is a sign flip
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        x0 = cos_theta * rho
        y0 = sin_theta * rho

        if abs(cos_theta) < 1e-6:
            x_near_ref = None
            y_near_ref = y0
        elif abs(sin_theta) < 1e-6:
            x_near_ref = x0
            y_near_ref = None
        else:
            x_near_ref = x0 + (y0 - ref_point[1]) * sin_theta / cos_theta
            y_near_ref = y0 + (x0 - ref_point[0]) * cos_theta / sin_theta

    return x_near_ref, y_near_ref


def is_close(best_lines, candidate, coord_near_ref, dist_thres, theta_thres):
    cand_rho, cand_theta = candidate
    # print('cand:', cand_rho, math.degrees(cand_theta))
    for line in best_lines:
        line, best_near_ref = line
        # print('best', line, best_near_ref)

        delta_dists = []
        if coord_near_ref[0] is not None and best_near_ref[0] is not None:
            delta_dists.append(abs(coord_near_ref[0] - best_near_ref[0]))
        if coord_near_ref[1] is not None and best_near_ref[1] is not None:
            delta_dists.append(abs(coord_near_ref[1] - best_near_ref[1]))
        if not delta_dists:
            return True
        delta_dist = min(delta_dists)

        # angle differences greater than 180deg are not real
        delta_theta = cand_theta - line[1]
        while delta_theta >= numpy.pi / 2:
            delta_theta -= numpy.pi
        while delta_theta <= -numpy.pi / 2:
            delta_theta += numpy.pi
        delta_theta = abs(delta_theta)

        # print('test:', line[0], math.degrees(line[1]), delta_dist, delta_theta)
        if delta_dist <= dist_thres and delta_theta <= theta_thres:
            return True
    return False


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """

    with CodeTimer("intersection"):
        rho1, theta1 = line1
        rho2, theta2 = line2
        if abs(theta1 - theta2) < 1e-6:
            # parallel
            return None

        A = numpy.array([[numpy.cos(theta1), numpy.sin(theta1)],
                         [numpy.cos(theta2), numpy.sin(theta2)]])
        b = numpy.array([[rho1], [rho2]])
        x0, y0 = numpy.linalg.solve(A, b)
        res = int(numpy.round(x0)), int(numpy.round(y0))
    return res


def plot_hough_line(frame, rho, theta, color, thickness=1):
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv2.line(frame, pt1, pt2, color, thickness)
    return

if __name__ == '__main__':
    pts = numpy.array([[[0, 0]], [[1, 1]]])
    print(pts)
    lines = cv2.HoughLinesPointSet(pts, 10, 1, -10.0, 100.0, 0.1, 0.0, math.pi, math.pi / 360.0)
    print(lines)
