from genericfinder import testutil_main

from fastfinder2022 import FastFinder2022 as CircleFinder
from fastfinder2022old import FastFinder2022 as CircleFinderOld
from hubfinder2022 import HubFinder2022 as ProjectedCircleFinder
import math
if __name__ == '__main__':
    testutil_main(
        {
            "circle_finder": CircleFinder,
            #"circle_finder_old": CircleFinderOld,
            "projected_circle_finder": ProjectedCircleFinder
        },
        output_dir = "output_dir",
        calib_file = "../data/calibration/c930e_848x480_calib.json",
        rotate_calib = True,
        input_files = ["../data/test_images/2022/shed/*.png"],
        annotations = {
            ("1648079368.07", "1648079370.17"): {"distance": 119, "angle": 0},
            ("1648079467.06", "1648079470.25"): {"distance": 119, "angle": 15},

            ("1648079546.11", "1648079547.18"): {"distance": 148, "angle": 0},
            ("1648079602.57", "1648079605.22"): {"distance": 148, "angle": -20},

            ("1648079636.14", "1648079638.27"): {"distance": 103, "angle": 0},

            ("1648079704.14", "1648079706.76"): {"distance": 71, "angle": 0},

            ("1648079764.39", "1648079775.23"): {"distance": 171, "angle": 10},

            ("1648081320.51", "1648081328.83"): {"distance": 205, "angle": 0},

            ("1648082856.95", "1648082859.02"): {"distance": 167, "angle": 0},

            ("1648083272.46", "1648083275.67"): {"distance": 107, "angle": 0},

        },
        metrics = [
            {
                "name": "angle",
                "unit": "deg",
                "array_index": 3,
                "convert": lambda x: -math.degrees(x)
            },
            {
                "name": "distance",
                "unit": "in",
                "array_index": 2,
                "convert": lambda x: x+26.6875
            }
        ]
    )
