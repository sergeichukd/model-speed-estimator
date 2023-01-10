import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Model Speed Estimator")
    parser.add_argument(
        "--video-path", help="Path to test video",
        default=None, required=False)
    parser.add_argument(
        "--img-dir-path", help="Path to directory with test images",
        default=None, required=False)
    return parser.parse_args()
