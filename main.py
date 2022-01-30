import argparse
import logging

import cv2

from shapeclf.image_processor import ImageProcessor


def create_parser():
    parser = argparse.ArgumentParser("Colorize figures according to shapes")

    parser.add_argument("--src_fpath", default="./img/test_figures.jpg", help="Path to input image")
    parser.add_argument("--dst_fpath", default="./img/test_figures_out.png", help="Path to output image")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    image = cv2.imread(args.src_fpath)[:, :, ::-1]
    image = (image > 100)[:, :, 0]
    logging.info(f"File {args.src_fpath} succesfully read, shape {image.shape}")

    processor = ImageProcessor()
    output_image = processor(image)
    logging.info("Image succesfully processed")

    cv2.imwrite(args.dst_fpath, output_image)
    logging.info(f"Result saved to file {args.dst_fpath}")


if __name__ == "__main__":
    main()