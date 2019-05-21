import argparse
import glob
from layoutanalysis.segmentation.segmentation import SegmentationSettings, Segmentator


def glob_all(filenames):
    files = []
    for f in filenames:
        files += glob.glob(f)

    return files


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', ''):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description='Segmentates lyric, text, systems and capitals from the image')
    parser.add_argument("--load", type=str,
                        help="Model to load")
    parser.add_argument("--gray", type=str, required=True, nargs="+",
                        help="directory name of the grayscale images")
    parser.add_argument("--staves", type=str, required=True, nargs="+",
                        help="directory name of the json staves files of the images")
    parser.add_argument("--processes", type=int, default=8,
                        help="Number of processes to use")
    parser.add_argument("--erode", type=bool, required=False, default=False,
                        help="Preprocessing step")
    parser.add_argument("--weight_threshold", type=float, required=False, default=0.5,
                        help="Weight Threshold used to separate text from systems")
    parser.add_argument("--debug", type=str2bool, default=False,
                        help="Display debug images")
    args = parser.parse_args()

    gray_file_paths = sorted(glob_all(args.gray))
    staffs_file_paths = sorted(glob_all(args.gray))

    settings = SegmentationSettings(
        erode=args.erode,
        processes = args.processes,
        debug=args.debug,
    )
    text_extractor = Segmentator(settings)

    for _ in text_extractor.segment([staffs_file_paths], [gray_file_paths]):
        pass


if __name__ == "__main__":
    main()
