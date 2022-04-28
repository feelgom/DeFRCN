import argparse
import glob
import multiprocessing as mp
import os
import itertools
import time
import json

import cv2
import tqdm
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from demo.predictor_pano import VisualizationDemo
from defrcn.config import get_cfg
from detectron2.data import (
    MetadataCatalog
)

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(
        description="FsDet demo for builtin models"
    )
    parser.add_argument(
        "--config-file",
        default="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam", action="store_true", help="Take inputs from webcam."
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    demo = VisualizationDemo(cfg, parallel=True)

    prediction_jsons = []
    
    RESULTS_JSON = {"info":{}, "model":{}, "images":[], "predictions":[], "categories":[]}
    
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
            
        image_id =-1
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            image_id +=1
            RESULTS_JSON["images"].append({"id":image_id, "url":path})
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions_dict, visualized_output_dict, prediction_json_list = demo.run_on_pano(
                img, path)

            for elem in prediction_json_list:
                elem["instances"]["image_id"] = image_id
            
            prediction_jsons += prediction_json_list

            key_dict = ['F', 'R', 'B', 'L']
            for view in key_dict:
                predictions = predictions_dict[view]
                visualized_output = visualized_output_dict[view]
            
                logger.info(
                    "{}: {} in {:.2f}s".format(
                        path,
                        "detected {} instances".format(
                            len(predictions["instances"])
                        )
                        if "instances" in predictions
                        else "finished",
                        time.time() - start_time,
                    )
                )

                if args.output:
                    if os.path.isdir(args.output):
                        assert os.path.isdir(args.output), args.output
                        out_filename = os.path.join(
                            args.output, os.path.basename(path)
                        )
                    else:
                        os.mkdir(args.output)
                        out_filename = os.path.join(
                            args.output, os.path.basename(path)
                        )
                        
                #     out_filename = out_filename.split('.')[0]+"_"+view+out_filename[1]
                #     visualized_output.save(out_filename)
                # else:
                #     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                #     cv2.imshow(
                #         WINDOW_NAME, visualized_output.get_image()[:, :, ::-1]
                #     )
                #     if cv2.waitKey(0) == 27:
                #         break  # esc to quit

        _coco_results = list(
            itertools.chain(*[x["instances"] for x in prediction_jsons])
        )

        # unmap the category ids for COCO
        if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k
                for k, v in metadata.thing_dataset_id_to_contiguous_id.items()
            }

            for id, result in enumerate(_coco_results):
                result["id"] = id
                result["category_id"] = reverse_id_mapping[
                    result["category_id"]
                ]
                result["custom_label"] = False
            
        categories = []
        if hasattr(metadata, "thing_classes"):
            for id, name in enumerate(metadata.thing_classes):
                categories.append({"id":id, "name":name})        
        
        RESULTS_JSON["predictions"] = _coco_results
        RESULTS_JSON["categories"] = categories
            
        if args.output:
            file_path = os.path.join(
                args.output, "coco_instances_results.json"
            )
            with open(file_path, "w") as f:
                f.write(json.dumps(RESULTS_JSON))
                f.flush()
