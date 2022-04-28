import atexit
import bisect
import multiprocessing as mp
from collections import deque
import numpy as np

import cv2
import torch
from detectron2.data import MetadataCatalog
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.structures import BoxMode
from detectron2.utils.createcubemap import e2c

from defrcn.engine import DefaultPredictor


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes
                from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image, file_path):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(
            image, self.metadata, instance_mode=self.instance_mode
        )
        prediction_json = {"file_name": file_path}
        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)
            prediction_json["instances"] = instances_to_coco_json(
                instances, file_path
            )
            vis_output = visualizer.draw_instance_predictions(
                predictions=instances
            )

        return predictions, vis_output, prediction_json
    
    
    def run_on_pano(self, image, file_path):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        cubemap_dict = e2c(image)
        predictions_dict = {}
        vis_output_dict= {}
        prediction_json_list = []

        key_dict = ['F', 'R', 'B', 'L']
        for view in key_dict:
            img = cubemap_dict[view]
            vis_output = None
            predictions = self.predictor(img)
            # visualizer = Visualizer(
            #     img[:, :, ::-1], self.metadata, instance_mode=self.instance_mode
            # )
            prediction_json = {"file_name": file_path}
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                prediction_json["instances"] = instances_to_coco_json(
                    instances, view
                )
                # vis_output = visualizer.draw_instance_predictions(
                #     predictions=instances
                # )
            predictions_dict[view] = predictions
            vis_output_dict[view] = vis_output
            prediction_json_list.append(prediction_json)
        
        return predictions_dict, vis_output_dict, prediction_json_list


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = (
                "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            )
            self.procs.append(
                AsyncPredictor._PredictWorker(
                    cfg, self.task_queue, self.result_queue
                )
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5


def instances_to_coco_json(instances, view):
    num_instance = len(instances)
    if num_instance == 0:
        return []

    ftDict = {'F': [0, 0], 'R': [np.pi/2, 0],
              'B': [np.pi, 0], 'L': [np.pi/2*3, 0]}
    ftu, ftv = ftDict[view]

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = bboxesCubemap2Pano(boxes, ftu, ftv)

    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "id": -99999,
            "category_id": classes[k],
            "bbox": boxes[k],
            "view": view,
            "score": scores[k],
        }
        results.append(result)

    return results


def cubemapCoord2Lonlat(x, y, WIDTH, ftu, ftv):
    an = np.sin(np.pi / 4)
    ak = np.cos(np.pi / 4)

    #   // Map face pixel coordinates to [-1, 1] on plane
    ny = y / WIDTH - 0.5
    nx = x / WIDTH - 0.5
    nx *= 2.0
    ny *= 2.0
    #   // Map [-1, 1] plane coords to [-an, an]
    #   // thats the coordinates in respect to a unit sphere
    #   // that contains our box.
    nx *= an
    ny *= an
    # print( nx, ny)

    if ftv == 0:
        # // Center faces
        u = np.arctan(nx / ak)
        v = np.arctan(ny * np.cos(u) / ak)
        u += ftu
    elif ftv > 0:
        # // Bottom face
        d = np.sqrt(nx * nx + ny * ny)
        v = np.pi / 2 - np.arctan(d / ak)
        if ny == 0 and nx == 0:
            u = 0
        else:
            u = np.arctan(ny / nx)
    else:
        # // Top face
        d = np.sqrt(nx * nx + ny * ny)
        v = -np.pi / 2 + np.arctan(d / ak)
        if ny == 0 and nx == 0:
            u = 0
        else:
            u = np.arctan(-ny / nx)

    # // Map from angular coordinates to [-1, 1], respectively.
    u = u / np.pi
    v = v / (np.pi / 2)

    # // Warp around, if our coordinates are out of bounds.
    while v < -1:
        v += 2
        u += 1
    while v > 1:
        v -= 2
        u += 1
    while u < -1:
        u += 2
    while u > 1:
        u -= 2

    # // Map from [-1, 1] to in texture space
    u = u * 180
    v = v * 90

    return [u, v]


def bboxesCubemap2Pano(boxes, ftu, ftv):
    new_boxes = []
    WIDTH = 1080

    for box in boxes:
        x1, y1, x2, y2 = box
        uv1 = cubemapCoord2Lonlat(x1, y1, WIDTH, ftu, ftv)
        uv2 = cubemapCoord2Lonlat(x2, y1, WIDTH, ftu, ftv)
        uv3 = cubemapCoord2Lonlat(x2, y2, WIDTH, ftu, ftv)
        uv4 = cubemapCoord2Lonlat(x1, y2, WIDTH, ftu, ftv)
        new_boxes.append([uv1, uv2, uv3, uv4])

    return new_boxes