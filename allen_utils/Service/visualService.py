
from yolact_edge.data import COCODetection, YoutubeVIS, get_label_map, MEANS, COLORS
from yolact_edge.data import cfg, set_cfg, set_dataset
from yolact_edge.yolact import Yolact
from yolact_edge.utils.augmentations import BaseTransform, BaseTransformVideo, FastBaseTransform_for_cpu, Resize
from yolact_edge.utils.functions import MovingAverage, ProgressBar
from yolact_edge.layers.box_utils import jaccard, center_size
from yolact_edge.utils import timer
from yolact_edge.utils.functions import SavePath
from yolact_edge.layers.output_utils import postprocess, undo_image_transformation
from yolact_edge.utils.tensorrt import convert_to_tensorrt
from collections import defaultdict

import numpy as np
import torch
import argparse
import time
import random
import json
import cv2
import os

class VisualService:   
    def str2bool(self,v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.') 
    def parse_args(self, argv=None):
        parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
        parser.add_argument('--trained_model',
                            default=None, type=str,
                            help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
        parser.add_argument('--top_k', default=5, type=int,
                            help='Further restrict the number of predictions to parse')
        parser.add_argument('--cuda', default=True, type=self.str2bool,
                            help='Use cuda to evaulate model')
        parser.add_argument('--fast_nms', default=True, type=self.str2bool,
                            help='Whether to use a faster, but not entirely correct version of NMS.')
        parser.add_argument('--display_masks', default=True, type=self.str2bool,
                            help='Whether or not to display masks over bounding boxes')
        parser.add_argument('--display_bboxes', default=False, type=self.str2bool,
                            help='Whether or not to display bboxes around masks')
        parser.add_argument('--display_text', default=False, type=self.str2bool,
                            help='Whether or not to display text (class [score])')
        parser.add_argument('--display_scores', default=False, type=self.str2bool,
                            help='Whether or not to display scores in addition to classes')
        parser.add_argument('--display', dest='display', action='store_true',
                            help='Display qualitative results instead of quantitative ones.')
        parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                            help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
        parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                            help='In quantitative mode, the file to save detections before calculating mAP.')
        parser.add_argument('--resume', dest='resume', action='store_true',
                            help='If display not set, this resumes mAP calculations from the ap_data_file.')
        parser.add_argument('--max_images', default=-1, type=int,
                            help='The maximum number of images from the dataset to consider. Use -1 for all.')
        parser.add_argument('--eval_stride', default=5, type=int,
                            help='The default frame eval stride.')
        parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                            help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
        parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                            help='The output file for coco bbox results if --coco_results is set.')
        parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                            help='The output file for coco mask results if --coco_results is set.')
        parser.add_argument('--config', default=None,
                            help='The config object to use.')
        parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                            help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
        parser.add_argument('--web_det_path', default='web/dets/', type=str,
                            help='If output_web_json is set, this is the path to dump detections into.')
        parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                            help='Do not output the status bar. This is useful for when piping to a file.')
        parser.add_argument('--display_lincomb', default=False, type=self.str2bool,
                            help='If the config uses lincomb masks, output a visualization of how those masks are created.')
        parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                            help='Equivalent to running display mode but without displaying an image.')
        parser.add_argument('--fast_eval', default=False, dest='fast_eval', action='store_true',
                            help='Skip those warping frames when there is no GT annotations.')
        parser.add_argument('--deterministic', default=False, dest='deterministic', action='store_true',
                            help='Whether to enable deterministic flags of PyTorch for deterministic results.')
        parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                            help='Do not sort images by hashed image ID.')
        parser.add_argument('--seed', default=None, type=int,
                            help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
        parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                            help='Outputs stuff for scripts/compute_mask.py.')
        parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                            help='Do not crop output masks with the predicted bounding box.')
        parser.add_argument('--image', default=None, type=str,
                            help='A path to an image to use for display.')
        parser.add_argument('--images', default=None, type=str,
                            help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
        parser.add_argument('--video', default=None, type=str,
                            help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
        parser.add_argument('--video_multiframe', default=1, type=int,
                            help='The number of frames to evaluate in parallel to make videos play at higher fps.')
        parser.add_argument('--score_threshold', default=0.3, type=float,
                            help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
        parser.add_argument('--dataset', default=None, type=str,
                            help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
        parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                            help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
        parser.add_argument('--yolact_transfer', dest='yolact_transfer', action='store_true',
                            help='Split pretrained FPN weights to two phase FPN (for models trained by YOLACT).')
        parser.add_argument('--coco_transfer', dest='coco_transfer', action='store_true',
                            help='[Deprecated] Split pretrained FPN weights to two phase FPN (for models trained by YOLACT).')
        parser.add_argument('--drop_weights', default=None, type=str,
                            help='Drop specified weights (split by comma) from existing model.')
        parser.add_argument('--calib_images', default=None, type=str,
                            help='Directory of images for TensorRT INT8 calibration, for explanation of this field, please refer to `calib_images` in `data/config.py`.')
        parser.add_argument('--trt_batch_size', default=1, type=int,
                            help='Maximum batch size to use during TRT conversion. This has to be greater than or equal to the batch size the model will take during inferece.')
        parser.add_argument('--disable_tensorrt', default=True, dest='disable_tensorrt', action='store_true',
                            help='Don\'t use TensorRT optimization when specified.')
        parser.add_argument('--use_fp16_tensorrt', default=False, dest='use_fp16_tensorrt', action='store_true',
                            help='This replaces all TensorRT INT8 optimization with FP16 optimization when specified.')
        parser.add_argument('--use_tensorrt_safe_mode', default=False, dest='use_tensorrt_safe_mode', action='store_true',
                            help='This enables the safe mode that is a workaround for various TensorRT engine issues.')

        parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                            benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False)

        args = parser.parse_args(argv)
        if args.output_web_json:
            args.output_coco_json = True
        
        if args.seed is not None:
            random.seed(args.seed)
        return args

    def prep_display(self, dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45):
        """
        Note: If undo_transform=False then im_h and im_w are allowed to be None.
        """
        args = self.parse_args()
        cfg.mask_proto_debug = args.mask_proto_debug
        if undo_transform:
            img_numpy = undo_image_transformation(img, w, h)
            img_gpu = torch.Tensor(img_numpy).cuda()
        else:
            img_gpu = img / 255.0
            h, w, _ = img.shape
        with timer.env('Postprocess'):
            t = postprocess(dets_out, w, h, visualize_lincomb = args.display_lincomb,
                                            crop_masks        = args.crop,
                                            score_threshold   = args.score_threshold)
            #torch.cuda.synchronize()

        with timer.env('Copy'):
            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][:args.top_k]
            #classes, scores, boxes = [x[:args.top_k].cpu().numpy() for x in t[:3]]
            classes, scores, boxes = [x[:args.top_k].detach().numpy() for x in t[:3]]
        num_dets_to_consider = min(args.top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < args.score_threshold:
                num_dets_to_consider = j
                break

        if num_dets_to_consider == 0:
            # No detections found so just output the original image
            return (img_gpu * 255).byte().cpu().numpy()

        # Quick and dirty lambda for selecting the color for a particular index
        # Also keeps track of a per-gpu color cache for maximum speed
        def get_color(j, on_gpu=None):
            color_cache = defaultdict(lambda: {})
            color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
            if on_gpu is not None and color_idx in color_cache[on_gpu]:
                return color_cache[on_gpu][color_idx]
            else:
                color = COLORS[color_idx]
                if not undo_transform:
                    # The image might come in as RGB or BRG, depending
                    color = (color[2], color[1], color[0])
                if on_gpu is not None:
                    #color = torch.Tensor(color).to(on_gpu).float() / 255.
                    color = torch.Tensor(color).float() / 255.

                    color_cache[on_gpu][color_idx] = color
                return color

        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
        if args.display_masks and cfg.eval_mask_branch:
            # After this, mask is of size [num_dets, h, w, 1]
            masks = masks[:num_dets_to_consider, :, :, None]
            array_masks = masks.detach().numpy()
            # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
            ##colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
            colors = torch.cat([get_color(j, on_gpu=0).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)

            masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
            # This is 1 everywhere except for 1-mask_alpha where the mask is
            inv_alph_masks = masks * (-mask_alpha) + 1
            # I did the math for this on pen and paper. This whole block should be equivalent to:
            #    for j in range(num_dets_to_consider):
            #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
            masks_color_summand = masks_color[0]
            if num_dets_to_consider > 1:
                inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(dim=0)
            img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
            maskImg = self.visualMask(img_gpu, inv_alph_masks.prod(dim=0), masks_color_summand)
        # Then draw the stuff that needs to be done on the cpu
        # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason

        img_numpy = (img_gpu * 255).byte().cpu().numpy()
        return img_numpy, maskImg#, np.sum(array_masks)
    def visualMask(self,img,mask,mask_color):
        h, w, _ = img.shape
        whiteShape = (h, w, 3)	
        whiteImg = np.full(whiteShape, 255).astype(np.uint8)  
        torch.from_numpy(whiteImg)   
        whiteImg = torch.from_numpy(whiteImg) *mask + mask_color
        return whiteImg.byte().cpu().numpy()
           
    def visualGrid(self, img, number):
        h, w, _ = img.shape
        for i in range(1,number):
            img[int(h/number*i),:]=0
            img[:,int(w/number*i)]=0  
        return img