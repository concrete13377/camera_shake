import sys
import importlib
from pathlib import Path

sys.path.append("../E2FGVI")

import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from core.utils import to_tensors
from torch.utils.data import Dataset


# sample reference frames from the whole video
def get_ref_index(f, neighbor_ids, length, num_ref, ref_length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref // 2))
        end_idx = min(length, f + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref:
                    break
                ref_index.append(i)
    return ref_index


def get_files(filespath):
    return list(
        sorted(
            filter(lambda x: x.is_file(), list(Path(str(filespath)).glob("*"))),
            key=lambda x: int(x.stem),
        )
    )


class FramesDS(Dataset):
    def __init__(self, framespath, size):
        self.framespaths = get_files(framespath)
        self.size = size

    def __len__(self):
        return len(self.framespaths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.framespaths[idx]))[..., ::-1]
        image = Image.fromarray(image)
        image = image.resize(self.size)
        return image


class MasksDS(Dataset):
    def __init__(self, framespath, size):
        self.framespaths = get_files(framespath)
        self.size = size

    def __len__(self):
        return len(self.framespaths)

    def __getitem__(self, idx):
        image = Image.open(str(self.framespaths[idx]))
        image = image.resize(self.size, Image.NEAREST)
        image = np.array(image.convert("L"))
        image = np.array(image > 0).astype(np.uint8)
        image = cv2.dilate(
            image, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4
        )
        image = Image.fromarray(image * 255)
        return image


def run_inpainting(video_path, mask_path, width, height, ckpt, outdir):
    with torch.no_grad():
        model = "e2fgvi_hq"
        # ref_length = 10
        # num_ref = -1
        # neighbor_stride = 5

        ref_length = 6
        num_ref = -1
        neighbor_stride = 3
        framestride = 12

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        size = (width, height)

        net = importlib.import_module("model." + model)
        model = net.InpaintGenerator().to(device)
        data = torch.load(ckpt, map_location=device)
        model.load_state_dict(data)
        print(f"Loading model from: {ckpt}")
        model.eval()

        framesds = FramesDS(video_path, size)
        masksds = MasksDS(mask_path, size)
        video_length = len(framesds)
        framepaths = framesds.framespaths
        maskpaths = masksds.framespaths
        comp_frames = [None] * video_length

        print(f"Start test...")
        x_frames_paths = []
        for i in range(0, len(framepaths), framestride):
            inner_paths = []
            for idxx in range(i, i + framestride):
                if idxx < len(framepaths):
                    inner_paths.append(framepaths[idxx])
            x_frames_paths.append(inner_paths)

        x_masks_paths = []
        for i in range(0, len(maskpaths), framestride):
            inner_paths = []
            for idxx in range(i, i + framestride):
                if idxx < len(maskpaths):
                    inner_paths.append(maskpaths[idxx])
            x_masks_paths.append(inner_paths)

        for itern in range(0, len(x_frames_paths), 1):
            print("intern", itern)
            stride_length = len(x_frames_paths[itern])
            strides = len(x_frames_paths)

            loopstartframe = 0
            loopendframe = stride_length

            xfram = x_frames_paths[itern]
            xmask = x_masks_paths[itern]

            if itern < strides - 1:
                for xframappend in range(0, neighbor_stride):
                    xfram.append(x_frames_paths[itern + 1][xframappend])
                    xmask.append(x_masks_paths[itern + 1][xframappend])

            # frames = [np.array(f).astype(np.uint8) for f in xfram]
            framesds.framespaths = xfram

            # binary_masks = [
            #     np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in xmask
            # ]
            # masks = to_tensors()(xmask).unsqueeze(0)
            # imgs, masks = imgs.half().to(device), masks.half().to(device)
            masksds.framespaths = xmask

            if itern > 0:
                loopstartframe = neighbor_stride
            else:
                loopstartframe = 0

            if itern < strides - 1:
                loopendframe = stride_length + neighbor_stride
            else:
                loopendframe = stride_length

            # completing holes by e2fgvi

            for outer_idx, f in tqdm(
                enumerate(range(loopstartframe, loopendframe, neighbor_stride))
            ):
                # for outer_idx, f in tqdm(enumerate(range(0, video_length, neighbor_stride))):
                neighbor_ids = [
                    i
                    for i in range(
                        max(loopstartframe, f - neighbor_stride),
                        min(loopendframe, f + neighbor_stride + 1),
                    )
                ]
                print(neighbor_ids)
                ref_ids = get_ref_index(
                    f, neighbor_ids, loopendframe, num_ref, ref_length
                )
                # selected_imgs = imgs[:1, neighbor_ids + ref_ids, :, :, :]
                selected_imgs = (
                    to_tensors()([framesds[idxxx] for idxxx in neighbor_ids + ref_ids])
                ).unsqueeze(0) * 2 - 1
                # selected_masks = masks_tensor[:1, neighbor_ids + ref_ids, :, :, :]
                selected_masks = (
                    to_tensors()([masksds[idxxx] for idxxx in neighbor_ids + ref_ids])
                ).unsqueeze(0)

                masked_imgs = selected_imgs * (1 - selected_masks)
                mod_size_h = 60
                mod_size_w = 108
                h_pad = (mod_size_h - height % mod_size_h) % mod_size_h
                w_pad = (mod_size_w - width % mod_size_w) % mod_size_w
                masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [3])], 3)[
                    :, :, :, : height + h_pad, :
                ]
                masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [4])], 4)[
                    :, :, :, :, : width + w_pad
                ]
                # print("masked imgs shape", masked_imgs.shape)
                pred_imgs, _ = model(masked_imgs, len(neighbor_ids))
                pred_imgs = pred_imgs[:, :, :height, :width]
                pred_imgs = (pred_imgs + 1) / 2
                pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
                for i in tqdm(range(len(neighbor_ids))):
                    idx = neighbor_ids[i]
                    binmask = np.expand_dims(
                        (np.array(masksds[idx]) != 0).astype(np.uint8), 2
                    )
                    img = np.array(pred_imgs[i]).astype(np.uint8) * binmask + np.array(
                        framesds[idx]
                    ).astype(np.uint8) * (1 - binmask)
                    if comp_frames[(itern * framestride) + idx] is None:
                        comp_frames[(itern * framestride) + idx] = img
                    else:
                        comp_frames[(itern * framestride) + idx] = (
                            comp_frames[(itern * framestride) + idx].astype(np.float32)
                            * 0.5
                            + img.astype(np.float32) * 0.5
                        )

                # save num_ref leftmost to disk
                if outer_idx >= 1:
                    for xxx in range(neighbor_stride):
                        idxtosave = (itern * framestride) + neighbor_ids[xxx]
                        imgtowrite = comp_frames[idxtosave].astype(np.uint8)[..., ::-1]

                        cv2.imwrite(
                            str(Path(outdir, str(idxtosave)).with_suffix(".jpg")),
                            imgtowrite,
                        )  # frames are in rgb
                        del comp_frames[idxtosave]
                        comp_frames.insert(idxtosave, None)
