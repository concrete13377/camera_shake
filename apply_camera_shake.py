import argparse
from pathlib import Path
import pickle
import shutil

import cv2
import numpy as np
from PIL import Image

# import matplotlib.pyplot as plt

from apply_e2fgvi import run_inpainting, get_files


def parse():
    parser = argparse.ArgumentParser(
        prog="apply camera shake",
        description="",
    )
    parser.add_argument("--input_video", type=str)
    parser.add_argument("--output_video", type=str)
    parser.add_argument("--inpaint", action="store_true")
    parser.add_argument("--inpaint_width", type=int, default=64)
    parser.add_argument("--inpaint_height", type=int, default=64)
    parser.add_argument("--scale", type=int)
    parser.add_argument("--copymoves", action="store_true")
    args = parser.parse_args()
    return args


def circulize(frame):
    frame2 = frame.copy()
    height, width = frame2.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = min(center[0], center[1])
    cv2.circle(mask, center, radius, (255), -1)
    masked_frame = cv2.bitwise_and(frame2, frame2, mask=mask)
    return masked_frame


if __name__ == "__main__":
    args = parse()
    print(
        args.input_video,
        args.output_video,
        args.inpaint,
        args.copymoves,
        args.scale,
        args.inpaint_width,
        args.inpaint_height,
    )

    if args.inpaint:
        framesdir = Path("temp_frames")
        masksdir = Path("temp_masks")
        outpath = Path("temp_outdir")

        framesdir.mkdir(parents=True, exist_ok=True)
        masksdir.mkdir(parents=True, exist_ok=True)
        outpath.mkdir(parents=True, exist_ok=True)

    with open("henry_kavil_moves.pkl", "rb") as file:
        henry_kavil_moves = pickle.load(file)

    video = cv2.VideoCapture(str(args.input_video))
    frame_count = 0
    idx = 0
    frame_rate = 30.0

    h, w = None, None
    video_writer = None
    n_iter = 0
    while video.isOpened():
        ret, frame = video.read()  # frame in bgr
        if ret:
            if n_iter == 600:
                break
            n_iter += 1
            if h is None and w is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                if args.inpaint:
                    w, h = args.inpaint_width, args.inpaint_height
                else:
                    h, w, c = frame.shape

                video_writer = cv2.VideoWriter(
                    str(args.output_video), fourcc, frame_rate, (w, h)
                )

            if args.copymoves:
                # just use moves from henry_kavil "motioncapture"
                mx, my = henry_kavil_moves[idx % len(henry_kavil_moves)]
                mx *= args.scale
                my *= args.scale
            else:
                # generate new mx my
                raise NotImplementedError()
            M = np.float32([[1, 0, mx], [0, 1, my]])
            moved_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

            if not args.inpaint:
                # copyborders
                mxx = int(mx) + 1
                myy = int(my) + 1

                if mxx >= 0:
                    dobor_x = frame[:, :mxx, :]
                    moved_frame[:, :mxx, :] = dobor_x
                else:
                    dobor_x = frame[:, mxx:, :]
                    moved_frame[:, mxx:, :] = dobor_x

                if myy >= 0:
                    dobor_y = frame[:myy, :, :]
                    moved_frame[:myy, :, :] = dobor_y
                else:
                    dobor_y = frame[myy:, :, :]
                    moved_frame[myy:, :, :] = dobor_y
                circled_frame = circulize(moved_frame)
                video_writer.write(circled_frame)  # circuled frame still in bgr

            else:
                # save frames and masks for inpainting
                mask_frame = np.zeros((moved_frame.shape[0], moved_frame.shape[1]))
                mask_frame = cv2.warpAffine(
                    mask_frame,
                    M,
                    (mask_frame.shape[1], mask_frame.shape[0]),
                    borderValue=255,
                )
                mask_frame = Image.fromarray(mask_frame)
                mask_frame = np.array(
                    mask_frame.resize((args.inpaint_width, args.inpaint_height))
                )

                moved_frame = Image.fromarray(moved_frame)
                moved_frame = np.array(
                    moved_frame.resize((args.inpaint_width, args.inpaint_height))
                )

                padding = 5
                padded_mask_frame = (
                    np.ones(
                        (
                            mask_frame.shape[0] + padding * 2,
                            mask_frame.shape[1] + padding * 2,
                        ),
                        dtype=np.uint8,
                    )
                    * 255
                )
                padded_mask_frame[
                    padding : moved_frame.shape[1] + padding,
                    padding : padding + moved_frame.shape[0],
                ] = mask_frame

                padded_frame = np.zeros(
                    (
                        moved_frame.shape[0] + padding * 2,
                        moved_frame.shape[1] + padding * 2,
                        3,
                    ),
                    dtype=np.uint8,
                )
                padded_frame[
                    padding : moved_frame.shape[1] + padding,
                    padding : padding + moved_frame.shape[0],
                    :,
                ] = moved_frame

                output_path = Path(framesdir, str(idx)).with_suffix(".jpg")
                output_path_mask = Path(masksdir, str(idx)).with_suffix(".jpg")
                cv2.imwrite(str(output_path), padded_frame)  # moved frame still in bgr
                cv2.imwrite(str(output_path_mask), padded_mask_frame)

            idx += 1

        else:
            break

    if args.inpaint:
        # inpaint frames
        ckpt = "../E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth"
        run_inpainting(
            framesdir,
            masksdir,
            args.inpaint_width + padding * 2,
            args.inpaint_height + padding * 2,
            ckpt,
            outpath,
        )
        for frame_path in get_files(outpath):
            frame = cv2.imread(str(frame_path))
            frame = frame[padding:-padding, padding:-padding, :]
            cv2.imwrite(str(frame_path), frame)
            frame = circulize(frame)
            video_writer.write(frame)

        try:
            shutil.rmtree(framesdir)
            shutil.rmtree(masksdir)
            # shutil.rmtree(outpath)
            print("directory is deleted")
        except OSError as x:
            print(f"during temp dir deletion an error occured: {x.strerror}")

    video.release()
    video_writer.release()

# python apply_camera_shake.py --input_video /mnt/f/works/test_task_videos/01.mp4 --output_video /mnt/f/works/test.mp4 --inpaint --scale 20 --copymoves --inpaint_width 200 --inpaint_height 200
# python apply_camera_shake.py --input_video /mnt/f/works/test_task_videos/01.mp4 --output_video /mnt/f/works/test2.mp4 --scale 20 --copymoves
