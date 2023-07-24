# camera_shake

usage:
1) use inpainting, copymoves, inpaint in resolution 200x200, scale original moves by 2, use first 600 frames
    ```python apply_camera_shake.py --input_video /mnt/f/works/test_task_videos/01.mp4 --output_video /mnt/f/works/test.mp4 --inpaint --scale 2 --copymoves --inpaint_width 200 --inpaint_height 200 --maxframes 600```

2) use bordercopy, copymoves, no inpainting, use first 600 frames
   ```python apply_camera_shake.py --input_video /mnt/f/works/test_task_videos/01.mp4 --output_video /mnt/f/works/test.mp4 --scale 2 --copymoves --maxframes 600```

more info in gdoc report
