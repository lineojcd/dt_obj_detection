import contextlib
import os

import numpy as np
import random
import math
import os


@contextlib.contextmanager
def directory(name):
    ret = os.getcwd()
    os.chdir(name)
    yield None
    os.chdir(ret)

@contextlib.contextmanager
def makedirs(name):
    try:
        os.makedirs(name)
    except:
        pass
    yield None

def seed(seed):
    # torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def launch_env(
        map_name="udem_spooky",
        camera_width=640,
        camera_height=480,
):
    import gym_duckietown
    from gym_duckietown.envs import DuckietownEnv
    env = DuckietownEnv(
        domain_rand=False,
        max_steps=math.inf,
        randomize_maps_on_reset=False,
        map_name=map_name,
        camera_width=camera_width,
        camera_height=camera_height,
    )
    return env

import cv2

def _mod_mask(mask):
    temp = mask.copy()
    temp[temp == 1] = 50
    temp[temp == 2] = 100
    temp[temp == 3] = 150
    temp[temp == 4] = 200
    temp = temp.astype("uint8")
    mask = cv2.applyColorMap(temp, cv2.COLORMAP_RAINBOW)
    return mask


def display_seg_mask(seg_img, masked):
    dsize = seg_img.shape[0]*3, seg_img.shape[1]*3

    seg_img = cv2.resize(seg_img, dsize)
    masked = cv2.resize(_mod_mask(masked), dsize)

    all = np.concatenate(
        ( seg_img, masked),
        axis=1
    )

    cv2.imshow("image", all)
    cv2.waitKey(0)

def display_img_seg_mask(real_img, seg_img, masked):
    masked = _mod_mask(masked)

    all = np.concatenate(
        (cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR), seg_img, masked),
        axis=1
    )

    cv2.imshow("image", all)
    cv2.waitKey(0)