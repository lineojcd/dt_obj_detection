# don't forget to submit to dropbox as amod20-hw8-training-dataset-<your-full-name>
import numpy as np
from agent import PurePursuitPolicy
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask
from PIL import Image, ImageOps
import cv2
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, opening
from skimage.color import rgb2gray
import scipy


npz_index = 0
DISPLAY = True
# DISPLAY_PATCH = True

object_colors = np.array(   [[100, 117, 226], # duckie purple
                            [226, 111, 101], # cone red
                            [116, 114, 117],  # truck grey
                            [216, 171, 15]]   # bus yellow
                        ).astype(np.uint8)
object_labels = np.array([1,2,3,4])

def save_npz(img, boxes, classes):
    global npz_index
    with makedirs("./dataset"):
        np.savez(f"./dataset/{npz_index}.npz", *(img, boxes, classes))
        npz_index += 1
    if npz_index%100 ==0:
        print("collecting ",npz_index, "data entries")

def color2label(color):
    colors = object_colors
    labels = [1,2,3,4]
    idx_arr = np.where((colors == color).all(axis=1))[0]
    if len(idx_arr) == 0: # background color
        return 0
    else:
        return labels[idx_arr[0]]

# In theory, the pepper noise point has 17 pixels
def patch2label(patch, threshold=50):
    flat_patch = np.reshape(patch,[-1,3]).astype(int) # ground to integer
    colors, counts = np.unique(flat_patch, axis=0, return_counts=True) # analyze dominating color
    ranking = np.argsort(counts)
    # only colors with >threshold pixels are considered dominating in this patch
    dominating_colors= colors[np.where(counts[ranking] > threshold)]
    for color in dominating_colors:
        label = color2label(color)
        if label != 0:
            return label

    # no dominating non-background color found
    return 0

# set all colors not in target color list to [0,0,0]
def extract_front(image, target_color):
    target_mask = (image == target_color).all(axis=2)
    target_image = np.zeros(image.shape).astype(np.uint8)
    target_image[target_mask] = target_color
    return target_image


def clean_segmented_image(seg_img, threshold=10):
    # Tip: use either of the two display functions found in util.py to ensure that your cleaning produces clean masks
    # (ie masks akin to the ones from PennFudanPed) before extracting the bounding boxes
    assert(seg_img.shape == (224, 224, 3))

    boxes = []
    classes = []
    for i in range(len(object_labels)):
        color = object_colors[i]
        class_label = object_labels[i]

        # set all background to [0,0,0]
        clean_img = extract_front(seg_img, color)
        gray_img = rgb2gray(clean_img)
        # apply threshold
        # thresh = threshold_otsu(gray_img)
        # Dilation followed by Erosion, remove noise in foreground
        # template = square(5)
        # template = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
        binary_map = gray_img > 1e-3
        # cannot use opening: too small filter does not work,
        # while too big filter segments duckie & shrinks cone
        # cleaned_map = opening(binary_map, template)
        label_image = label(binary_map)

        for region in regionprops(label_image):
            ymin, xmin, ymax, xmax = region.bbox
            # filter out rect-like bbox of salt&pepper noise
            if (ymax - ymin) > threshold or (xmax - xmin) > threshold:
                # minr, minc, maxr, maxc = region.bbox
                # patch = seg_img[ymin:ymax, xmin:xmax]
                # class_label = patch2label(patch, threshold)
                boxes.append([xmin, ymin, xmax, ymax])
                classes.append(class_label)

    if DISPLAY:
        disp_img = seg_img.copy()
        i = 0
        for i in range(len(classes)):
            xmin, ymin, xmax, ymax = boxes[i]
            class_label = classes[i]

            cv2.rectangle(disp_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(disp_img,str(class_label),(xmax+5,ymax+5),0,0.3,(0,255,0))

        # cv2.resize(disp_img, (448,448))
        cv2.imshow("Rectangled", disp_img[:, :, ::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    # convert data type
    boxes = np.array(boxes)
    classes = np.array(classes)
    return boxes, classes



seed(123)
# environment = launch_env()
environment = launch_env(camera_width=224, camera_height=224)
policy = PurePursuitPolicy(environment)

MAX_STEPS = 4000

nb_of_steps = 0

while nb_of_steps < MAX_STEPS:
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    while True:
        action = policy.predict(np.array(obs))

        obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
        segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array

        rewards.append(rew)
        # environment.render(segment=int(nb_of_steps / 50) % 2 == 0)
        environment.render(segment=False)

        boxes, classes = clean_segmented_image(segmented_obs)
        if len(classes) > 0:
            save_npz(obs, boxes, classes)
            nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break
