import cv2
import numpy as np
import os
import time

STREAM=1
SINGLE=2
MODE=STREAM
# MODE=SINGLE

# since PIL channel order is different from cv2 channel order,
# with channels swaped, bbox will be more contrastive to the object
object_colors = [[100, 117, 226], # duckie purple
        [0, 200, 0],      # ghostie green
        [116, 114, 117],  # truck grey
        [216, 171, 15]]   # bus yellow

def image_bbox_show(image, boxes, classes):
    disp_img = image.copy()
    for i in range(len(classes)):
        xmin, ymin, xmax, ymax = boxes[i]
        class_label = classes[i]
        cv2.rectangle(disp_img, (xmin, ymin), (xmax, ymax), object_colors[class_label-1], 3)
        cv2.putText(disp_img, str(class_label), (xmax + 5, ymax + 5), 0, 0.3, object_colors[class_label-1])

    cv2.resize(disp_img, (448, 448))
    cv2.imshow("image", disp_img[:, :, ::-1])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    if MODE == STREAM:
        num_files = len(os.listdir("./dataset"))
        # num_files = len(os.listdir("./data_collection/newstyle_validation_dataset"))
        for i in range(num_files):
        # for i in range(100):
            # print("streaming ",i,"th pic now.")
            with np.load(f'./dataset/{i}.npz') as data:
            # with np.load(f'./data_collection/newstyle_validation_dataset/{i}.npz') as data:
                img, boxes, classes = tuple([data[f"arr_{j}"] for j in range(3)])
                image_bbox_show(img, boxes, classes)
            time.sleep(0.2)
            # print(img.shape) (224, 224, 3)
    if MODE == SINGLE:
        i=1
        with np.load(f'./dataset/{i}.npz') as data:
            img, boxes, classes = tuple([data[f"arr_{j}"] for j in range(3)])
            # image_bbox_show(img, boxes, classes)

            disp_img = img.copy()
            for i in range(len(classes)):
                xmin, ymin, xmax, ymax = boxes[i]
                class_label = classes[i]
                cv2.rectangle(disp_img, (xmin, ymin), (xmax, ymax), object_colors[class_label - 1], 3)
                cv2.putText(disp_img, str(class_label), (xmax + 5, ymax + 5), 0, 0.3, object_colors[class_label - 1])

            cv2.resize(disp_img, (448, 448))
            cv2.imshow("image", disp_img[:, :, ::-1])
            cv2.waitKey(0)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
    print("Model finished")