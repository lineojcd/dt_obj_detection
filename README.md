# dt_obj_detection
Duckietown LRA Object detection Exercise

cloned from https://github.com/duckietown-ethz/object-detection-ex-template

Teammate: Junting Chen


### How to collect data from the Duckietown simulator
```bash
git clone https://github.com/lineojcd/dt_obj_detection.git
cd dt_obj_detection/data_collection
python3 data_collection.py
```

### How to train the model
```bash
git clone https://github.com/lineojcd/dt_obj_detection.git
cd dt_obj_detection/model
python3 train.py
```

### How to build Docker image
```bash
cd dt_obj_detection
docker build -f Dockerfile -t [DOCKER_IMAGE_NAME] .
```

### How to evaluate your model
```bash
cd dt_obj_detection/eval
make eval-gpu SUB=[DOCKER_IMAGE_NAME]
or 
make eval-cpu SUB=[DOCKER_IMAGE_NAME]
```

### How to get the pre-trained model
You can find the pre-trained model from my Dockerhub image: lineojcd/dt_obj_detection:latest
