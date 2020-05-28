# Traffic-Sign-Classification-and-Detection
Course Project for THUEE Media and Cognition

To run this project, first run `conda create --name <env> --file requirements.txt` or `conda install --yes --file requirements.txt` or `pip install -r requirements.txt` to install all required modules.

Data are placed in `./data` directory with `data/Classification` and `data/Detection`.

All python scripts are run in this root directory except training YOLOv3.

# Models

Models are stored in root directory except for YOLOv3 models and are listed as follows.

- `SVM.m` - model for SVM in task 1
- `paramsResNet18.pkl` - model for ResNet18 in task 2 (could be used in task 4)
- `paramsResNet50.pkl` - model for ResNet50 in task 2 (could be used in task 4)
- `paramsDenseNet.pkl` - model for DenseNet in task 2 (could be used in task 4)
- `paramsInception.pkl` - model for Inception v3 in task 2 (could be used in task 4)
- `best_model.pth` - best model for prototypical network in task 3
- `last_model.pth` - last model for prototypical network in task 3
- `Detection/yolov3/weights/best.pt` - best model for YOLOv3 in task 4
- `Detection/yolov3/weights/last.pt` - last model for YOLOv3 in task 4

## Outputs

Outputs are json files in root directory as follows.

- `pred1.json` - output of task1
- `pred2.json` - output of task2
- `pred3.json` - output of task3
- `pred_annotations.json` - output of task4

## Classification

### Task 1

Run `python Classification/SVM_train.py` to train the model.

Run `python Classification/SVM_test.py` to test the model. It would output `pred1.json` in root directory.

Model would be saved in `SVM.m` in root directory, another `hog.npz` would be saved with HOG features of all images inside. `hog.npz` could be loaded if you uncomment certain lines in `Classification/SVM_train.py` instead of loading all images, but it wouldn't be needed if you train from scratch.

### Task 2

Run `python Classification/main.py` to train the model (default model ResNet18). 

Arguments are listed as follows, you could run `python Classification/main.py -h` for help

```
optional arguments:
  -h, --help     show this help message and exit
  --cuda CUDA
  --epoch EPOCH
  --model MODEL  You can choose: ResNet18, ResNet50, Inception, DenseNet
  --lr LR
  --BS BS        BatchSize
  --load LOAD    if you load an existing model or not
```

For example, to train an inception model, you could run `python Classification/main.py --epoch 250 --model Inception`.

Run `python Classification/pred.py` to test the model. It would output `pred2.json` in root directory. Arguments include `--model`. You could choose from 'ResNet18', 'ResNet50', 'Inception', 'DenseNet'.

### Task 3

#### KNN

Run `python Classification/FewShot/KNN/fewshot.py` to train and validate the model. This KNN model doesn't contain a test or prediction module.

#### Prototypical Network

Configs could be modified in `Classification/FewShot/prototypical/config.py`, which contains the following variables.

- `PRE_TRAINING_DATA_PATH` is the path of pre-training data, currently `data/Classification/Data/Train`.
- `VALIDATION_DATA_PATH` is the path of validation data, currently `data/Classification/Data/Test`.
- `DATA_FEW_SHOT` is the path of few shots data (including train and test), currently `data/Classification/DataFewShot`.
- `EXPERIMENT_PATH` is the path for output json file, model file and other training results, currently root directory.

Run `python Classification/FewShot/prototypical/train_protonet.py` to train the model. The document of arguments in this script could be found at https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch#training.

Run `python Classification/FewShot/prototypical/test_protonet.py` to test the model. It would output `pred3.json` in root directory. 

## Detection

### Task 4

Task 4 uses YOLOv3 from https://github.com/ultralytics/yolov3. 

First run `python Detection/makeLabel.py` to make detection labels.

To train YOLOv3, the following commands trains for 1000 epochs and uses batch-size 8. You could modify other arguments except `--cfg cfg/yolov3.cfg` and `--data data/traffic.data`. To resume the training, simply use argument `--resume`.

```
cd Detection/yolov3
python train.py --epochs 1000 --cfg cfg/yolov3.cfg --data data/traffic.data --batch-size 8 --weights ""
```

Arguments are listed as follows. You could go to the original repo or run `python train.py -h` for help.

```
usage: train.py [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--cfg CFG]
                [--data DATA] [--multi-scale]
                [--img-size IMG_SIZE [IMG_SIZE ...]] [--rect] [--resume]
                [--nosave] [--notest] [--evolve] [--bucket BUCKET]
                [--cache-images] [--weights WEIGHTS] [--name NAME]
                [--device DEVICE] [--adam] [--single-cls]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS
  --batch-size BATCH_SIZE
  --cfg CFG             *.cfg path
  --data DATA           *.data path
  --multi-scale         adjust (67% - 150%) img_size every 10 batches
  --img-size IMG_SIZE [IMG_SIZE ...]
                        [min_train, max-train, test]
  --rect                rectangular training
  --resume              resume training from last.pt
  --nosave              only save final checkpoint
  --notest              only test final epoch
  --evolve              evolve hyperparameters
  --bucket BUCKET       gsutil bucket
  --cache-images        cache images for faster training
  --weights WEIGHTS     initial weights path
  --name NAME           renames results.txt to results_name.txt if supplied
  --device DEVICE       device id (i.e. 0 or 0,1 or cpu)
  --adam                use adam optimizer
  --single-cls          train as single-class dataset
```

To test the model, go back to source directory and run `python Detection/predLabel.py`. It would output `pred_annotations.json` in root directory. 

Arguments include:

- `--model` - default: 'YOLO', could also choose 'ResNet18', 'ResNet50', 'Inception', 'DenseNet'
- `--cuda` - default: 'True'