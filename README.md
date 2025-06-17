# Dog & Cat Classifier

A simple CNN network that can distinguish kitties and puppies.

Use `pip` to install dependencies:

> We recommend using `conda` to create a virtual environment and then to perform this task.

```bash
pip install -r requirements.txt
```

You can run the train script to train your own weights:

```bash
python train.py
```

Run the evaluation GUI and test your own photos of kitties and puppies:

```bash
python test.py
```

## Dataset

We use the [Cats and Dogs Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset).

We made both dogs' and cats' 0~2499.jpg images be the evaluation sets. So the overall dataset hierachy looks like this:

```
+ Cat&Dog
+- train
 +- Cat
  +-- 2500.jpg
  +-- 2501.jpg
  +-- ...
 +- Dog
  +-- 2500.jpg
  +-- 2501.jpg
  +-- ...
+- eval
 +- Cat
  +-- 0.jpg
  +-- 1.jpg
  +-- ...
 +- Dog
  +-- 0.jpg
  +-- 1.jpg
  +-- ...
```

> The original dataset contains some corrupted images. We have removed them and recompiled a clean set. You can download it from this [link](https://drive.google.com/file/d/1xhCYij0z735-hoZEPQAhGEAIQ566VXXZ/view?usp=drive_link). Or you can leverage `remove_single_channel_images.py` to do the manual cleaning.

## Weights

We provide our final training weights using `train.py`. You can download [it](https://drive.google.com/file/d/1D_MRaqkT4kbAna6_HA3ldWtMwORcR1bv/view?usp=drive_link) and place it in `./checkpoints/` folder.

