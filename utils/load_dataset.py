from datasets import load_dataset
from PIL import Image

ds = load_dataset("uoft-cs/cifar10")
train_set = ds["train"]
test_set = ds["test"]
