# ciFAIR-10 Dataset

ciFAIR is a variant of the popular CIFAR dataset, which uses a slightly modified test set avoiding near-duplicates between training and test data.
It comprises RGB images of size 32x32 spanning 10 classes of everyday objects.

CIFAR homepage: <https://www.cs.toronto.edu/~kriz/cifar.html>  
CIFAR paper: <https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf>

ciFAIR homepage: <https://cvjena.github.io/cifair/>  
ciFAIR Paper: <https://arxiv.org/abs/1902.00423>

![Example images from ciFAIR-10](example_images.png)


## Splits

We provide the following splits of the ciFAIR-10 dataset for testing small-data performance:

|   Split   | Total Images | Images / Class |
|:----------|-------------:|---------------:|
| train     |  300 / 3,000 |             30 |
| val       |  200 / 2,000 |             20 |
| trainval  |  500 / 5,000 |             50 |
| fulltrain |       50,000 |    5,000 / 500 |
| test      |       10,000 |    1,000 / 100 |

`fulltrain` and `test` correspond to the original data.
`train` comprises the first 30 training images from each class and `val` the following 20.
`trainval` is a combination of both.


## Baseline Performance

We achieved the following baseline performance using a Wide ResNet 16-8 trained on the `trainval` split and averaged over 10 runs.

| Dataset Variant | Accuracy |
|:----------------|---------:|
| ciFAIR-10       |   58.22% |


The ciFAIR-10 baseline can be reproduced with the provided training script as follows:

```bash
python scripts/train.py cifair10 \
    --architecture wrn-16-8 \
    --rand-shift 4 \
    --epochs 500 \
    --batch-size 10 \
    --lr 4.55e-3 \
    --weight-decay 5.29e-3 \
    --eval-interval 10
```


## Usage

This dataset can be loaded using `small_data.datasets.ciFAIR10`.
These data loaders will automatically download the data into the data directory if it is not present.
When using the `small_data.get_dataset` function, the respective dataset identifier is `"cifair10"`.
