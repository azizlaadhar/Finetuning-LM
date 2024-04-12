#### `tensorboard/`

In this directory, your tensorboard logs should be stored when you're training/evaluating the models. Training and evaluation functions already include the code to push logs to tensorboard. You just have to make sure that you don't change them and hand over the tensorboard directory at the end.

We suggest you inspect the tensorboard logs after your training and evaluation to make sure they look as expected.
You can run the tensorboard interface using the following command:
```sh
tensorboard --logdir=./tensorboard
```

_You can find more information on Tensorboard with pytorch [here](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)._