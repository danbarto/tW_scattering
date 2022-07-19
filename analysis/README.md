# Analysis workflows

## Boosted Information tree

Make sure the BIT submodule is on the main branch (it should be).

Training with e.g. (benchmark run)

``` shell
ipython -i boosted_information_tree.py -- --version v31 --allBkg --max_label 100 --fit
```
--> this will not use weights during the training process.
