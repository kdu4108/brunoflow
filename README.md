## Using Brunoflow
We recommend installing brunoflow into a fresh conda environment (Python3.9+). To create and then activate a conda environment, run `conda create --name bflow python=3.9` and then `conda activate bflow`.


Then, to  install brunoflow, clone the repo and run `pip install -e .` (make sure this is within your new conda env). Since this is an editable install, you should be able to edit files directly and then those changes should automatically update when you use the library.

### Example code
```
import brunoflow as bf
x = bf.Node(val=10, name="x")
output = x ** 2
output.backprop()
print(x.grad) # expected: 20.0
print(x.compute_entropy().val) # expected: 0.0
```

## Modifying Brunoflow
### Pre-commit
Run `pre-commit install` to get nice pre-commit hooks for linting/formatting.

### Details on how Brunoflow works
See the README inside of the `brunoflow` directory.
