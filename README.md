aloha

## What I learned

### Git and GitHub

I learned to clone a GitHub repository, stage files with git add, commit changes,
and push commits to the remote repository. I practiced creating the for_fun
branch, switching branches with checkout, and merging it into main. Since
for_fun was already an ancestor of main, the merge needed no conflict resolution.
I also checked out an earlier commit and returned to main. Git log records
commits, while reflog records local HEAD movements.

### Python environments and Hugging Face

I learned to use a Python virtual environment to isolate dependencies and record
package versions in requirements.txt. I used Hugging Face Transformers to load
a pretrained ResNet-18 model and its image processor. The .gitignore file keeps
the virtual environment out of Git, and model weights and datasets stay outside
the repository.

### ResNet inference on MNIST

I learned to convert grayscale MNIST images to RGB and resize and normalize them
for ResNet. I used evaluation mode and disabled gradients for batch inference
with MPS on my Mac. Processing 10,000 test images took about 18 seconds. The
raw class-index accuracy was 0.01%. ImageNet classes and MNIST digit labels have
different meanings, so this is not meaningful digit-recognition accuracy.
