# Lab 1: Custom Pruning of ResNet-18 and ViT-Tiny

ECSE 397/600: Efficient Deep Learning

Instructor: Prof. Gourav Datta 

Case Western Reserve University

Deadline: September 29, 11:59 pm EST

# 1 Objective

The objective of this lab is to explore custom neural network pruning techniques to compress modern architectures while maintaining high performance. You will work with:

- A standard CNN (ResNet-18)
- A transformer-based model (ViT-Tiny)

on the CIFAR-10 dataset.

Extra credit will be awarded to the student who achieves the highest pruning percentage (weight sparsity) while still meeting the accuracy requirements.

# 2 Models and Dataset

### 2.1 Dataset: CIFAR-10

- 50,000 training images, 10,000 test images
- 32×32 RGB images, 10 classes
- Reference: [PyTorch CIFAR-10 Documentation](https://pytorch.org/vision/stable/datasets.html#cifar)

## 2.2 Models

ResNet-18 (CNN): Use the standard PyTorch ResNet-18 implementation:

- GitHub Reference: [TorchVision ResNet Code](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
- Pre-trained ImageNet weights available through torchvision.models.resnet18(pretrained=True)

ViT-Tiny (Vision Transformer): Two versions must be implemented:

- 1. Pre-trained on JFT-21M: Start from a high-quality ViT-Tiny pre-trained model for transfer learning.
  - GitHub Reference: [timm \(PyTorch Image Models Library\)](https://github.com/rwightman/pytorch-image-models)
  - Example code: timm.create\_model('vit\_tiny\_patch16\_224', pretrained=True)
- 2. Train from Scratch: Build and train ViT-Tiny directly on CIFAR-10 without pre-trained weights.

# 3 Directory Structure

The code must be organized into the following structure. Feel free to use additional folders for helper files, plots, etc.

```
pruning_lab/
|
+-- data/
| -- dataloader.py # CIFAR-10 loading and augmentations
|
+-- models/
| -- resnet18.py # ResNet-18 implementation
| -- vit_tiny.py # ViT-Tiny implementation
|
+-- train/
| -- train_loop.py # Training and validation loops
| -- prune.py # Custom pruning algorithms
|
+-- inference/
| -- test.py # Evaluate pruned models
|
+-- utils/ # incudes all the scripts
|
+-- main.py # Command-line entry point
```

# 4 Assignment Tasks

## 4.1 Task 1: Data Loader

- Implement a PyTorch DataLoader for CIFAR-10 in data/dataloader.py.
- Include augmentations such as random cropping, horizontal flipping, and normalization.
- Provide a function get\_loaders(batch\_size) that returns both training and test dataloaders.

### 4.2 Task 2: ResNet-18

- Implement or import ResNet-18 from torchvision.models.
- Train it on CIFAR-10 to achieve 90%+ test accuracy before pruning.

### 4.3 Task 3: ViT-Tiny

- Fine-tune a ViT-Tiny pre-trained on JFT-21M to achieve 92%+ accuracy.
- Train a separate ViT-Tiny from scratch that achieves at least 85% accuracy.

Resources: You may choose any library to implement and train your models. Below are suggested resources and instructions for each option.

• ResNet-18 and CIFAR-10 DataLoader (torchvision):

- GitHub: <https://github.com/pytorch/vision>
- Why use torchvision?
  - ∗ Provides ready-to-use implementations of classic CNN architectures like ResNet-18.
  - ∗ Built-in CIFAR-10 dataset loader with augmentations.
- How to load a pre-trained ResNet-18:

```
import torchvision.models as models
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 10) # For CIFAR-10
```

– Where to find model files: The PyTorch implementation of ResNet-18 is located at:

```
torchvision/models/resnet.py
```

You can copy this file into your models/ directory and modify it to fit your pruning experiments.

– How to set up a CIFAR-10 DataLoader:

```
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = CIFAR10(root='./data', train=True,
                        download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False,
                       download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
```

- CIFAR-10 tutorial: [https://pytorch.org/tutorials/beginner/blitz/cifar10\\_tutorial.html](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- Vision Transformer (ViT) using timm:
  - GitHub: <https://github.com/rwightman/pytorch-image-models>
  - Why use timm?
    - ∗ Provides a wide variety of Vision Transformers, including ViT-Tiny.
    - ∗ Includes pre-trained weights on datasets like ImageNet-21k and JFT-300M.
  - How to load a pre-trained ViT-Tiny:

```
import timm
model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
model.head = torch.nn.Linear(model.head.in_features, 10) # CIFAR-10
```

– Where to find model files: You can copy ViT definitions from:

```
timm/models/vision_transformer.py
timm/models/resnet.py
```

Copy these into your models/ folder to modify them for pruning.

- Vision Transformer (ViT) using Hugging Face:
  - Docs: [https://huggingface.co/docs/transformers/model\\_doc/vit](https://huggingface.co/docs/transformers/model_doc/vit)
  - Pre-trained models: <https://huggingface.co/google>
  - Example: Load a pre-trained ViT:

```
from transformers import ViTForImageClassification
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=10 # CIFAR-10
)
```

- Google's Official Vision Transformer Repository:
  - GitHub: [https://github.com/google-research/vision\\_transformer](https://github.com/google-research/vision_transformer)
  - Contains the original JAX/Flax implementation of ViTs.
  - Useful for understanding the original model design or downloading official checkpoints.

### 4.4 Task 4: Custom Pruning Algorithm

- Implement your own custom pruning logic in train/prune.py.
- Torch's built-in pruning functions are NOT allowed. You must write the pruning code manually using standard PyTorch tensor operations.
- You are required to implement and experiment with both:
  - 1. Unstructured pruning: Prune individual weights across the network without regard to their spatial or channel grouping. This generally achieves a much higher pruning ratio but does not directly translate to hardware speedup.
  - 2. Structured pruning (channel-wise only): Prune entire channels (filters) in convolutional layers or attention heads in ViTs. This produces more hardware-friendly models but typically results in lower pruning ratios.
- You are free to choose how to decide which weights or channels to prune. Some example strategies include:
  - Magnitude-based pruning: Remove the smallest weights or channels by absolute value.

- Iterative pruning: Gradually prune the network over several epochs, retraining between pruning steps.
- Hessian-based pruning: Use second-order information (loss sensitivity) to determine importance.
- Custom scoring methods: Design your own criteria for pruning importance.
- For each model (ResNet-18 and ViT-Tiny), you must:
  - 1. Perform unstructured pruning and record:
    - Final pruning ratio (percentage of weights removed).
    - Test accuracy after pruning and fine-tuning.
  - 2. Perform structured channel-wise pruning and record:
    - Final pruning ratio (percentage of channels removed).
    - Test accuracy after pruning and fine-tuning.

### 4.5 Task 5: Accuracy and Pruning Requirements

- After pruning, it is recommended that the model satisfies:
  - 1. CNN (ResNet-18): Test accuracy ≥ 85%, ≥ 70% unstructured weight sparsity, and ≥ 25% structured weight sparsity.
  - 2. ViT-Tiny (Pre-trained): Test accuracy ≥ 88%, ≥ 70% unstructured weight sparsity, and ≥ 25% structured weight sparsity..
  - 3. ViT-Tiny (Scratch): Test accuracy ≥ 80% and ≥ 70% unstructured weight sparsity, and ≥ 25% structured weight sparsity.
- Even if you do not fully achieve the target sparsity ratios, as long as your pruning method is well-designed and demonstrates a sound, reasonable approach that is clear from your code and documentation, you will receive full points.

### 4.6 Task 6: Reporting

- You must generate a report.json file that includes:
  - 1. Initial accuracies of the unpruned models:
    - Final test accuracy of the CNN (ResNet-18) before pruning.
    - Final test accuracy of the ViT-Tiny (Pre-trained) before pruning.
    - Final test accuracy of the ViT-Tiny (Scratch) before pruning.
  - 2. Results for both unstructured and structured (channel-wise) pruning:
    - Original model accuracy before pruning (same as initial accuracy for that model).
    - Final accuracy after pruning and fine-tuning.
    - Final pruning ratio (percentage of weights or channels removed).

Listing 1: Example format of the report.json file:

```
{
    " initial_accuracies ": {
         " cnn_before_pruning ": 0.912 ,
         " vit_before_pruning ": 0.927 ,
         " vit_scratch_before_pruning ": 0.862
    } ,
    " unstructured_pruning ": {
         " cnn ": {
              " original_accuracy ": 0.912 ,
              " pruned_accuracy ": 0.861 ,
              " pruning_percentage ": 93.5
         } ,
         " vit ": {
              " original_accuracy ": 0.927 ,
              " pruned_accuracy ": 0.852 ,
              " pruning_percentage ": 92.0
         } ,
         " vit_scratch ": {
              " original_accuracy ": 0.862 ,
              " pruned_accuracy ": 0.815 ,
              " pruning_percentage ": 91.0
         }
    } ,
    " structured_pruning ": {
         " cnn ": {
              " original_accuracy ": 0.912 ,
              " pruned_accuracy ": 0.884 ,
              " pruning_percentage ": 70.0
         } ,
         " vit ": {
              " original_accuracy ": 0.927 ,
              " pruned_accuracy ": 0.872 ,
              " pruning_percentage ": 68.5
         } ,
         " vit_scratch ": {
              " original_accuracy ": 0.862 ,
              " pruned_accuracy ": 0.821 ,
              " pruning_percentage ": 65.0
         }
    }
}
```

- This JSON file must be saved in the root of your submission folder as report.json.
- Compress your entire pruning\_lab/ folder into a .zip file before submission. The folder must strictly follow the required directory structure:

```
pruning_lab/
|
+-- data/
+-- models/
+-- train/
+-- inference/
+-- utils/
+-- main.py
+-- report.json
+-- models_saved/ # Folder for all model checkpoints
```

- Inside the models\_saved/ folder, include the model checkpoint (.pth) files for:
  - 1. CNN (ResNet-18) before pruning
  - 2. ViT-Tiny (Pre-trained) before pruning
  - 3. ViT-Tiny (Scratch) before pruning
  - 4. CNN and both ViT models after structured pruning
  - 5. CNN and both ViT models after unstructured pruning
- Use the following naming convention for clarity:

```
# CNN checkpoints (3 total)
cnn_before_pruning.pth
cnn_after_structured_pruning.pth
cnn_after_unstructured_pruning.pth

# ViT-Tiny Pre-trained checkpoints (3 total)
vit_before_pruning.pth
vit_after_structured_pruning.pth
vit_after_unstructured_pruning.pth

# ViT-Tiny Scratch checkpoints (3 total)
vit_scratch_before_pruning.pth
vit_scratch_after_structured_pruning.pth
vit_scratch_after_unstructured_pruning.pth
```

**Total: 9 checkpoint files required**

# 5 Submission Guidelines

- 1. Compress the entire pruning\_lab folder into studentID\_pruning.zip.
- 2. (Optional) Include a README.md if you have additional files/folders that need explanation to run your code.
- 3. Submit through the course Canvas before the deadline.