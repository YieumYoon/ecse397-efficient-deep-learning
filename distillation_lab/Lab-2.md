# Lab 2: Knowledge Distillation with ResNet-18 and ViT-Tiny

ECSE 397/600: Efficient Deep Learning

Instructor: Prof. Gourav Datta Case Western Reserve University

Deadline: October 15, 11:59 pm EST

### 1 Objective

The objective of this lab is to implement Knowledge Distillation (KD) to train compact student models guided by larger teacher models. You will work with:

- CNN: Distill knowledge from ResNet-18 (teacher) into a smaller CNN student (ResNet-8).
- ViT: Distill knowledge from ViT-Tiny (teacher) into a shallower, smaller ViT student.

### 2 Models and Dataset

### 2.1 Dataset: CIFAR-10

- 50,000 training images, 10,000 test images
- 32×32 RGB images, 10 classes

#### 2.2 Models

### Teacher Models:

- ResNet-18 from torchvision.models.
- ViT-Tiny from timm or Hugging Face Transformers.

#### Student Models:

- ResNet-8: A smaller ResNet variant with fewer layers (2 conv blocks per stage instead of 4). GitHub reference: [https://github.com/akamaster/pytorch\\_resnet\\_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10)
- ViT-Student: A reduced ViT-Tiny (e.g., 6 transformer layers instead of 12, embedding size reduced to 192). Example lightweight ViT repo: <https://github.com/FrancescoSaverioZuppichini/ViT>

# 3 Directory Structure (Use the utils directory only if needed.

```
distillation_lab/
|
+-- data/
| -- dataloader.py
|
+-- models/
| -- teacher_resnet.py
| -- student_resnet.py
| -- teacher_vit.py
| -- student_vit.py
|
+-- models_saved/
| -- cnn_teacher.pth
| -- cnn_student_no_kd.pth
| -- cnn_student_with_kd.pth
| -- vit_teacher.pth
| -- vit_student_no_kd.pth
| -- vit_student_with_kd.pth
|
+-- train/
| -- train_teacher.py
| -- distill.py
|
+-- inference/
| -- test.py
|
+-- utils/
| -- kd_losses.py
|
+-- main.py
```

# 4 Assignment Tasks

#### 4.1 Task 0: DataLoader

• Same as the Lab 1.

#### 4.2 Task 1: Teacher Models

- Train/fine-tune ResNet-18 and ViT-Tiny on CIFAR-10.
- Save their accuracies and checkpoints.

#### 4.3 Task 2: Student Models

- Implement ResNet-8 and a smaller ViT student model.
- Train each student model without KD as baselines.

#### 4.4 Task 3: Knowledge Distillation Loss

- Implement KD in train/distill.py. At minimum:
  - 1. Soft-target KD: KL divergence between teacher and student logits at temperature τ .
  - 2. Combined Loss:

$$\mathcal{L}_{KD} = \alpha \cdot \mathcal{L}_{CE}(y, p_s) + (1 - \alpha) \cdot \tau^2 \cdot KL(p_t^{\tau} \parallel p_s^{\tau})$$

• Optionally: Add feature-based KD by aligning intermediate representations. This technique, known as FitNets (see Resources below), allows the student to match hidden feature maps of the teacher in addition to logits.

#### Resources for Distillation:

- Hinton's original KD paper (2015): [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- PyTorch KD example implementation: [https://github.com/peterliht/knowledge-distillation](https://github.com/peterliht/knowledge-distillation-pytorch)[pytorch](https://github.com/peterliht/knowledge-distillation-pytorch)
- FitNets: Hints for Thin Deep Nets (2015): <https://arxiv.org/abs/1412.6550>
- Hugging Face DistilBERT docs (for NLP, but same concept): [https://huggingface.co/docs/transformers/model\\_doc/distilbert](https://huggingface.co/docs/ /transformers/model_doc/distilbert)

#### 4.5 Task 4: Training with Distillation

- Train students with KD.
- Experiment with different α and τ values.
- Compare results with and without KD.

#### 4.6 Task 5: Reporting

• Generate a report.json file:

Listing 1: Example report.json

```
{
    " cnn ": {
         " teacher_accuracy ": 0.912 ,
         " student_accuracy_without_kd ": 0.845 ,
         " student_accuracy_with_kd ": 0.872
    } ,
    " vit ": {
         " teacher_accuracy ": 0.927 ,
         " student_accuracy_without_kd ": 0.812 ,
         " student_accuracy_with_kd ": 0.854
    }
}
```

• Save checkpoints in models\_saved/:

```
cnn_teacher.pth
cnn_student_no_kd.pth
cnn_student_with_kd.pth
vit_teacher.pth
vit_student_no_kd.pth
vit_student_with_kd.pth
```

# 5 Submission Guidelines

- 1. Compress distillation\_lab/ into studentID\_distillation\_lab.zip.
- 2. Include report.json, and models\_saved/.
- 3. Submit through Canvas before the deadline.