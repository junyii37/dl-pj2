#%%
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from VGG_BatchNorm.models.vgg import VGG_A
from VGG_BatchNorm.models.vgg import VGG_A_BatchNorm # you need to implement this network
from VGG_BatchNorm.data.loaders import get_cifar_loader

#%%
# ## Constants (parameters) initialization
# device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

# add our package dir to path
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
# device_id = device_id
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# device = torch.device("cuda:{}".format(3) if torch.cuda.is_available() else "cpu")
# print(device)
# print(torch.cuda.get_device_name(3))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
for X,y in train_loader:
    print(X[0])
    print(y[0])
    print(X[0].shape)
    img = np.transpose(X[0], [1,2,0])
    plt.imshow(img*0.5 + 0.5)
    plt.savefig('sample.png')
    print(X[0].max())
    print(X[0].min())
    break



# This function is used to calculate the accuracy of model classification
def get_accuracy():
    pass

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)

    # Record average loss, training accuracy and validation accuracy of each epoch
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n

    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)  # num of batches
    losses_list = []
    grads = []

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # You may need to record some variable values here
            # if you want to get loss gradient, use

            # Update the model
            loss.backward()
            optimizer.step()

            # Record the grad and loss of each step
            grad.append(model.classifier[4].weight.grad.clone())
            loss_list.append(loss.item())
            learning_curve[epoch] += loss.item()


        losses_list.append(loss_list)
        grads.append(grad)
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))

        learning_curve[epoch] /= batches_n
        axes[0].plot(learning_curve)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss Curve')
        axes[0].legend()

        # Test your model and save figure here (not required)
        # remember to use
        model.eval()
        with torch.no_grad():
            train_accuracy_curve[epoch] = 0
            total_train = 0
            for x_train, y_train in train_loader:
                x_train, y_train = x_train.to(device), y_train.to(device)
                logits_train = model(x_train)
                preds_train = logits_train.argmax(dim=1)
                train_accuracy_curve[epoch] += (preds_train == y_train).sum().item()
                total_train += len(y_train)
            train_accuracy_curve[epoch] /= total_train

            val_accuracy_curve[epoch] = 0
            total_val = 0
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                logits_val = model(x_val)
                preds_val = logits_val.argmax(dim=1)
                val_accuracy_curve[epoch] += (preds_val == y_val).sum().item()
                total_val += len(y_val)
            val_accuracy_curve[epoch] /= total_val

        axes[1].plot(train_accuracy_curve, label='Train Acc')
        axes[1].plot(val_accuracy_curve, label='Val Acc')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training & Validation Accuracy")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

        if best_model_path is not None and val_accuracy_curve[epoch] > max_val_accuracy:
            max_val_accuracy = val_accuracy_curve[epoch]
            max_val_accuracy_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

        model.train()

    print(f"Best validation accuracy: {max_val_accuracy:.4f} at epoch {max_val_accuracy_epoch}")

    return losses_list, grads
#%%

# Train your model
# feel free to modify
epo = 20
loss_save_path = 'reports'
grad_save_path = 'reports'

set_random_seeds(seed_value=2020, device=device)
model = VGG_A()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()
loss, grads = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
np.savetxt(os.path.join(loss_save_path, 'loss.txt'), loss, fmt='%s', delimiter=' ')
np.savetxt(os.path.join(grad_save_path, 'grads.txt'), grads, fmt='%s', delimiter=' ')

# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
min_curve = []
max_curve = []

loss_arr = np.array(loss)
min_curve = np.min(loss_arr, axis=0).tolist()
max_curve = np.max(loss_arr, axis=0).tolist()

# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape():
    # --------------------
    # Add your code
    # 确保 min_curve 和 max_curve 在此作用域可访问
    steps = np.arange(len(min_curve))
    plt.figure(figsize=(8, 5))
    # 画出最小 loss 曲线和最大 loss 曲线
    plt.plot(steps, min_curve, label='Min Loss across epochs')
    plt.plot(steps, max_curve, label='Max Loss across epochs')
    # 填充两条曲线之间的面积
    plt.fill_between(steps, min_curve, max_curve, alpha=0.3)
    plt.xlabel('Step (Batch Index)')
    plt.ylabel('Loss')
    plt.title('Loss Landscape Across Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # --------------------
    pass

# 在训练结束后，调用绘图函数即可看到损失地形
plot_loss_landscape()