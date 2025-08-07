import pandas as pd
import matplotlib.pyplot as plt

loss_df = pd.read_csv("../../../cmake-build-release/loss_history.csv")
acc_df = pd.read_csv("../../../cmake-build-release/accuracy_history.csv")

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Plot loss
axs[0].plot(loss_df['epoch'], loss_df['train_loss'], label='Train Loss')
axs[0].plot(loss_df['epoch'], loss_df['valid_loss'], label='Valid Loss')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[0].set_title('Training and Validation Loss')

# Plot accuracy
axs[1].plot(acc_df['epoch'], acc_df['train_acc'], label='Train Accuracy')
axs[1].plot(acc_df['epoch'], acc_df['valid_acc'], label='Valid Accuracy')
axs[1].set_ylabel('Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].legend()
axs[1].set_title('Training and Validation Accuracy')

plt.tight_layout()
plt.savefig("training_metrics.png")
# plt.show()
