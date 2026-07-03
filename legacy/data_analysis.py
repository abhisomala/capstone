import matplotlib.pyplot as plt


# Epoch-wise results
epochs = list(range(1, 21))


# Loss and accuracy data from your results
loss = [65.0220, 35.9403, 28.7328, 25.8728, 21.8187, 20.8269, 20.7290, 18.2786,
        15.3641, 13.3840, 11.8104, 10.0832, 8.6414, 8.0569, 7.9388, 5.5727,
        4.1028, 4.0271, 5.4363, 2.8033]


accuracy = [95.16, 97.42, 97.79, 97.85, 98.54, 98.53, 98.61, 98.78, 99.05, 99.26,
            99.35, 99.50, 99.59, 99.61, 99.56, 99.80, 99.86, 99.85, 99.61, 99.89]


# Create a figure with two subplots: one for loss and one for accuracy
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))


# Subplot 1: Loss
ax1.plot(epochs, loss, '-o', color='#1f77b4', linewidth=3, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax1.set_title('Training Loss Over Epochs', fontsize=16, fontweight='bold')
ax1.set_xlabel('Epochs', fontsize=14)
ax1.set_ylabel('Loss', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.7, color='gray')  # Lighter gridlines for aesthetics
ax1.set_facecolor('#f5f5f5')  # Light background color for the subplot


# Subplot 2: Accuracy
ax2.plot(epochs, accuracy, '-s', color='#d62728', linewidth=3, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax2.set_title('Training Accuracy Over Epochs', fontsize=16, fontweight='bold')
ax2.set_xlabel('Epochs', fontsize=14)
ax2.set_ylabel('Accuracy (%)', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.7, color='gray')  
ax2.set_facecolor('#f5f5f5')  


plt.tight_layout()
plt.show()



