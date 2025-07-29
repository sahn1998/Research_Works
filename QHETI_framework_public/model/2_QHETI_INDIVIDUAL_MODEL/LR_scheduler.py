from tensorflow.keras.callbacks import Callback

# Inherit from Keras' Callback class to create a custom learning rate scheduler
class RollingWindowLRScheduler(Callback):
    def __init__(self, window=100, factor=0.5, min_lr=1e-6, patience=1, cooldown=100, verbose=1):
        super().__init__()
        self.window = window                  # Number of epochs in each rolling window
        self.factor = factor                  # Factor to reduce the learning rate by (e.g., 0.5 = halve it)
        self.min_lr = min_lr                  # Lower bound for the learning rate
        self.patience = patience              # Number of bad windows before reducing the learning rate
        self.cooldown = cooldown              # Minimum number of epochs to wait after reducing LR before checking again
        self.verbose = verbose                # If 1, print updates when LR is reduced

        self.val_losses = []                  # Store validation loss for each epoch
        self.wait = 0                         # Count of consecutive "bad" windows (no improvement)
        self.cooldown_counter = 0             # Counter to enforce cooldown period after LR reduction

    # Called at the end of each epoch during training
    def on_epoch_end(self, epoch, logs=None):
        # Initialization logic omitted