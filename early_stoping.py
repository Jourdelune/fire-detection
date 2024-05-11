class EarlyStopper:
    """Class that implement early stopping"""

    def __init__(self, patience: int = 1, min_delta: int = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss: float) -> bool:
        """Method that indicate if the the training should be stopped

        :param validation_loss: a float that indicate the loss on the test set.
        :type validation_loss: float
        :return: boolean that indicate if the training must be stopped.
        :rtype: bool
        """

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
