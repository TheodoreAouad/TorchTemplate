class Defreezer:

    def __init__(self, model=None):

        self.current_batch = 0
        self.model = model
        self.init_freeze()


    def init_freeze(self):
        pass

    def defreeze_epoch(self, epoch):
        pass


class FreezeFeaturesResNet(Defreezer):

    def init_freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True


class FreezeFeaturesVGG(Defreezer):

    def init_freeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True