from old_src.modules.module import Module


class VisionModule(Module):

    def __init__(self):
        super(VisionModule, self).__init__()
        self.vision_encoder = None

    def on_feedback(self, *args):
        pass

    def on_observation(self, visual_image):
        e_vision = self.vision_encoder(visual_image)

        self.emit_observation(e_vision)

    def on_feedback(self, *args):
        pass
