from src.modules.module import Module


class TouchModule(Module):

    def __init__(self):
        super(TouchModule, self).__init__()
        self.touch_encoder = None
        self.touch_decoder = None
        self.touch_ae = self.touch_encoder(self.touch_encoder)

    def on_goal(self, *args):
        pass

    def on_observation(self, touch_image):
        e_touch = self.touch_encoder(touch_image)

        self.emit_observation(e_touch)

    def on_feedback(self, *args):
        touch_batch = sample(touch_rb)

        self.touch_ae.fit_batch([touch_batch, touch_batch])

