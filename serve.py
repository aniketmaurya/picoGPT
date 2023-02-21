import lightning as L
from lightning.app.components import PythonServer
from lightning.app.components.serve import Text

from gpt2 import PicoGPT


class ServePicoGPT(PythonServer):
    def setup(self, *args, **kwargs) -> None:
        self._model = PicoGPT()

    def predict(self, request):
        return {"text": self._model(request.text)}


app = L.LightningApp(ServePicoGPT(input_type=Text, output_type=Text))
