from typing import Tuple, Any
from PIL import Image 
from torchvision.datasets import Flowers102

class SafeFlowers102(Flowers102):
    """
    A safer wrapper around torchvision Flowers102.
    If an image fails to decode, it returns a tiny placeholder image
    and keeps count of decode errors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decode_errors =0

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        try:
            img, target = super().__getitem__(index)
            return img, int(target)
        except Exception:
            self.decode_errors += 1

            target = -1
            try:
                target = int(self._labels[index])
            except Exception:
                pass

            return Image.new("RGB", (1,1)), target