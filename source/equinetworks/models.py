import os
from typing import Any, Dict, Optional

import requests
import torch
import torch.hub
import torchvision.models
from escnn import gspaces
from torchvision.datasets.utils import download_file_from_google_drive

from equivision.e2resnet import E2BasicBlock, E2BottleNeck, E2ResNet

WEIGHT_FILE_IDS = {
    "c4resnet18": "c4resnet18.pt",
    "c8resnet18": "c8resnet18.pt",
    "c4resnet50": "c4resnet50.pt",
    "c8resnet50": "c8resnet50.pt",
    "c4resnet101": "c4resnet101.pt",
}


def load_state_dict_from_google_drive(
    file_id: str, model_dir: Optional[str] = None
) -> Dict[str, Any]:
    hub_dir = torch.hub.get_dir()

    model_dir = os.path.join("checkpoints")

    os.makedirs(model_dir, exist_ok=True)

    cached_file = os.path.join(model_dir, file_id)


    if not os.path.exists(cached_file):
        print("Can not Find checkpoints")

    return torch.load(cached_file)


def resnet18(*args, **kwargs):
    model = torchvision.models.resnet18()
    model.name = "resnet18"
    return model


def resnet50(*args, **kwargs):
    model = torchvision.models.resnet50()
    model.name = "resnet50"
    return model


def resnet101(*args, **kwargs):
    model = torchvision.models.resnet101()
    model.name = "resnet101"
    return model


def c1resnet18(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.trivialOnR2(),
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000,
        base_width=78 if fixed_params else 64,
        initialize=initialize,
    )
    model.name = "c1resnet18"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_google_drive(WEIGHT_FILE_IDS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def d1resnet18(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.flip2dOnR2(),
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000,
        base_width=55 if fixed_params else 32,
        initialize=initialize,
    )
    model.name = "d1resnet18"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_google_drive(WEIGHT_FILE_IDS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def c4resnet18(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.rot2dOnR2(N=4),
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000,
        base_width=39 if fixed_params else 16,
        initialize=initialize,
    )
    model.name = "c4resnet18"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_google_drive(WEIGHT_FILE_IDS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model.cuda()


def d4resnet18(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.flipRot2dOnR2(N=4),
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000,
        base_width=28 if fixed_params else 8,
        initialize=initialize,
    )
    model.name = "d4resnet18"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_google_drive(WEIGHT_FILE_IDS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def c8resnet18(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.rot2dOnR2(N=8),
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000,
        base_width=28 if fixed_params else 8,
        initialize=initialize,
    )
    model.name = "c8resnet18"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_google_drive(WEIGHT_FILE_IDS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def c1resnet50(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.trivialOnR2(),
        block=E2BottleNeck,
        layers=[3, 4, 6, 3],
        num_classes=1000,
        base_width=70 if fixed_params else 64,
        initialize=initialize,
    )
    model.name = "c1resnet50"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_google_drive(WEIGHT_FILE_IDS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def d1resnet50(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.flip2dOnR2(),
        block=E2BottleNeck,
        layers=[3, 4, 6, 3],
        num_classes=1000,
        base_width=50 if fixed_params else 32,
        initialize=initialize,
    )
    model.name = "d1resnet50"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_google_drive(WEIGHT_FILE_IDS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def c4resnet50(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.rot2dOnR2(N=4),
        block=E2BottleNeck,
        layers=[3, 4, 6, 3],
        num_classes=1000,
        base_width=35 if fixed_params else 16,
        initialize=initialize,
    )
    model.name = "c4resnet50"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_google_drive(WEIGHT_FILE_IDS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def d4resnet50(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.flipRot2dOnR2(N=4),
        block=E2BottleNeck,
        layers=[3, 4, 6, 3],
        num_classes=1000,
        base_width=25 if fixed_params else 8,
        initialize=initialize,
    )
    model.name = "d4resnet50"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_google_drive(WEIGHT_FILE_IDS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def c8resnet50(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.rot2dOnR2(N=8),
        block=E2BottleNeck,
        layers=[3, 4, 6, 3],
        num_classes=1000,
        base_width=25 if fixed_params else 8,
        initialize=initialize,
    )
    model.name = "c8resnet50"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_google_drive(WEIGHT_FILE_IDS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def c1resnet101(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.trivialOnR2(),
        block=E2BottleNeck,
        layers=[3, 4, 23, 3],
        num_classes=1000,
        base_width=70 if fixed_params else 64,
        initialize=initialize,
    )
    model.name = "c1resnet101"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_google_drive(WEIGHT_FILE_IDS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def d1resnet101(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.flip2dOnR2(),
        block=E2BottleNeck,
        layers=[3, 4, 23, 3],
        num_classes=1000,
        base_width=50 if fixed_params else 32,
        initialize=initialize,
    )
    model.name = "d1resnet101"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_google_drive(WEIGHT_FILE_IDS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def c4resnet101(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.rot2dOnR2(N=4),
        block=E2BottleNeck,
        layers=[3, 4, 23, 3],
        num_classes=1000,
        base_width=35 if fixed_params else 16,
        initialize=initialize,
    )
    model.name = "c4resnet101"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_google_drive(WEIGHT_FILE_IDS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def d4resnet101(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.flipRot2dOnR2(N=4),
        block=E2BottleNeck,
        layers=[3, 4, 23, 3],
        num_classes=1000,
        base_width=25 if fixed_params else 8,
        initialize=initialize,
    )
    model.name = "d4resnet101"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_google_drive(WEIGHT_FILE_IDS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def c8resnet101(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.rot2dOnR2(N=8),
        block=E2BottleNeck,
        layers=[3, 4, 23, 3],
        num_classes=1000,
        base_width=25 if fixed_params else 8,
        initialize=initialize,
    )
    model.name = "c8resnet101"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_google_drive(WEIGHT_FILE_IDS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def count_params(m: torch.nn.Module):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def measure_inference(
    model: torch.nn.Module, batch_size: int = 1, repetitions: int = 100
):
    dummy_input = torch.randn((batch_size, 3, 224, 224), dtype=torch.float32).cuda()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = []

    for _ in range(10):
        _ = model(dummy_input)

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))

    return sum(timings) / len(timings)


def get_model(model_name: str, **config):
    fn = {
        "resnet18": resnet18,
        "c1resnet18": c1resnet18,
        "d1resnet18": d1resnet18,
        "c4resnet18": c4resnet18,
        "d4resnet18": d4resnet18,
        "c8resnet18": c8resnet18,
        "resnet50": resnet50,
        "c1resnet50": c1resnet50,
        "d1resnet50": d1resnet50,
        "c4resnet50": c4resnet50,
        "d4resnet50": d4resnet50,
        "c8resnet50": c8resnet50,
        "resnet101": resnet101,
        "c1resnet101": c1resnet101,
        "d1resnet101": d1resnet101,
        "c4resnet101": c4resnet101,
        "d4resnet101": d4resnet101,
        "c8resnet101": c8resnet101,
    }[model_name]

    return fn(**config)


if __name__ == "__main__":
    for layers in [18, 50, 101]:
        for group in ["", "d1", "c4", "d4", "c8"]:
            model = (
                eval(group + "resnet" + str(layers))(
                    initialize=False, fixed_params=False
                )
                .cuda()
                .eval()
            )
            mem = torch.cuda.memory_allocated()
            inf_time = measure_inference(model, batch_size=32)
            print(
                f"{model.name}: {count_params(model)*1e-6:.1f}M |"
                f" {inf_time:.1f}ms | {mem * 1e-9:.2f}Gb"
            )
        print()
