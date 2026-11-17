from typing import Any

from torch import Tensor

from lwfm.core.types import Batch, Port


class PortError(KeyError):
    pass


def resolve(port: Port, batch: Batch, outputs: dict[Port, Tensor]) -> Any:
    if port in outputs:
        return outputs[port]
    head, *keys = port.split(".")
    value: Any = batch
    try:
        if head != "batch" or not keys:
            raise KeyError(head)
        for key in keys:
            value = value[key]
    except KeyError:
        raise PortError(f"no port {port!r}; node outputs: {sorted(outputs)}") from None
    return value


def resolve_all(
    binding: dict[str, Port], batch: Batch, outputs: dict[Port, Tensor]
) -> dict[str, Any]:
    return {arg: resolve(port, batch, outputs) for arg, port in binding.items()}
