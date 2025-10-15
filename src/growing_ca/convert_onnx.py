import torch

from growing_ca.core.model import CAModel
from pathlib import Path
from pydantic import BaseModel, Field


def convert_onnx(model_path: Path, onnx_model_path: Path | None = None) -> None:
    """Convert a PyTorch model to an ONNX model"""
    import onnx

    if onnx_model_path is None:
        onnx_model_path = model_path.with_suffix(".onnx")

    pytorch_model = CAModel(
        channel_n=16, fire_rate=0.5, device=torch.device("cpu"), hidden_size=128
    )
    pytorch_model.load_state_dict(torch.load(model_path))
    pytorch_model.eval()
    dummy_input = torch.zeros(1, 72, 72, 16)

    # Export to temporary location first
    temp_path = str(onnx_model_path) + ".tmp"
    torch.onnx.export(
        pytorch_model,
        (dummy_input,),
        temp_path,
        verbose=True,
        export_params=True,
        do_constant_folding=True,
    )

    # Load the model and convert external data to raw data
    model = onnx.load(temp_path, load_external_data=True)

    # Save without external data
    onnx.save(model, str(onnx_model_path), save_as_external_data=False)

    # Clean up temp files
    import os

    if os.path.exists(temp_path):
        os.remove(temp_path)
    temp_data = temp_path + ".data"
    if os.path.exists(temp_data):
        os.remove(temp_data)

    print(f"Converted model saved to {onnx_model_path} (standalone, no external data)")


class ConvertOnnx(BaseModel):
    """Convert a PyTorch model to an ONNX model"""

    model_path: Path = Field(description="Path to the model to convert")
    onnx_model_path: Path | None = Field(
        default=None,
        description="Path to save the converted model (optional, defaults to <model_path>.onnx)",
    )

    def cli_cmd(self) -> None:
        convert_onnx(self.model_path, self.onnx_model_path)


if __name__ == "__main__":
    base_path = Path("models")
    for model_path in base_path.glob("*.pth"):
        onnx_model_path = model_path.with_suffix(".onnx")
        convert_onnx(model_path, onnx_model_path)
