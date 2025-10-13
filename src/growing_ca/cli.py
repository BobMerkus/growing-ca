from pydantic_settings import CliApp, CliSubCommand, BaseSettings, SettingsConfigDict
import dotenv

from growing_ca.train import TrainCaModel
from growing_ca.main_pygame_dl import VisualizeCaModel
from growing_ca.convert_onnx import ConvertOnnx


class GrowingCa(BaseSettings):
    """Growing-Neural-Cellular-Automata

    Reference:
    Mordvintsev, et al., "Growing Neural Cellular Automata", Distill, 2020.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    train: CliSubCommand[TrainCaModel]
    visualize: CliSubCommand[VisualizeCaModel]
    convert_onnx: CliSubCommand[ConvertOnnx]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    dotenv.load_dotenv()  # Load .env file for environment variables related to pytorch
    cmd = CliApp.run(GrowingCa)
    print(cmd.model_dump())


if __name__ == "__main__":
    main()
