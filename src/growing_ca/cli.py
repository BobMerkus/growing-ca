import dotenv
from pydantic_settings import CliApp, CliSubCommand, BaseSettings, SettingsConfigDict

from growing_ca.train import TrainCaModel
from growing_ca.main_pygame_dl import VisualizeCaModel


class GrowingCa(BaseSettings):
    """Growing-Neural-Cellular-Automata

    Reference:
    Mordvintsev, et al., "Growing Neural Cellular Automata", Distill, 2020.

    Usage:
    ```
    growing-ca train --help
    growing-ca visualize --help
    ```
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    train: CliSubCommand[TrainCaModel]
    visualize: CliSubCommand[VisualizeCaModel]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    dotenv.load_dotenv()  # apply environment variables for platform specifc settings
    cmd = CliApp.run(GrowingCa)
    print(cmd.model_dump())


if __name__ == "__main__":
    main()
