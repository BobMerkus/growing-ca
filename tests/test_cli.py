from unittest.mock import MagicMock, patch


from growing_ca.cli import GrowingCa, main


class TestGrowingCa:
    """Test GrowingCa CLI class."""

    def test_model_class_exists(self):
        """Test GrowingCa class exists and has cli_cmd method."""
        assert GrowingCa is not None
        assert hasattr(GrowingCa, "cli_cmd")
        # train and visualize are CliSubCommand fields, not direct attributes
        assert "train" in GrowingCa.model_fields
        assert "visualize" in GrowingCa.model_fields

    @patch("growing_ca.cli.CliApp.run_subcommand")
    def test_cli_cmd_calls_run_subcommand(self, mock_run_subcommand):
        """Test that cli_cmd calls CliApp.run_subcommand."""
        # Create a mock instance
        mock_instance = MagicMock(spec=GrowingCa)
        GrowingCa.cli_cmd(mock_instance)
        mock_run_subcommand.assert_called_once_with(mock_instance)


class TestMain:
    """Test main function."""

    @patch("growing_ca.cli.dotenv.load_dotenv")
    @patch("growing_ca.cli.CliApp.run")
    def test_main_function(self, mock_run, mock_load_dotenv):
        """Test main function execution."""
        mock_cmd = MagicMock()
        mock_cmd.model_dump.return_value = {"test": "data"}
        mock_run.return_value = mock_cmd

        with patch("builtins.print") as mock_print:
            main()

        # Verify dotenv was loaded
        mock_load_dotenv.assert_called_once()
        # Verify CliApp.run was called with GrowingCa
        mock_run.assert_called_once_with(GrowingCa)
        # Verify model_dump was printed
        mock_print.assert_called_once_with({"test": "data"})

    @patch("growing_ca.cli.dotenv.load_dotenv")
    @patch("growing_ca.cli.CliApp.run")
    def test_main_dotenv_loaded_first(self, mock_run, mock_load_dotenv):
        """Test that dotenv is loaded before running CLI."""
        call_order = []
        mock_load_dotenv.side_effect = lambda: call_order.append("dotenv")
        mock_cmd = MagicMock()
        mock_cmd.model_dump.return_value = {}
        mock_run.side_effect = lambda x: (call_order.append("run"), mock_cmd)[1]

        with patch("builtins.print"):
            main()

        # Verify dotenv was loaded before CLI run
        assert call_order == ["dotenv", "run"]
