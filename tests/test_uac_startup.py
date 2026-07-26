import os
import sys
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import main_ui


def _run_main(argv, *, relaunch_result=None):
    app = mock.MagicMock()
    app.exec.return_value = 37
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(main_ui, "QApplication", return_value=app) as app_type,
        mock.patch.object(main_ui, "FloatingWindow") as window_type,
        mock.patch("log_config.setup_logging"),
        mock.patch("crash_handler.install"),
        mock.patch.object(main_ui, "is_admin", return_value=False),
        mock.patch.object(
            main_ui,
            "_request_admin_relaunch",
            return_value=relaunch_result,
        ) as relaunch,
    ):
        result = main_ui.main()
    return result, app_type, window_type, relaunch


def test_default_start_does_not_request_process_wide_admin_rights():
    result, app_type, window_type, relaunch = _run_main(["AirControl.exe"])

    assert result == 37
    relaunch.assert_not_called()
    app_type.assert_called_once()
    window_type.return_value.show.assert_called_once()


def test_rejected_uac_request_falls_back_to_normal_privilege_startup():
    result, app_type, _window_type, relaunch = _run_main(
        ["AirControl.exe", "--elevate"],
        relaunch_result=False,
    )

    assert result == 37
    relaunch.assert_called_once()
    app_type.assert_called_once()


def test_successful_elevated_relaunch_exits_original_process():
    result, app_type, _window_type, relaunch = _run_main(
        ["AirControl.exe", "--elevate"],
        relaunch_result=True,
    )

    assert result == 0
    relaunch.assert_called_once()
    app_type.assert_not_called()


def test_shell_execute_error_code_is_not_treated_as_success():
    with mock.patch(
        "ctypes.windll.shell32.ShellExecuteW",
        return_value=5,
    ):
        assert not main_ui._request_admin_relaunch(["AirControl.exe", "--elevate"])


def test_shell_execute_success_code_is_accepted():
    with mock.patch(
        "ctypes.windll.shell32.ShellExecuteW",
        return_value=33,
    ):
        assert main_ui._request_admin_relaunch(["AirControl.exe", "--elevate"])


def test_incomplete_native_shutdown_uses_controlled_process_exit():
    app = mock.MagicMock()
    app.exec.return_value = 37
    window = mock.MagicMock()
    window._shutdown_incomplete = True
    with (
        mock.patch.object(sys, "argv", ["AirControl.exe"]),
        mock.patch.object(main_ui, "QApplication", return_value=app),
        mock.patch.object(main_ui, "FloatingWindow", return_value=window),
        mock.patch("log_config.setup_logging"),
        mock.patch("crash_handler.install"),
        mock.patch.object(main_ui.logging, "shutdown"),
        mock.patch.object(main_ui.os, "_exit") as forced_exit,
    ):
        result = main_ui.main()

    forced_exit.assert_called_once_with(37)
    assert result == 37
