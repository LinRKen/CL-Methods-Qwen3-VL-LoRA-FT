"""CPU-only safety and integration checks for the production W&B GPU helpers."""
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "qwen-vl-finetune/qwenvl/train/wandb_utils.py"
spec = importlib.util.spec_from_file_location("smoke_production_wandb_utils", SOURCE)
utils = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = utils
spec.loader.exec_module(utils)
UUIDS = {i: "GPU-" + str(i) * 8 + "-1111-2222-3333-" + str(i) * 12 for i in range(4)}

def response(ids=(2, 3)):
    return SimpleNamespace(stdout=chr(10).join(f"{i}, {UUIDS[i]}" for i in ids))

class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.environ = patch.dict(os.environ, {}, clear=True)
        self.environ.start()
        self.addCleanup(self.environ.stop)

    def test_numeric_ids_keep_order_and_query_selected_only(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = " 3, 2 "
        with patch.object(utils.subprocess, "run", return_value=response()) as run:
            selection = utils.prepare_training_gpu_selection()
        self.assertEqual(selection.physical_ids, (3, 2))
        self.assertEqual(selection.uuids, (UUIDS[3], UUIDS[2]))
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], ",".join(selection.uuids))
        self.assertEqual(run.call_args.args[0][1], "--id=3,2")
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertEqual(run.call_args.kwargs["timeout"], 15)

    def test_uuid_and_mixed_inputs(self):
        for visible, expected in [(UUIDS[2], (2,)), (UUIDS[3] + ",2", (3, 2)), ("1", (1,))]:
            with self.subTest(visible=visible):
                os.environ["CUDA_VISIBLE_DEVICES"] = visible
                with patch.object(utils.subprocess, "run", return_value=response(expected)):
                    actual = utils.prepare_training_gpu_selection()
                self.assertEqual(actual.physical_ids, expected)

    def test_invalid_input_rejected_before_query(self):
        for visible in [None, "", " ", "0", "2,0", "4", "-1", "all", "cuda:0", "2,2", "2,", ",2", "2,,3", "GPU-abcd", UUIDS[2] + "," + UUIDS[2]]:
            with self.subTest(visible=visible), patch.object(utils.subprocess, "run") as run:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                if visible is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = visible
                with self.assertRaises(RuntimeError):
                    utils.prepare_training_gpu_selection()
                run.assert_not_called()

    def test_forbidden_uuid_rejected_with_mock_only(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = UUIDS[0]
        with patch.object(utils.subprocess, "run", return_value=response((0,))):
            with self.assertRaises(RuntimeError):
                utils.prepare_training_gpu_selection()

    def test_aliases_for_same_device_rejected(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "2," + UUIDS[2]
        with patch.object(utils.subprocess, "run", return_value=response((2,))):
            with self.assertRaises(RuntimeError):
                utils.prepare_training_gpu_selection()

    def test_query_errors_fail_closed(self):
        for error in [FileNotFoundError("mock"), subprocess.CalledProcessError(1, ["mock"]), subprocess.TimeoutExpired(["mock"], 15)]:
            with self.subTest(error=type(error).__name__):
                os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
                with patch.object(utils.subprocess, "run", side_effect=error):
                    with self.assertRaises(RuntimeError):
                        utils.prepare_training_gpu_selection()
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,3")

    def test_malformed_identity_results_rejected(self):
        for output in ["", "2", "2,a,b", "bad," + UUIDS[2], "2,invalid", f"2,{UUIDS[2]}" + chr(10) + f"2,{UUIDS[3]}", f"1,{UUIDS[1]}", f"2,{UUIDS[2]}" + chr(10) + f"3,{UUIDS[2]}"]:
            with self.subTest(output=output):
                os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
                with patch.object(utils.subprocess, "run", return_value=SimpleNamespace(stdout=output)):
                    with self.assertRaises(RuntimeError):
                        utils.prepare_training_gpu_selection()

class MonitoringTests(unittest.TestCase):
    def setUp(self):
        self.selection = utils.TrainingGPUSelection((2, 3), (UUIDS[2], UUIDS[3]))
        self.environ = patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ",".join(self.selection.uuids)}, clear=True)
        self.environ.start()
        self.addCleanup(self.environ.stop)
        self.fake_wandb = patch.dict(sys.modules, {"wandb": SimpleNamespace(run=None)})
        self.fake_wandb.start()
        self.addCleanup(self.fake_wandb.stop)

    def configure(self, report_to=("wandb",), rank=0, selection=None):
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            utils.configure_wandb_gpu_monitoring(SimpleNamespace(report_to=report_to, process_index=rank), selection or self.selection)
        return stream.getvalue()

    def test_wandb_and_all_enable_filter_and_override_stale_values(self):
        for report in [["wandb"], "wandb", "tensorboard,wandb", ["all"], ["WANDB"]]:
            with self.subTest(report=report):
                os.environ["WANDB_X_STATS_GPU_DEVICE_IDS"] = "[0]"
                os.environ["WANDB__STATS_GPU_DEVICE_IDS"] = "[1]"
                self.assertIn("Monitoring training physical GPUs only", self.configure(report))
                for key in ["WANDB_X_STATS_GPU_DEVICE_IDS", "WANDB__STATS_GPU_DEVICE_IDS"]:
                    self.assertEqual(json.loads(os.environ[key]), [2, 3])

    def test_disabled_integrations_leave_settings_untouched(self):
        for report in [None, [], "none", ["tensorboard"]]:
            with self.subTest(report=report):
                before = dict(os.environ)
                self.assertEqual(self.configure(report), "")
                self.assertEqual(dict(os.environ), before)

    def test_other_rank_is_configured_without_printing(self):
        self.assertEqual(self.configure(rank=1), "")
        self.assertEqual(json.loads(os.environ["WANDB_X_STATS_GPU_DEVICE_IDS"]), [2, 3])

    def test_changed_cuda_selection_rejected(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        with self.assertRaises(RuntimeError):
            self.configure()

    def test_empty_or_forbidden_selection_rejected(self):
        for selection in [utils.TrainingGPUSelection((), ()), utils.TrainingGPUSelection((0,), (UUIDS[0],))]:
            with self.subTest(selection=selection), self.assertRaises(RuntimeError):
                self.configure(selection=selection)

    def test_initialized_run_rejected(self):
        sys.modules["wandb"].run = object()
        with self.assertRaises(RuntimeError):
            self.configure()

class IntegrationTests(unittest.TestCase):
    def test_training_entry_calls_helpers_in_required_order(self):
        tree = ast.parse((ROOT / "qwen-vl-finetune/qwenvl/train/train_qwen.py").read_text())
        prepare = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "prepare_training_gpu_selection"]
        torch_import = [n.lineno for n in tree.body if isinstance(n, ast.Import) and any(x.name == "torch" for x in n.names)]
        self.assertEqual(len(prepare), 1)
        self.assertLess(prepare[0], min(torch_import))
        train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
        calls = {name: [n.lineno for n in ast.walk(train) if isinstance(n, ast.Call) and (getattr(n.func, "id", None) == name or getattr(n.func, "attr", None) == name)] for name in ["parse_args_into_dataclasses", "configure_wandb_gpu_monitoring", "Trainer"]}
        self.assertLess(calls["parse_args_into_dataclasses"][0], calls["configure_wandb_gpu_monitoring"][0])
        self.assertLess(calls["configure_wandb_gpu_monitoring"][0], calls["Trainer"][0])

    def test_installed_wandb_reads_both_env_spellings(self):
        from wandb import Settings
        for env in [{"WANDB_X_STATS_GPU_DEVICE_IDS": "[2, 3]"}, {"WANDB__STATS_GPU_DEVICE_IDS": "[2, 3]"}]:
            with self.subTest(env=env):
                settings = Settings()
                settings.update_from_env_vars(env)
                self.assertEqual(list(settings.x_stats_gpu_device_ids), [2, 3])

if __name__ == "__main__":
    unittest.main(verbosity=2)
