import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = {
    "classification/train_grpo.sh": (
        "src/open_r1/grpo_classification_reward.py", 200, 128, True
    ),
    "classification/train_sft.sh": (
        "src/open_r1/sft_vision.py", 200, 128, True
    ),
    "detection/train_grpo.sh": (
        "src/open_r1/ship_detection.py", 200, 128, True
    ),
    "detection/train_sft.sh": (
        "src/open_r1/sft_vision_detection.py", 200, 128, True
    ),
    "ood/train_grpo.sh": (
        "src/open_r1/grpo_classification_ood_openset.py", 200, 128, True
    ),
    "ood/train_sft.sh": ("src/open_r1/sft_vision_ood.py", 200, 128, True),
}


def find_bash():
    # Prefer Git Bash over the Windows WSL launcher for local path handling.
    if os.name == "nt" and (git := shutil.which("git")):
        candidate = Path(git).resolve().parents[1] / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)
    return shutil.which("bash")


BASH = find_bash()


@unittest.skipUnless(BASH, "Bash is required for launcher checks")
class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ISeeShip launch tests ")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.source = self.work / "original source"
        for entry, _, _, _ in LAUNCHERS.values():
            path = self.source / entry
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Mock source; never executed.\n", encoding="utf-8")
        config = self.source / "local_scripts" / "zero3.json"
        config.parent.mkdir()
        config.write_text("{}\n", encoding="utf-8")
        self.data = self.work / "prepared data"
        self.data.mkdir()
        self.bin = self.work / "mock bin"
        self.bin.mkdir()
        mock = self.bin / "torchrun"
        mock.write_text("#!/usr/bin/env bash\nprintf '%s\\n' MOCK_TORCHRUN \"$@\"\n", encoding="utf-8")
        mock.chmod(0o755)
        self.env = os.environ.copy()
        for key in (
            "TRAINING_ROOT", "DATA_PATH", "CKPT_PATH", "SAVE_PATH",
            "CUDA_VISIBLE_DEVICES", "NPROC_PER_NODE", "DEEPSPEED_CONFIG",
            "MASTER_PORT", "DEBUG_MODE", "LOG_PATH", "BASH_ENV",
            "SHOT", "SEED",
        ):
            self.env.pop(key, None)
        self.env.update({
            "TRAINING_ROOT": self.source.as_posix(),
            "DATA_PATH": self.data.as_posix(),
            "CKPT_PATH": (self.work / "initial model").as_posix(),
            "SAVE_PATH": (self.work / "new output").as_posix(),
            "CUDA_VISIBLE_DEVICES": "0,2",
            "NPROC_PER_NODE": "2",
            "SHOT": "4",
            "SEED": "100",
            "PATH": str(self.bin) + os.pathsep + os.environ.get("PATH", ""),
        })

    def run_launcher(self, launcher, env=None):
        return subprocess.run(
            [BASH, (ROOT / "src/scripts" / launcher).as_posix()],
            cwd=self.work, env=self.env if env is None else env,
            text=True, encoding="utf-8", capture_output=True, check=False,
        )

    def assert_preflight_failure(self, result, message):
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(message, result.stderr)
        self.assertNotIn("MOCK_TORCHRUN", result.stdout)

    def test_bash_syntax(self):
        for launcher in ["common.sh", *LAUNCHERS]:
            with self.subTest(launcher=launcher):
                result = subprocess.run(
                    [BASH, "-n", (ROOT / "src/scripts" / launcher).as_posix()],
                    text=True, capture_output=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_paper_defaults_and_paths_with_spaces(self):
        for launcher, (entry, steps, rank, deepspeed) in LAUNCHERS.items():
            with self.subTest(launcher=launcher):
                result = self.run_launcher(launcher)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = result.stdout.splitlines()
                self.assertEqual(args[0], "MOCK_TORCHRUN")
                self.assertIn("--nproc_per_node=2", args)
                self.assertIn((self.source / entry).as_posix(), args)
                self.assertEqual(args[args.index("--max_steps") + 1], str(steps))
                self.assertEqual(args[args.index("--lora_r") + 1], str(rank))
                self.assertEqual(args[args.index("--lora_alpha") + 1], "256")
                self.assertEqual(args[args.index("--seed") + 1], "100")
                self.assertEqual(args[args.index("--optim") + 1], "adamw_torch")
                self.assertEqual(args[args.index("--per_device_train_batch_size") + 1], "1")
                self.assertEqual(args[args.index("--gradient_accumulation_steps") + 1], "2")
                if launcher.endswith("train_grpo.sh"):
                    self.assertEqual(args[args.index("--num_generations") + 1], "6")
                else:
                    self.assertNotIn("--num_generations", args)
                self.assertEqual(args[args.index("--model_name_or_path") + 1], self.env["CKPT_PATH"])
                self.assertEqual(args[args.index("--output_dir") + 1], self.env["SAVE_PATH"])
                data_option = "--dataset_name" if "--dataset_name" in args else "--dataset_path"
                self.assertEqual(args[args.index(data_option) + 1], self.env["DATA_PATH"])
                self.assertEqual("--deepspeed" in args, deepspeed)
                if launcher.startswith("classification/"):
                    self.assertEqual(float(args[args.index("--learning_rate") + 1]), 2e-5)
                elif launcher.startswith("detection/"):
                    self.assertEqual(float(args[args.index("--learning_rate") + 1]), 1e-6)
                if launcher == "detection/train_grpo.sh":
                    start = args.index("--reward_funcs") + 1
                    self.assertEqual(args[start:], ["accuracy_iou_progressive", "accuracy_confidence", "format"])

    def test_required_environment_variables(self):
        for key in ("DATA_PATH", "CKPT_PATH", "SAVE_PATH", "CUDA_VISIBLE_DEVICES"):
            with self.subTest(key=key):
                env = self.env.copy()
                env.pop(key)
                self.assert_preflight_failure(self.run_launcher(next(iter(LAUNCHERS)), env), f"Set {key}")

    def test_missing_original_source(self):
        env = self.env.copy()
        env["TRAINING_ROOT"] = (self.work / "missing source").as_posix()
        self.assert_preflight_failure(self.run_launcher(next(iter(LAUNCHERS)), env), "Training entry not found")

    def test_missing_data_and_deepspeed(self):
        env = self.env.copy()
        env["DATA_PATH"] = (self.work / "missing data").as_posix()
        self.assert_preflight_failure(self.run_launcher(next(iter(LAUNCHERS)), env), "DATA_PATH must be an existing")
        env = self.env.copy()
        env["DEEPSPEED_CONFIG"] = (self.work / "missing config.json").as_posix()
        self.assert_preflight_failure(self.run_launcher(next(iter(LAUNCHERS)), env), "Missing DeepSpeed configuration")

    def test_device_count_and_process_count(self):
        for value in ("0", "0,", ",0", "0,,2"):
            with self.subTest(devices=value):
                env = self.env.copy()
                env["CUDA_VISIBLE_DEVICES"] = value
                self.assert_preflight_failure(self.run_launcher(next(iter(LAUNCHERS)), env), "Select exactly")
        env = self.env.copy()
        env["NPROC_PER_NODE"] = "0"
        self.assert_preflight_failure(self.run_launcher(next(iter(LAUNCHERS)), env), "positive integer")

    def test_support_size_and_seed_selection(self):
        for seed in ("42", "43", "44"):
            with self.subTest(seed=seed):
                env = self.env.copy()
                env.update({"SHOT": "8", "SEED": seed})
                result = self.run_launcher(next(iter(LAUNCHERS)), env)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = result.stdout.splitlines()
                self.assertEqual(args[args.index("--seed") + 1], seed)
                self.assertIn(f"8shot_seed{seed}", args[args.index("--run_name") + 1])
        env = self.env.copy()
        env.pop("SHOT")
        env.pop("SEED")
        result = self.run_launcher(next(iter(LAUNCHERS)), env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("4shot_seed100", result.stdout)

    def test_invalid_support_size_and_seed(self):
        env = self.env.copy()
        env["SHOT"] = "5"
        self.assert_preflight_failure(self.run_launcher(next(iter(LAUNCHERS)), env), "Set SHOT to 4 or 8")
        for seed in ("-1", "abc"):
            with self.subTest(seed=seed):
                env = self.env.copy()
                env["SEED"] = seed
                self.assert_preflight_failure(self.run_launcher(next(iter(LAUNCHERS)), env), "nonnegative integer")


if __name__ == "__main__":
    unittest.main()
