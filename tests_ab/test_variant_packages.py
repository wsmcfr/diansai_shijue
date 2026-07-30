"""验证A/B应用清单、独立ZIP和平铺部署导入。"""

from pathlib import Path
import runpy
import subprocess
import sys
import zipfile

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANT_RELEASES = (
    (
        "maixcam2_app_A_quad",
        "diansai_quad",
        "2.0.0",
        "diansai_quad-v2.0.0.zip",
        {
            "__init__.py",
            "A版实机调试手册.md",
            "MaixCAM2与STM32F4串口协议说明.md",
            "app.yaml",
            "assembly_planner.py",
            "calibration_ui.py",
            "config.py",
            "main.py",
            "paper_locator.py",
            "puzzle_vision.py",
            "serial_protocol.py",
            "settings_store.py",
            "template_store.py",
            "touch_ui.py",
        },
    ),
    (
        "maixcam2_app_B_warp",
        "diansai_warp",
        "1.1.0",
        "diansai_warp-v1.1.0.zip",
        {
            "__init__.py",
            "app.yaml",
            "calibration_ui.py",
            "config.py",
            "main.py",
            "paper_locator.py",
            "paper_warp.py",
            "puzzle_vision.py",
            "settings_store.py",
            "template_store.py",
            "touch_ui.py",
        },
    ),
)


@pytest.mark.parametrize(
    ("package_name", "app_id", "app_version", "archive_name", "runtime_files"),
    VARIANT_RELEASES,
)
def test_manifest_has_independent_id_version_and_all_runtime_files(
    package_name,
    app_id,
    app_version,
    archive_name,
    runtime_files,
):
    """验证应用清单的独立ID、指定版本和显式运行模块集合。"""
    del archive_name
    manifest_text = (PROJECT_ROOT / package_name / "app.yaml").read_text(encoding="utf-8")

    assert f"id: {app_id}\n" in manifest_text
    assert f"name: {app_id}\n" in manifest_text
    assert f"version: {app_version}\n" in manifest_text
    for filename in runtime_files:
        assert f"  - {filename}\n" in manifest_text


@pytest.mark.parametrize(
    ("package_name", "app_id", "app_version", "archive_name", "runtime_files"),
    VARIANT_RELEASES,
)
def test_release_zip_is_flat_exact_and_excludes_generated_files(
    package_name,
    app_id,
    app_version,
    archive_name,
    runtime_files,
):
    """验证ZIP只含白名单文件，且文件直接位于根目录供MaixVision平铺运行。"""
    del app_id, app_version
    archive_path = PROJECT_ROOT / package_name / "dist" / archive_name
    assert archive_path.is_file()

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())

    assert names == runtime_files
    assert all("/" not in name and "\\" not in name for name in names)
    assert not any("__pycache__" in name or name.endswith(".json") for name in names)
    if package_name.endswith("A_quad"):
        assert "paper_warp.py" not in names
    else:
        assert "paper_warp.py" in names


@pytest.mark.parametrize(
    ("package_name", "app_id", "app_version", "archive_name", "runtime_files"),
    VARIANT_RELEASES,
)
def test_release_zip_file_bytes_match_current_source(
    package_name,
    app_id,
    app_version,
    archive_name,
    runtime_files,
):
    """发布ZIP每个条目必须与当前源码逐字节一致，防止测试通过却上传旧main.py。"""
    del app_id, app_version
    package_path = PROJECT_ROOT / package_name
    archive_path = package_path / "dist" / archive_name

    with zipfile.ZipFile(archive_path) as archive:
        for filename in runtime_files:
            assert archive.read(filename) == (package_path / filename).read_bytes()


@pytest.mark.parametrize(
    ("package_name", "app_id", "app_version", "archive_name", "runtime_files"),
    VARIANT_RELEASES,
)
def test_release_zip_flat_main_imports_without_package_directory(
    package_name,
    app_id,
    app_version,
    archive_name,
    runtime_files,
    tmp_path,
):
    """验证ZIP解压后从独立临时目录导入main不会依赖顶层包。"""
    del app_id, app_version, runtime_files
    extract_path = tmp_path / package_name
    extract_path.mkdir()
    archive_path = PROJECT_ROOT / package_name / "dist" / archive_name
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_path)

    command = (
        "import runpy, sys; "
        f"sys.path.insert(0, {str(extract_path)!r}); "
        f"runpy.run_path({str(extract_path / 'main.py')!r}, run_name='flat_release_import')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=extract_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_package_script_exposes_two_explicit_release_specs():
    """验证打包脚本公开两个白名单规格，便于测试与文档核对。"""
    module_globals = runpy.run_path(str(PROJECT_ROOT / "tools" / "package_variants.py"))

    assert set(module_globals["RELEASE_SPECS"]) == {
        "maixcam2_app_A_quad",
        "maixcam2_app_B_warp",
    }
