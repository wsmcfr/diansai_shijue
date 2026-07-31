"""按显式白名单生成两个可直接导入MaixVision的平铺应用ZIP。"""

from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 每个发布规格同时固定目录、ZIP名和运行模块，防止缓存或设备JSON被意外打包。
RELEASE_SPECS = {
    "maixcam2_app_A_quad": {
        "archive": "diansai_quad-v2.2.0.zip",
        "files": (
            "__init__.py",
            "A版实机调试手册.md",
            "app.yaml",
            "assembly_planner.py",
            "calibration_ui.py",
            "config.py",
            "four_piece_solver.py",
            "four_piece_vision.py",
            "main.py",
            "MaixCAM2与STM32F4串口协议说明.md",
            "paper_locator.py",
            "puzzle_vision.py",
            "serial_protocol.py",
            "settings_store.py",
            "template_store.py",
            "touch_ui.py",
        ),
    },
    "maixcam2_app_B_warp": {
        "archive": "diansai_warp-v1.1.0.zip",
        "files": (
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
        ),
    },
}


def _read_manifest_files(manifest_path):
    """读取简单Maix app.yaml的files列表并返回文件名元组。

    当前清单的files段固定置于文件开头；本函数只接受两个空格加短横线的条目，遇到
    首个顶层字段即结束，避免把id/name误当作文件。
    """
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    try:
        start_index = lines.index("files:") + 1
    except ValueError as error:
        raise ValueError(f"应用清单缺少files字段: {manifest_path}") from error
    files = []
    for line in lines[start_index:]:
        if not line.startswith("  - "):
            break
        files.append(line[4:].strip())
    if not files:
        raise ValueError(f"应用清单files字段为空: {manifest_path}")
    return tuple(files)


def _validate_release_spec(package_name, spec):
    """核对源码、清单和白名单完全一致，返回包目录与有序文件列表。"""
    package_path = PROJECT_ROOT / package_name
    if not package_path.is_dir():
        raise FileNotFoundError(f"变体目录不存在: {package_path}")
    runtime_files = tuple(spec["files"])
    manifest_files = _read_manifest_files(package_path / "app.yaml")
    if manifest_files != runtime_files:
        raise ValueError(
            f"{package_name} app.yaml文件顺序或集合与发布白名单不一致"
        )
    missing = [filename for filename in runtime_files if not (package_path / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"{package_name}缺少运行模块: {','.join(missing)}")
    return package_path, runtime_files


def package_variant(package_name, spec):
    """生成单个平铺ZIP并返回输出路径。

    ZIP条目使用固定时间戳，保证源码不变时发布物可重复；条目名称只取文件名，绝不
    带顶层包目录。输出目录不存在时自动创建。
    """
    package_path, runtime_files = _validate_release_spec(package_name, spec)
    output_directory = package_path / "dist"
    output_directory.mkdir(parents=True, exist_ok=True)
    archive_path = output_directory / str(spec["archive"])

    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for filename in runtime_files:
            source_path = package_path / filename
            archive_info = zipfile.ZipInfo(filename, date_time=(2026, 7, 29, 0, 0, 0))
            archive_info.compress_type = zipfile.ZIP_DEFLATED
            archive_info.external_attr = 0o644 << 16
            archive.writestr(archive_info, source_path.read_bytes())
    return archive_path


def package_all():
    """按固定顺序生成A/B两个发布包并返回路径列表。"""
    return [
        package_variant(package_name, RELEASE_SPECS[package_name])
        for package_name in ("maixcam2_app_A_quad", "maixcam2_app_B_warp")
    ]


def main():
    """命令行生成两个发布包，逐行打印路径并返回0。"""
    for archive_path in package_all():
        print(archive_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
