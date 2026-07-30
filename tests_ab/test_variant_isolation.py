"""验证 A/B 应用目录、持久化路径和包内导入完全隔离。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANT_PACKAGES = ("maixcam2_app_A_quad", "maixcam2_app_B_warp")


def test_variant_packages_use_independent_setting_paths():
    """验证两个变体使用不同设备目录，避免现场参数互相覆盖。"""
    from maixcam2_app_A_quad.config import PERSISTENT_SETTINGS_PATH as path_a
    from maixcam2_app_B_warp.config import PERSISTENT_SETTINGS_PATH as path_b

    assert path_a == "/root/maixcam2_puzzle_A/vision_settings.json"
    assert path_b == "/root/maixcam2_puzzle_B/vision_settings.json"
    assert path_a != path_b


def test_variant_sources_do_not_import_stable_package():
    """逐个扫描变体源码，确保运行时不回落到稳定版业务模块。"""
    for package_name in VARIANT_PACKAGES:
        package_path = PROJECT_ROOT / package_name
        assert package_path.is_dir(), f"缺少变体目录: {package_name}"
        for source_path in package_path.glob("*.py"):
            source_text = source_path.read_text(encoding="utf-8")
            assert "from maixcam2_app." not in source_text


def test_variant_app_ids_are_unique():
    """验证两个应用清单包含独立 ID，允许同时安装和分别打包。"""
    app_a = (PROJECT_ROOT / VARIANT_PACKAGES[0] / "app.yaml").read_text(
        encoding="utf-8"
    )
    app_b = (PROJECT_ROOT / VARIANT_PACKAGES[1] / "app.yaml").read_text(
        encoding="utf-8"
    )

    assert "id: diansai_quad" in app_a
    assert "id: diansai_warp" in app_b
    assert app_a != app_b
