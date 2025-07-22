# -*- mode: python ; coding: utf-8 -*-

import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 5)

block_cipher = None

a = Analysis(
    ['pyside_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('models', 'models'), ('config_transmorph.py', '.'), ('funcs_transmorph.py', '.'), ('datapaths.yaml', '.')
            , ('app_build_env/lib/python3.12/site-packages/napari','napari'),
            ('app_build_env/lib/python3.12/site-packages/vispy','vispy')],
    hiddenimports=['GUI_scripts', 'utils', 'pydicom', 'skimage', 'scipy', 'napari', 'napari.view_layers', 'torch', 'PySide6', 'ultralytics', 'h5py', 'natsort', 'torchvision', 'tqdm', 'timm', 'ml_collections'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['ipykernel','jupyter_client','IPython'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OCT_mac_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OCT_mac_app',
)
app = BUNDLE(
    coll,
    name='OCT_mac_app.app',
    icon=None,
    bundle_identifier=None,
)
