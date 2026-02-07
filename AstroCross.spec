# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import copy_metadata

datas = []
binaries = []
hiddenimports = ["xisf", "zstandard", "lz4", "lz4.block"]
datas += collect_data_files("xisf")
datas += copy_metadata("xisf")
hiddenimports += collect_submodules("xisf")
hiddenimports += collect_submodules("zstandard")
hiddenimports += collect_submodules("lz4")
binaries += collect_dynamic_libs("zstandard")
binaries += collect_dynamic_libs("lz4")


a = Analysis(
    ['AstroCross.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AstroCross',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Post-build: create a zip containing the built executable for easy distribution.
import os
import shutil
import sys

# __file__ is not defined in PyInstaller spec execution context.
spec_dir = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv else os.getcwd()
dist_dir = os.path.join(spec_dir, "dist")
exe_path = os.path.join(dist_dir, "AstroCross.exe")
zip_path = os.path.join(dist_dir, "AstroCross-win64.zip")

if os.path.exists(exe_path):
    if os.path.exists(zip_path):
        os.remove(zip_path)
    shutil.make_archive(zip_path[:-4], "zip", dist_dir, "AstroCross.exe")
