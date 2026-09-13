"""
讓matplotlib畫圖時中文不會變成方框(tofu)的小工具，windows/mac/linux都能用。

之前踩過的坑：第一版把Linux(這個開發環境)才有的字型路徑寫死進程式碼，
結果同學在Windows上執行直接FileNotFoundError。改成：先看系統上有沒有
matplotlib「已經認得」的中文字型名稱，如果都沒有，才在Linux常見路徑裡
找字型檔案手動註冊；Windows/Mac不需要手動註冊，系統字型matplotlib通常
自己就掃得到。
"""
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 依平台常見的中文字型名稱，越前面優先度越高
CANDIDATE_FONTS = [
    "Microsoft JhengHei",   # Windows 內建繁體中文字型
    "PingFang TC",          # macOS 內建繁體中文字型
    "Noto Sans CJK TC",
    "Noto Sans CJK JP",     # 有些系統把CJK字型都註冊在JP這個family名稱下
    "SimHei",
    "Arial Unicode MS",
]

# Linux上常見、但matplotlib不一定會自動掃到的字型檔路徑，找到就手動註冊
LINUX_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
]


def setup_cjk_font():
    available = {f.name for f in fm.fontManager.ttflist}

    if not any(name in available for name in CANDIDATE_FONTS):
        for path in LINUX_FONT_PATHS:
            if os.path.exists(path):
                try:
                    fm.fontManager.addfont(path)
                except Exception:
                    pass
                break

    plt.rcParams["font.sans-serif"] = CANDIDATE_FONTS + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
