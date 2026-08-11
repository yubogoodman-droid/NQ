"""
Deprecated entrypoint.

日常請用 shadow_neckline_volume.py：
  選幣 = 24h 漲幅榜前 10
  偵測 = shadow_neckline_logic.VOLUME（爆量≥2.5× + 結構過濾）
"""

from shadow_neckline_volume import main

if __name__ == "__main__":
    main()
