"""
Deprecated entrypoint.

日常請用 shadow_neckline_volume.py：
  選幣 = 24h 漲幅榜前 10
  偵測 = STRUCTURE（結構過濾，不看爆量）→ Telegram
"""

from shadow_neckline_volume import main

if __name__ == "__main__":
    main()
