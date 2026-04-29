import shutil
from pathlib import Path

src_rain = Path(r'c:\Users\johnk\Documents\GENAi 2926 Docs\kenya-climate-data-1991-2016-rainfallmm.csv')
src_temp = Path(r'c:\Users\johnk\Documents\GENAi 2926 Docs\kenya-climate-data-1991-2016-temp-degress-celcius.csv')
dst_dir = Path('data')
dst_dir.mkdir(exist_ok=True)

if src_rain.exists():
    shutil.copy2(src_rain, dst_dir / 'rainfall.csv')
    print(f'Copied rainfall CSV')
else:
    print(f'Not found: {src_rain}')

if src_temp.exists():
    shutil.copy2(src_temp, dst_dir / 'temperature.csv')
    print(f'Copied temperature CSV')
else:
    print(f'Not found: {src_temp}')

print("Climate data files ready")
