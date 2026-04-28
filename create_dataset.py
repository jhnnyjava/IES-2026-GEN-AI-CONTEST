"""Generate a synthetic Kenya maize production dataset for testing the AgriResilAI+ workflow."""

import csv
from pathlib import Path

# Define the output path
output_path = Path(__file__).parent / "data" / "ken_maize_production.csv"

# Define regions and administrative levels (realistic Kenya geography)
adlevels1 = ["Central", "Coastal", "Eastern", "Nyanza", "Rift Valley", "Western"]
adlevels2 = {
    "Central": ["Kiambu", "Muranga", "Nyeri"],
    "Coastal": ["Mombasa", "Kwale", "Lamu"],
    "Eastern": ["Makueni", "Machakos", "Embu"],
    "Nyanza": ["Kisumu", "Migori", "Homa Bay"],
    "Rift Valley": ["Nakuru", "Uasin Gishu", "Trans-Nzoia"],
    "Western": ["Kisii", "Nyamira"],
}
adlevels3 = {
    "Kiambu": ["Thika", "Ruiru", "Gatundu"],
    "Muranga": ["Murang'a", "Kangema"],
    "Nyeri": ["Nyeri", "Othaya"],
    "Mombasa": ["Mombasa", "Port Reitz"],
    "Kwale": ["Kwale", "Ukunda"],
    "Lamu": ["Lamu", "Kiunga"],
    "Makueni": ["Makueni", "Mwala"],
    "Machakos": ["Machakos", "Kangundo"],
    "Embu": ["Embu", "Runyenjes"],
    "Kisumu": ["Kisumu", "Siaya"],
    "Migori": ["Migori", "Kericho"],
    "Homa Bay": ["Homa Bay", "Suba"],
    "Nakuru": ["Nakuru", "Njoro"],
    "Uasin Gishu": ["Eldoret", "Turbo"],
    "Trans-Nzoia": ["Kitale", "Cherangany"],
    "Kisii": ["Kisii", "Kericho"],
    "Nyamira": ["Nyamira", "Nyaribari"],
}

years = ["86-90", "91-95", "96-00", "01-05", "06-10", "11-15"]

# Generate synthetic rows
rows = []
row_id = 1

for level1 in adlevels1:
    for level2 in adlevels2.get(level1, []):
        for level3 in adlevels3.get(level2, [level2]):
            for year in years:
                # Synthetic but realistic data
                area_harvested = 500 + (hash(f"{level1}{level2}{level3}{year}") % 5000)
                total_production = area_harvested * (8 + (hash(f"prod{level1}{level2}{level3}{year}") % 12))
                maize_yield = total_production / area_harvested if area_harvested > 0 else 0

                rows.append({
                    "_id": row_id,
                    "FID": row_id,
                    "the_geom": f"POLYGON(({row_id},{row_id},{row_id+1},{row_id},{row_id+1},{row_id+1},{row_id},{row_id+1},{row_id},{row_id}))",
                    "AREA": 50 + (hash(f"area{row_id}") % 500),
                    "PERIMETER": 30 + (hash(f"perim{row_id}") % 200),
                    "REGIONS_": f"Region_{row_id % 10}",
                    "REGIONS_ID": row_id % 10,
                    "SQKM": 50 + (hash(f"sqkm{row_id}") % 500),
                    "ADMSQKM": 40 + (hash(f"admsqkm{row_id}") % 400),
                    "CODE": f"KEN{row_id:05d}",
                    "ADMINID": row_id,
                    "COUNTRY": "Kenya",
                    "ADLEVEL1": level1,
                    "ADLEVEL2": level2,
                    "ADLEVEL3": level3,
                    "TOTMAZPROD": int(total_production),
                    "MAZYIELD": round(maize_yield, 2),
                    "AREAHARV": area_harvested,
                    "YEAR": year,
                })
                row_id += 1

# Write to CSV
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "_id", "FID", "the_geom", "AREA", "PERIMETER", "REGIONS_", "REGIONS_ID",
            "SQKM", "ADMSQKM", "CODE", "ADMINID", "COUNTRY", "ADLEVEL1", "ADLEVEL2",
            "ADLEVEL3", "TOTMAZPROD", "MAZYIELD", "AREAHARV", "YEAR"
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"✓ Synthetic dataset created: {output_path}")
print(f"  Rows: {len(rows)}")
print(f"  Regions: {len(adlevels1)}, Years: {len(years)}")
