from sentinelsat import SentinelAPI, geojson_to_wkt

# 🔐 Copernicus login (replace these)
USER = "YOUR_USERNAME"
PASS = "YOUR_PASSWORD"

api = SentinelAPI(USER, PASS, "https://apihub.copernicus.eu/apihub")

# 🌍 Niger Delta area
footprint = geojson_to_wkt({
    "type": "Polygon",
    "coordinates": [[
        [5.0, 4.5],
        [7.5, 4.5],
        [7.5, 6.5],
        [5.0, 6.5],
        [5.0, 4.5]
    ]]
})

print("Searching Sentinel-2 data...")

products = api.query(
    footprint,
    date=("20240101", "20240131"),
    platformname="Sentinel-2",
    processinglevel="Level-2A",
    cloudcoverpercentage=(0, 20)
)

print(f"Found {len(products)} images")

api.download_all(products)

print("Download complete")