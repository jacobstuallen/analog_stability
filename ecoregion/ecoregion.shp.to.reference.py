"""
Rasterize the TNC terrestrial ecoregions onto the analog reference grid.

Key difference from biome.shp.to.reference.py: we rasterize ECOREGION only,
then derive BIOME_ID from a lookup table. This guarantees that every grid cell's
biome is the biome of the ecoregion that owns that cell, so the two layers are
nested by construction. Rasterizing them independently would break nesting at
coastlines and ecoregion edges.
"""

import geopandas as gpd
import xarray as xr
import rasterio
from rasterio import features
import numpy as np
import json

SHAPEFILE = "shp/tnc_terr_ecoregions.shp"
REFERENCE_NC = "/glade/u/home/jsallen/projects/tnc_2025/tnc_climate_analogs/biomes/biomes.analog.gridded.nc"
OUTPUT_NC = "ecoregions.analog.gridded.nc"
FILL = -1

# Reference grid
ref = xr.open_dataset(REFERENCE_NC)
lat = ref["lat"].values
lon = ref["lon"].values
res_lat = abs(lat[1] - lat[0])
res_lon = abs(lon[1] - lon[0])
height, width = len(lat), len(lon)

# Rasterio needs a north-up grid. If the reference lat is ascending, flip it for
# the rasterize step and flip the result back afterwards.
lat_ascending = lat[0] < lat[-1]
lat_for_raster = lat[::-1] if lat_ascending else lat

minx = lon.min() - res_lon / 2.0
maxy = lat_for_raster.max() + res_lat / 2.0
transform = rasterio.transform.from_origin(minx, maxy, res_lon, res_lat)

# Shapefile
gdf = gpd.read_file(SHAPEFILE)
gdf = gdf.dropna(subset=["geometry", "ECO_ID_U", "WWF_MHTNUM"])
gdf = gdf.to_crs("EPSG:4326")
gdf["geometry"] = gdf["geometry"].buffer(0)

# Sequential ECO_ID (1..N). The native ECO_ID_U starts at 10000, which is fine
# but a compact 1..N index is easier to work with downstream.
gdf = gdf.sort_values("ECO_ID_U").reset_index(drop=True)
gdf["ECO_IDX"] = np.arange(1, len(gdf) + 1, dtype=np.int32)

# Lookup tables (ECO_IDX -> everything else)
eco_lookup = {
    int(r.ECO_IDX): {
        "eco_name": r.ECO_NAME,
        "eco_code": r.ECO_CODE,
        "eco_id_u": int(r.ECO_ID_U),
        "biome_id": int(r.WWF_MHTNUM),
        "biome_name": r.WWF_MHTNAM,
        "realm": r.WWF_REALM2,
    }
    for r in gdf.itertuples()
}

biome_lookup = (
    gdf[["WWF_MHTNUM", "WWF_MHTNAM"]]
    .drop_duplicates()
    .sort_values("WWF_MHTNUM")
    .set_index("WWF_MHTNUM")["WWF_MHTNAM"]
    .to_dict()
)
biome_lookup = {int(k): v for k, v in biome_lookup.items()}

print(f"{len(eco_lookup)} ecoregions across {len(biome_lookup)} biomes")

# Rasterize ecoregions
shapes = ((geom, val) for geom, val in zip(gdf.geometry, gdf["ECO_IDX"]))
eco_raster = features.rasterize(
    shapes=shapes,
    out_shape=(height, width),
    transform=transform,
    fill=FILL,
    dtype=np.int32,
    all_touched=False,
)

if lat_ascending:
    eco_raster = eco_raster[::-1, :]

# Derive the biome raster from the ecoregion raster via the lookup.
# Build a vectorized map array: index = ECO_IDX, value = biome id.
max_idx = int(gdf["ECO_IDX"].max())
eco_to_biome = np.full(max_idx + 1, FILL, dtype=np.int32)
for idx, meta in eco_lookup.items():
    eco_to_biome[idx] = meta["biome_id"]

biome_raster = np.full_like(eco_raster, FILL)
valid = eco_raster > 0
biome_raster[valid] = eco_to_biome[eco_raster[valid]]

# Sort longitude to match the reference convention
lon_order = np.argsort(lon)
lon = lon[lon_order]
eco_raster = eco_raster[:, lon_order]
biome_raster = biome_raster[:, lon_order]

coords = {"lat": lat, "lon": lon}
dims = ("lat", "lon")

ds = xr.Dataset(
    {
        "ECO_ID": xr.DataArray(
            eco_raster, coords=coords, dims=dims,
            attrs={"long_name": "TNC terrestrial ecoregion index (1..N)",
                   "missing_value": FILL},
        ),
        "BIOME_ID": xr.DataArray(
            biome_raster, coords=coords, dims=dims,
            attrs={"long_name": "WWF major habitat type (biome), derived from ECO_ID",
                   "missing_value": FILL},
        ),
    },
    coords=coords,
)

ds["lon"].attrs["units"] = "degrees_east"
ds["lat"].attrs["units"] = "degrees_north"
ds.attrs["eco_lookup"] = json.dumps(eco_lookup)
ds.attrs["biome_lookup"] = json.dumps(biome_lookup)
ds.attrs["source"] = "TNC Terrestrial Ecoregions (Olson et al. 2001 / TNC 2009)"
ds.attrs["note"] = "BIOME_ID derived from ECO_ID lookup: layers are exactly nested."

ds.to_netcdf(OUTPUT_NC)

# Diagnostics
n_land = int(valid.sum())
counts = np.bincount(eco_raster[valid], minlength=max_idx + 1)[1:]
print(f"Land cells: {n_land}")
print(f"Ecoregions with 0 cells: {(counts == 0).sum()}")
print(f"Ecoregions with <10 cells: {((counts > 0) & (counts < 10)).sum()}")
print(f"Median cells per ecoregion: {np.median(counts[counts > 0]):.0f}")
print(f"Saved -> {OUTPUT_NC}")
