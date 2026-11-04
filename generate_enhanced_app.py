#!/usr/bin/env python3
"""
Script to generate the enhanced app.py with full place search, 115 Indian cities autocomplete,
OpenStreetMap Nominatim geocoding, and nearest CAMS grid climatology lookup.
"""

import json

# Load cities
with open("results/india_cities.json") as f:
    cities_json_str = f.read()

# Load climatology
with open("results/cell_climatology.json") as f:
    climatology_json_str = f.read()

print("Loaded datasets for embedding.")

