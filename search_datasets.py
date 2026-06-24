import requests

r = requests.get(
    "https://huggingface.co/api/datasets?search=code&sort=downloads&direction=-1&limit=30"
)
data = r.json()
for ds in data:
    private = ds.get("private", False)
    gated = ds.get("gated", False) 
    if not private and not gated:
        print(f"  {ds['id']} - downloads: {ds.get('downloads', 0):,}")
