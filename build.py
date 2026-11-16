"""Pack the CSVs into the dashboard HTML as compact JSON."""
import pandas as pd, json, os
H = os.path.dirname(os.path.abspath(__file__)); DD = H + "/data"
jobs = pd.read_csv(f"{DD}/service_jobs.csv", keep_default_na=False)
parts = pd.read_csv(f"{DD}/parts_catalogue.csv"); faults = pd.read_csv(f"{DD}/fault_codes.csv")
techs = pd.read_csv(f"{DD}/technicians.csv")

months = sorted(jobs.month.unique())
regions = list(techs.region.unique())
cities = sorted(set(techs.city))
brands = ["Arçelik", "Beko", "Grundig", "Altus"]
cats = list(faults.category.unique())
reasons = ["", "Missing part", "Wrong diagnosis", "Repair not successful", "Customer not available"]
pidx = {c: i for i, c in enumerate(parts.part_code)}
fidx = {c: i for i, c in enumerate(faults.fault_code)}
tidx = {c: i for i, c in enumerate(techs.technician_id)}
split = lambda s: [pidx[p] for p in s.split("; ")] if s else []

rows = []
for j in jobs.itertuples():
    rows.append([int(j.job_id[1:]), months.index(j.month), int(j.first_visit_date[-2:]), regions.index(j.region),
                 cities.index(j.city), tidx[j.technician_id], brands.index(j.brand), cats.index(j.category),
                 fidx[j.fault_code], int(j.product_age_years), reasons.index(j.second_visit_reason),
                 int(j.days_to_resolve), split(j.parts_needed), split(j.missing_parts)])
D = dict(months=months, regions=regions, cities=cities, brands=brands, cats=cats, reasons=reasons,
         techs=[[t.technician_id, regions.index(t.region), cities.index(t.city), int(t.tenure_years)] for t in techs.itertuples()],
         faults=[[f.fault_code, f.category, f.fault_description] for f in faults.itertuples()],
         parts=[[p.part_code, p.part_name, float(p.assumed_van_stock_rate)] for p in parts.itertuples()],
         jobs=rows)
js = json.dumps(D, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
html = open(f"{H}/template.html", encoding="utf-8").read().replace("/*__DATA__*/null", js)
os.makedirs(f"{H}/out", exist_ok=True)
open(f"{H}/out/uc19_parts_prediction_dashboard.html", "w", encoding="utf-8").write(html)
print("html size KB:", len(html.encode()) // 1024)
