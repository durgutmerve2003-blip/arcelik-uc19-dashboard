"""Dummy service-jobs dataset for Arçelik AI use case 19:
Parts prediction before the technician leaves.
All data is synthetic. Seeded, so re-running gives the same files."""
import numpy as np, pandas as pd, json, os

rng = np.random.default_rng(19)
OUT = os.path.dirname(os.path.abspath(__file__)) + "/data"
os.makedirs(OUT, exist_ok=True)

# ---------------- reference tables ----------------
# part_code: (name, van_stock_rate = share of vans that carry it)
PARTS = {
 "P-WM-DP":  ("Drain pump (WM)", .92), "P-WM-FLT": ("Pump filter (WM)", .95), "P-WM-DH": ("Drain hose (WM)", .88),
 "P-WM-CB":  ("Carbon brushes", .93), "P-WM-MOT": ("Drive motor (WM)", .40), "P-WM-BLT": ("Drive belt (WM)", .90),
 "P-WM-MCB": ("Motor control board (WM)", .45), "P-WM-DG": ("Door gasket (WM)", .80), "P-WM-IH": ("Inlet hose", .94),
 "P-WM-TSK": ("Tub seal kit", .50), "P-WM-DD": ("Detergent drawer", .70), "P-WM-HE": ("Heating element (WM)", .88),
 "P-NTC":    ("NTC temperature sensor", .95), "P-WM-DL": ("Door lock (WM)", .93), "P-WM-DHN": ("Door handle (WM)", .75),
 "P-WM-MB":  ("Main control board (WM)", .38), "P-WM-DB": ("Display board (WM)", .45), "P-PWR": ("Power cable / EMI filter", .85),
 "P-WM-BRG": ("Drum bearing kit", .35), "P-WM-SA": ("Shock absorber pair", .82), "P-WM-CW": ("Counterweight bolt set", .90),
 "P-DW-DP":  ("Drain pump (DW)", .90), "P-DW-FS": ("Filter set (DW)", .93), "P-DW-WP": ("Wash pump (DW)", .62),
 "P-DW-SA":  ("Spray arm", .85), "P-DW-DV": ("Diverter valve", .55), "P-DW-AQ": ("Aquastop hose", .78),
 "P-DW-DS":  ("Door seal (DW)", .80), "P-FLS": ("Float switch", .88), "P-DW-HP": ("Heater pump (DW)", .58),
 "P-DW-MB":  ("Main control board (DW)", .40), "P-DW-DB": ("Display board (DW)", .48),
 "P-RF-CMP": ("Compressor (RF)", .25), "P-RF-SR": ("Start relay (RF)", .92), "P-GAS": ("Refrigerant recharge kit", .85),
 "P-RF-TH":  ("Thermostat (RF)", .88), "P-RF-DH": ("Defrost heater", .72), "P-RF-DS": ("Defrost sensor", .90),
 "P-RF-EF":  ("Evaporator fan", .70), "P-RF-DT": ("Drain tube heater", .80), "P-RF-WF": ("Water filter (RF)", .75),
 "P-RF-CF":  ("Condenser fan", .68), "P-RF-DG": ("Door gasket (RF)", .62), "P-RF-LED": ("LED lamp (RF)", .95),
 "P-RF-IM":  ("Ice maker unit", .30), "P-RF-WV": ("Water inlet valve (RF)", .78),
 "P-OV-BE":  ("Bottom element (OV)", .80), "P-OV-TE": ("Top / grill element (OV)", .78), "P-OV-FE": ("Fan element (OV)", .75),
 "P-OV-FM":  ("Fan motor (OV)", .70), "P-OV-TH": ("Oven thermostat", .85), "P-OV-TS": ("Oven temperature sensor", .88),
 "P-OV-HG":  ("Door hinge (OV)", .82), "P-OV-GL": ("Inner door glass (OV)", .22),
 "P-DR-CU":  ("Condenser unit (DR)", .40), "P-DR-HE": ("Heating element (DR)", .80), "P-DR-FM": ("Fan motor (DR)", .62),
 "P-DR-BLT": ("Drive belt (DR)", .92), "P-DR-TN": ("Belt tensioner (DR)", .82), "P-DR-DP": ("Drain pump (DR)", .78),
 "P-AC-CAP": ("Capacitor (AC)", .95), "P-AC-OPCB": ("Outdoor PCB (AC)", .35), "P-AC-IPCB": ("Indoor PCB (AC)", .42),
 "P-AC-CH":  ("Condensate hose", .95), "P-AC-DPAN": ("Drain pan (AC)", .50), "P-AC-CBL": ("Interconnect cable (AC)", .88),
 "P-AC-RC":  ("Remote control (AC)", .82), "P-AC-RB": ("Receiver board (AC)", .55),
}

# fault_code: (category, description, {part: prob of being needed}, extra non-part failure risk)
FAULTS = {
 "WM-E01": ("Washing machine", "Not draining",            {"P-WM-DP": .70, "P-WM-FLT": .25, "P-WM-DH": .10}, 0.00),
 "WM-E02": ("Washing machine", "Drum not spinning",       {"P-WM-CB": .45, "P-WM-MOT": .20, "P-WM-BLT": .30, "P-WM-MCB": .15}, .03),
 "WM-E03": ("Washing machine", "Leaking water",           {"P-WM-DG": .50, "P-WM-IH": .20, "P-WM-TSK": .20, "P-WM-DD": .10}, .03),
 "WM-E04": ("Washing machine", "Not heating",             {"P-WM-HE": .75, "P-NTC": .35}, 0.00),
 "WM-E05": ("Washing machine", "Door will not lock/open", {"P-WM-DL": .80, "P-WM-DHN": .20}, 0.00),
 "WM-E06": ("Washing machine", "No power / electronic fault", {"P-WM-MB": .55, "P-WM-DB": .25, "P-PWR": .15}, .06),
 "WM-E07": ("Washing machine", "Noise / vibration",       {"P-WM-BRG": .55, "P-WM-SA": .40, "P-WM-CW": .10}, .02),
 "DW-E01": ("Dishwasher", "Not draining",                 {"P-DW-DP": .65, "P-DW-FS": .35}, 0.00),
 "DW-E02": ("Dishwasher", "Poor wash result",             {"P-DW-WP": .40, "P-DW-SA": .45, "P-DW-DV": .15}, .04),
 "DW-E03": ("Dishwasher", "Water leak / aquastop",        {"P-DW-AQ": .55, "P-DW-DS": .30, "P-FLS": .25}, .02),
 "DW-E04": ("Dishwasher", "Not heating",                  {"P-DW-HP": .70, "P-NTC": .30}, .01),
 "DW-E05": ("Dishwasher", "Control fault",                {"P-DW-MB": .65, "P-DW-DB": .25}, .05),
 "RF-E01": ("Refrigerator", "Not cooling",                {"P-RF-CMP": .30, "P-RF-SR": .40, "P-GAS": .35, "P-RF-TH": .15}, .06),
 "RF-E02": ("Refrigerator", "Frost build-up (No Frost)",  {"P-RF-DH": .55, "P-RF-DS": .40, "P-RF-EF": .20}, .03),
 "RF-E03": ("Refrigerator", "Water inside cabinet",       {"P-RF-DT": .40, "P-RF-WF": .15}, .01),
 "RF-E04": ("Refrigerator", "Noisy fan",                  {"P-RF-EF": .60, "P-RF-CF": .40}, .01),
 "RF-E05": ("Refrigerator", "Door seal / light",          {"P-RF-DG": .65, "P-RF-LED": .35}, 0.00),
 "RF-E06": ("Refrigerator", "Ice maker not working",      {"P-RF-IM": .55, "P-RF-WV": .45}, .03),
 "OV-E01": ("Oven", "Not heating",                        {"P-OV-BE": .40, "P-OV-TE": .35, "P-OV-FE": .30}, .01),
 "OV-E02": ("Oven", "Fan not working",                    {"P-OV-FM": .75}, 0.00),
 "OV-E03": ("Oven", "Temperature wrong",                  {"P-OV-TH": .55, "P-OV-TS": .45}, .03),
 "OV-E04": ("Oven", "Door glass / hinge",                 {"P-OV-HG": .60, "P-OV-GL": .35}, 0.00),
 "DR-E01": ("Dryer", "Not drying",                        {"P-DR-CU": .30, "P-DR-HE": .30, "P-DR-FM": .25, "P-NTC": .25}, .05),
 "DR-E02": ("Dryer", "Drum not turning",                  {"P-DR-BLT": .70, "P-DR-TN": .30}, 0.00),
 "DR-E03": ("Dryer", "Water tank error",                  {"P-DR-DP": .65, "P-FLS": .35}, .01),
 "AC-E01": ("Air conditioner", "Not cooling",             {"P-GAS": .50, "P-AC-CAP": .35, "P-AC-OPCB": .20}, .04),
 "AC-E02": ("Air conditioner", "Water dripping indoors",  {"P-AC-CH": .55, "P-AC-DPAN": .25}, .01),
 "AC-E03": ("Air conditioner", "Communication error",     {"P-AC-IPCB": .45, "P-AC-OPCB": .40, "P-AC-CBL": .25}, .05),
 "AC-E04": ("Air conditioner", "No response to remote",   {"P-AC-RC": .50, "P-AC-RB": .45}, .02),
}

CATEGORY_MIX = {"Washing machine": .30, "Refrigerator": .24, "Dishwasher": .16, "Oven": .10, "Dryer": .07, "Air conditioner": .13}
REGIONS = {  # weight, van-stock modifier, extra lead days for missing parts, cities
 "Marmara":            (.30, +.04, 0, ["İstanbul", "Bursa", "Kocaeli"]),
 "İç Anadolu":         (.18, +.01, 1, ["Ankara", "Konya", "Kayseri"]),
 "Ege":                (.15, +.03, 0, ["İzmir", "Manisa", "Aydın"]),
 "Akdeniz":            (.14, .00, 1, ["Antalya", "Adana", "Mersin"]),
 "Karadeniz":          (.09, -.07, 2, ["Samsun", "Trabzon"]),
 "Güneydoğu Anadolu":  (.08, -.10, 3, ["Gaziantep", "Diyarbakır"]),
 "Doğu Anadolu":       (.06, -.13, 4, ["Erzurum", "Van"]),
}
BRANDS = {"Arçelik": .45, "Beko": .38, "Grundig": .10, "Altus": .07}
MONTHS = pd.period_range("2025-09", "2026-10", freq="M")   # 14 months
# monthly seasonality per category (AC and fridges peak in summer)
def season(cat, m):  # m = calendar month 1..12
    if cat == "Air conditioner": return [0.2,0.2,0.3,0.5,1.0,2.2,3.0,2.8,1.4,0.5,0.3,0.2][m-1]
    if cat == "Refrigerator":    return [0.85,0.85,0.9,0.95,1.05,1.3,1.5,1.45,1.15,1.0,0.9,0.85][m-1]
    if cat == "Oven":            return [1.1,1.0,1.0,0.95,0.9,0.85,0.85,0.85,0.95,1.05,1.15,1.25][m-1]
    return 1.0

# technicians: ~64, allocated by region weight
techs = []
n_tech_total = 64
for reg, (w, *_ , cities) in REGIONS.items():
    n = max(3, round(w * n_tech_total))
    for i in range(n):
        skill = rng.normal(0, 0.035)               # van-stock / planning discipline
        diag = rng.normal(0, 0.012)                # diagnosis error modifier
        techs.append(dict(technician_id=f"T-{len(techs)+1:03d}", region=reg, city=cities[i % len(cities)],
                          tenure_years=int(rng.integers(1, 16)), _skill=skill, _diag=diag))
# two weak technicians and one strong one, so the ranking has a story
techs[5]["_skill"] -= .10; techs[5]["_diag"] += .03
techs[40]["_skill"] -= .08
techs[12]["_skill"] += .06
tech_df = pd.DataFrame(techs)

BASE_JOBS_PER_MONTH = 880
jobs, lines = [], []
jid = 0
for mi, per in enumerate(MONTHS):
    trend = 0.012 * mi / (len(MONTHS) - 1)        # van stocking improves slightly over the year
    for cat, cw in CATEGORY_MIX.items():
        n = rng.poisson(BASE_JOBS_PER_MONTH * cw * season(cat, per.month))
        cat_faults = [f for f, v in FAULTS.items() if v[0] == cat]
        fw = np.array([1.6 if f.endswith(("E01", "E02")) else 1.0 for f in cat_faults]); fw /= fw.sum()
        for _ in range(n):
            jid += 1
            reg = rng.choice(list(REGIONS), p=np.array([v[0] for v in REGIONS.values()]) / sum(v[0] for v in REGIONS.values()))
            rw, rmod, rlead, cities = REGIONS[reg]
            t = tech_df[tech_df.region == reg].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            fc = rng.choice(cat_faults, p=fw)
            _, fdesc, pmap, extra = FAULTS[fc]
            brand = rng.choice(list(BRANDS), p=list(BRANDS.values()))
            age = int(min(14, rng.gamma(2.2, 1.9)))
            day = int(rng.integers(1, per.days_in_month + 1))
            d1 = pd.Timestamp(per.year, per.month, day)
            needed = [p for p, pr in pmap.items() if rng.random() < pr]
            missing = []
            for p in needed:
                stock = np.clip(PARTS[p][1] + 0.09 + rmod + t._skill + trend, 0.05, 0.99)
                on_van = rng.random() < stock
                if not on_van: missing.append(p)
                lines.append(dict(job_id=f"J{jid:06d}", part_code=p, part_name=PARTS[p][0], qty=2 if p in ("P-WM-SA",) else 1,
                                  on_van_first_visit="Y" if on_van else "N"))
            # outcome of first visit
            if missing:
                reason = "Missing part"
            else:
                r = rng.random()
                p_diag, p_fail, p_cust = 0.025 + t._diag + extra * .5, 0.015 + extra * .5, 0.015
                if r < p_diag: reason = "Wrong diagnosis"
                elif r < p_diag + p_fail: reason = "Repair not successful"
                elif r < p_diag + p_fail + p_cust: reason = "Customer not available"
                else: reason = ""
            ftf = reason == ""
            if reason == "Missing part":
                lead = int(rng.integers(2, 8)) + rlead + (3 if any(PARTS[p][1] < .4 for p in missing) else 0)
            elif reason:
                lead = int(rng.integers(1, 5))
            else:
                lead = 0
            jobs.append(dict(
                job_id=f"J{jid:06d}", first_visit_date=d1.date().isoformat(), month=str(per),
                region=reg, city=t.city, technician_id=t.technician_id, brand=brand, category=cat,
                fault_code=fc, fault_description=fdesc, product_age_years=age,
                warranty="In warranty" if age < 3 else "Out of warranty",
                parts_needed_count=len(needed), parts_needed="; ".join(needed),
                missing_parts="; ".join(missing),
                first_time_fix=int(ftf), second_visit=int(not ftf), second_visit_reason=reason,
                second_visit_date=(d1 + pd.Timedelta(days=lead)).date().isoformat() if not ftf else "",
                days_to_resolve=lead))

jobs = pd.DataFrame(jobs).sort_values(["first_visit_date", "job_id"]).reset_index(drop=True)
lines = pd.DataFrame(lines)
parts = pd.DataFrame([dict(part_code=k, part_name=v[0], assumed_van_stock_rate=v[1]) for k, v in PARTS.items()])
faults = pd.DataFrame([dict(fault_code=k, category=v[0], fault_description=v[1]) for k, v in FAULTS.items()])

jobs.to_csv(f"{OUT}/service_jobs.csv", index=False)
lines.to_csv(f"{OUT}/job_parts.csv", index=False)
parts.to_csv(f"{OUT}/parts_catalogue.csv", index=False)
faults.to_csv(f"{OUT}/fault_codes.csv", index=False)
tech_df.drop(columns=["_skill", "_diag"]).to_csv(f"{OUT}/technicians.csv", index=False)
print(len(jobs), "jobs,", len(lines), "part lines,", jobs.technician_id.nunique(), "technicians")
print("FTF overall:", round(jobs.first_time_fix.mean() * 100, 1))
print(jobs.groupby("fault_code").first_time_fix.mean().sort_values().round(3).head(12))
print(jobs.groupby("region").first_time_fix.mean().sort_values().round(3))
print(jobs.second_visit_reason.value_counts())
