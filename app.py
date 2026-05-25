"""
Farm Nutrient Dashboard — Flask Backend
Based on Sentinel-2 crop analysis (Ludhiana District, Dec 2023 – Apr 2024)
"""

from flask import Flask, render_template, jsonify
import numpy as np
import random

app = Flask(__name__)

# ── Simulated farm data based on the notebook's clustering results ────────────
# The notebook uses K-Means/GMM/K-Medoids with 5 clusters on Sentinel-2 imagery.
# Here we simulate realistic nutrient/health data for each farm zone.

FARM_CENTER = {"lat": 30.4350, "lon": 74.1254}
FARM_BBOX = {
    "min_lat": 30.4277, "max_lat": 30.4421,
    "min_lon": 74.1145, "max_lon": 74.1364
}

MONTHS = ["December", "January", "February", "March", "April"]
MONTH_SHORT = ["Dec '23", "Jan '24", "Feb '24", "Mar '24", "Apr '24"]

# Zone definitions (from unsupervised clustering)
ZONE_DEFINITIONS = {
    "Zone A": {
        "id": "zone_a",
        "label": "Zone A — Healthy Wheat",
        "label_hi": "ज़ोन A — स्वस्थ गेहूं",
        "color": "#2ecc71",
        "cluster": 0,
        "area_pct": 35,
        "crop": "Wheat (गेहूं)",
        "health": "Excellent",
        "health_hi": "उत्तम",
    },
    "Zone B": {
        "id": "zone_b",
        "label": "Zone B — Moderate Growth",
        "label_hi": "ज़ोन B — मध्यम विकास",
        "color": "#f1c40f",
        "cluster": 1,
        "area_pct": 25,
        "crop": "Wheat (गेहूं)",
        "health": "Moderate",
        "health_hi": "मध्यम",
    },
    "Zone C": {
        "id": "zone_c",
        "label": "Zone C — Nutrient Deficient",
        "label_hi": "ज़ोन C — पोषक तत्वों की कमी",
        "color": "#e67e22",
        "cluster": 2,
        "area_pct": 20,
        "crop": "Wheat (गेहूं)",
        "health": "Low",
        "health_hi": "कम",
    },
    "Zone D": {
        "id": "zone_d",
        "label": "Zone D — Stressed / Dry",
        "label_hi": "ज़ोन D — तनावग्रस्त / सूखा",
        "color": "#e74c3c",
        "cluster": 3,
        "area_pct": 12,
        "crop": "Mixed / Fallow",
        "health": "Poor",
        "health_hi": "खराब",
    },
    "Zone E": {
        "id": "zone_e",
        "label": "Zone E — Water / Settlement",
        "label_hi": "ज़ोन E — पानी / बस्ती",
        "color": "#3498db",
        "cluster": 4,
        "area_pct": 8,
        "crop": "Non-crop",
        "health": "N/A",
        "health_hi": "लागू नहीं",
    },
}


def generate_ndvi_trends():
    """Simulated NDVI trends mimicking wheat growth cycle in Punjab."""
    np.random.seed(42)
    trends = {}
    # Wheat: low in Dec (early), rises Jan-Mar (tillering → heading), drops Apr (harvest)
    base_profiles = {
        "Zone A": [0.28, 0.52, 0.72, 0.78, 0.45],
        "Zone B": [0.22, 0.40, 0.58, 0.62, 0.38],
        "Zone C": [0.18, 0.30, 0.42, 0.46, 0.30],
        "Zone D": [0.12, 0.18, 0.25, 0.28, 0.20],
        "Zone E": [0.05, 0.06, 0.07, 0.06, 0.05],
    }
    for zone, profile in base_profiles.items():
        noise = np.random.normal(0, 0.02, len(profile))
        trends[zone] = [round(max(0, min(1, v + n)), 3) for v, n in zip(profile, noise)]
    return trends


def generate_nutrient_data():
    """Simulated nutrient levels per zone (N, P, K, Organic Carbon, pH)."""
    return {
        "Zone A": {
            "nitrogen": {"value": 285, "unit": "kg/ha", "status": "adequate", "status_hi": "पर्याप्त"},
            "phosphorus": {"value": 22, "unit": "kg/ha", "status": "adequate", "status_hi": "पर्याप्त"},
            "potassium": {"value": 195, "unit": "kg/ha", "status": "adequate", "status_hi": "पर्याप्त"},
            "organic_carbon": {"value": 0.72, "unit": "%", "status": "good", "status_hi": "अच्छा"},
            "ph": {"value": 7.2, "unit": "", "status": "neutral", "status_hi": "सामान्य"},
        },
        "Zone B": {
            "nitrogen": {"value": 210, "unit": "kg/ha", "status": "moderate", "status_hi": "मध्यम"},
            "phosphorus": {"value": 15, "unit": "kg/ha", "status": "low", "status_hi": "कम"},
            "potassium": {"value": 160, "unit": "kg/ha", "status": "moderate", "status_hi": "मध्यम"},
            "organic_carbon": {"value": 0.55, "unit": "%", "status": "moderate", "status_hi": "मध्यम"},
            "ph": {"value": 7.8, "unit": "", "status": "slightly alkaline", "status_hi": "हल्का क्षारीय"},
        },
        "Zone C": {
            "nitrogen": {"value": 140, "unit": "kg/ha", "status": "deficient", "status_hi": "कमी"},
            "phosphorus": {"value": 8, "unit": "kg/ha", "status": "deficient", "status_hi": "कमी"},
            "potassium": {"value": 120, "unit": "kg/ha", "status": "low", "status_hi": "कम"},
            "organic_carbon": {"value": 0.35, "unit": "%", "status": "low", "status_hi": "कम"},
            "ph": {"value": 8.3, "unit": "", "status": "alkaline", "status_hi": "क्षारीय"},
        },
        "Zone D": {
            "nitrogen": {"value": 85, "unit": "kg/ha", "status": "very low", "status_hi": "बहुत कम"},
            "phosphorus": {"value": 5, "unit": "kg/ha", "status": "very low", "status_hi": "बहुत कम"},
            "potassium": {"value": 90, "unit": "kg/ha", "status": "deficient", "status_hi": "कमी"},
            "organic_carbon": {"value": 0.20, "unit": "%", "status": "very low", "status_hi": "बहुत कम"},
            "ph": {"value": 8.8, "unit": "", "status": "highly alkaline", "status_hi": "अधिक क्षारीय"},
        },
        "Zone E": {
            "nitrogen": {"value": 0, "unit": "kg/ha", "status": "N/A", "status_hi": "लागू नहीं"},
            "phosphorus": {"value": 0, "unit": "kg/ha", "status": "N/A", "status_hi": "लागू नहीं"},
            "potassium": {"value": 0, "unit": "kg/ha", "status": "N/A", "status_hi": "लागू नहीं"},
            "organic_carbon": {"value": 0, "unit": "%", "status": "N/A", "status_hi": "लागू नहीं"},
            "ph": {"value": 0, "unit": "", "status": "N/A", "status_hi": "लागू नहीं"},
        },
    }


def generate_recommendations():
    """Zone-specific farming recommendations."""
    return {
        "Zone A": {
            "title": "Maintain Current Practices",
            "title_hi": "मौजूदा तरीके जारी रखें",
            "actions": [
                {"en": "Continue balanced fertilization (NPK 120:60:40)", "hi": "संतुलित खाद देना जारी रखें (NPK 120:60:40)"},
                {"en": "Optimal irrigation schedule maintained", "hi": "सिंचाई का सही समय बनाए रखें"},
                {"en": "Monitor for pest/disease in March–April", "hi": "मार्च-अप्रैल में कीट/रोग की निगरानी करें"},
            ],
            "priority": "low",
        },
        "Zone B": {
            "title": "Increase Phosphorus Application",
            "title_hi": "फॉस्फोरस की मात्रा बढ़ाएं",
            "actions": [
                {"en": "Apply DAP (Di-Ammonium Phosphate) @ 50 kg/ha", "hi": "DAP 50 kg/ha डालें"},
                {"en": "Add organic compost to improve soil carbon", "hi": "जैविक खाद डालकर मिट्टी का कार्बन बढ़ाएं"},
                {"en": "Consider foliar spray of micronutrients", "hi": "सूक्ष्म पोषक तत्वों का छिड़काव करें"},
            ],
            "priority": "medium",
        },
        "Zone C": {
            "title": "Urgent — Nutrient Deficiency Treatment",
            "title_hi": "तुरंत — पोषक तत्वों की कमी का इलाज करें",
            "actions": [
                {"en": "Apply Urea @ 65 kg/ha for nitrogen boost", "hi": "यूरिया 65 kg/ha नाइट्रोजन के लिए डालें"},
                {"en": "Apply SSP (Single Super Phosphate) @ 100 kg/ha", "hi": "SSP 100 kg/ha फॉस्फोरस के लिए डालें"},
                {"en": "Apply Muriate of Potash @ 40 kg/ha", "hi": "म्यूरेट ऑफ पोटाश 40 kg/ha डालें"},
                {"en": "Soil testing recommended for pH correction", "hi": "pH सुधार के लिए मिट्टी की जांच कराएं"},
            ],
            "priority": "high",
        },
        "Zone D": {
            "title": "Critical — Soil Rehabilitation Needed",
            "title_hi": "गंभीर — मिट्टी का पुनर्वास ज़रूरी",
            "actions": [
                {"en": "Apply Gypsum @ 2.5 t/ha to reduce alkalinity", "hi": "जिप्सम 2.5 t/ha क्षारीयता कम करने के लिए"},
                {"en": "Green manuring with Dhaincha before next season", "hi": "अगले सीज़न से पहले ढैंचा की हरी खाद लगाएं"},
                {"en": "Install drip/sprinkler irrigation", "hi": "ड्रिप/स्प्रिंकलर सिंचाई लगवाएं"},
                {"en": "Consider crop rotation with mustard/chickpea", "hi": "सरसों/चने के साथ फसल चक्र अपनाएं"},
            ],
            "priority": "critical",
        },
        "Zone E": {
            "title": "Non-Agricultural Zone",
            "title_hi": "गैर-कृषि क्षेत्र",
            "actions": [
                {"en": "This area is water body or settlement — no action needed", "hi": "यह क्षेत्र जल निकाय या बस्ती है — कोई कार्रवाई नहीं"},
            ],
            "priority": "info",
        },
    }


def generate_farm_grid():
    """Generate a 20x20 grid representing farm zones (simulated segmentation map)."""
    np.random.seed(42)
    grid = np.zeros((20, 20), dtype=int)

    # Zone A (healthy) — top-left and center
    grid[0:8, 0:10] = 0
    grid[8:12, 5:15] = 0

    # Zone B (moderate) — right side
    grid[0:8, 10:16] = 1
    grid[12:16, 5:12] = 1

    # Zone C (deficient) — bottom-center
    grid[12:18, 12:18] = 2
    grid[8:12, 15:20] = 2

    # Zone D (stressed) — bottom-left
    grid[16:20, 0:8] = 3
    grid[14:20, 0:5] = 3

    # Zone E (water/settlement) — scattered
    grid[0:4, 16:20] = 4
    grid[18:20, 16:20] = 4

    # Add some noise to make it look realistic
    for _ in range(30):
        r, c = random.randint(0, 19), random.randint(0, 19)
        grid[r, c] = random.choice([0, 1, 2, 3])

    return grid.tolist()


def compute_overall_stats():
    """Compute overall farm statistics."""
    ndvi_trends = generate_ndvi_trends()
    nutrient_data = generate_nutrient_data()

    # Weighted average NDVI (by area percentage)
    avg_ndvi_per_month = []
    for month_idx in range(5):
        weighted = sum(
            ndvi_trends[z][month_idx] * ZONE_DEFINITIONS[z]["area_pct"] / 100
            for z in ZONE_DEFINITIONS if z != "Zone E"
        )
        avg_ndvi_per_month.append(round(weighted, 3))

    # Count deficiency alerts
    alerts = []
    for zone, nutrients in nutrient_data.items():
        if zone == "Zone E":
            continue
        for nutrient, info in nutrients.items():
            if info["status"] in ("deficient", "very low", "low"):
                alerts.append({
                    "zone": zone,
                    "zone_hi": ZONE_DEFINITIONS[zone]["label_hi"],
                    "nutrient": nutrient.replace("_", " ").title(),
                    "value": info["value"],
                    "unit": info["unit"],
                    "status": info["status"],
                    "status_hi": info["status_hi"],
                })

    return {
        "total_area_ha": 42.5,
        "avg_ndvi": avg_ndvi_per_month,
        "peak_ndvi_month": "March",
        "alert_count": len(alerts),
        "alerts": alerts,
        "healthy_pct": ZONE_DEFINITIONS["Zone A"]["area_pct"],
        "at_risk_pct": ZONE_DEFINITIONS["Zone C"]["area_pct"] + ZONE_DEFINITIONS["Zone D"]["area_pct"],
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/overview")
def api_overview():
    stats = compute_overall_stats()
    return jsonify({
        "farm": FARM_CENTER,
        "bbox": FARM_BBOX,
        "total_area_ha": stats["total_area_ha"],
        "months": MONTH_SHORT,
        "avg_ndvi": stats["avg_ndvi"],
        "peak_ndvi_month": stats["peak_ndvi_month"],
        "alert_count": stats["alert_count"],
        "healthy_pct": stats["healthy_pct"],
        "at_risk_pct": stats["at_risk_pct"],
    })


@app.route("/api/zones")
def api_zones():
    return jsonify(ZONE_DEFINITIONS)


@app.route("/api/ndvi")
def api_ndvi():
    return jsonify({
        "months": MONTH_SHORT,
        "trends": generate_ndvi_trends(),
    })


@app.route("/api/nutrients")
def api_nutrients():
    return jsonify(generate_nutrient_data())


@app.route("/api/recommendations")
def api_recommendations():
    return jsonify(generate_recommendations())


@app.route("/api/farmgrid")
def api_farmgrid():
    return jsonify({
        "grid": generate_farm_grid(),
        "rows": 20,
        "cols": 20,
        "zone_colors": {str(v["cluster"]): v["color"] for v in ZONE_DEFINITIONS.values()},
        "zone_labels": {str(v["cluster"]): k for k, v in ZONE_DEFINITIONS.items()},
    })


@app.route("/api/alerts")
def api_alerts():
    stats = compute_overall_stats()
    return jsonify(stats["alerts"])


if __name__ == "__main__":
    print("\n=== Farm Nutrient Dashboard ===")
    print("    Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
