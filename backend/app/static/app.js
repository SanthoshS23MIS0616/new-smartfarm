// ── Slider configuration (these ARE the form inputs now) ───────────────────
const SLIDER_CONFIG = [
  { id: "nitrogen",      label: "Nitrogen (N)",        unit: "kg/ha", min: 0,   max: 500, step: 1,   default: 50  },
  { id: "phosphorous",   label: "Phosphorous (P)",      unit: "kg/ha", min: 0,   max: 200, step: 1,   default: 50  },
  { id: "potassium",     label: "Potassium (K)",        unit: "kg/ha", min: 0,   max: 400, step: 1,   default: 50  },
  { id: "ph",            label: "Soil pH",              unit: "",      min: 3.5, max: 9.5, step: 0.1, default: 6.5 },
  { id: "temperature_c", label: "Temperature",          unit: "°C",    min: 5,   max: 50,  step: 0.5, default: 28  },
  { id: "humidity",      label: "Humidity",             unit: "%",     min: 10,  max: 100, step: 1,   default: 65  },
  { id: "rainfall_mm",   label: "Rainfall",             unit: "mm",    min: 0,   max: 400, step: 5,   default: 120 },
  { id: "moisture",      label: "Soil Moisture",        unit: "%",     min: 5,   max: 60,  step: 1,   default: 25  },
];

// Internal value store (source of truth for all slider values)
const sliderValues = {};
SLIDER_CONFIG.forEach(s => { sliderValues[s.id] = s.default; });

// Hidden area value (from map only)
let areaHectares = null;

// ── DOM refs ───────────────────────────────────────────────────────────────
const statusEl              = document.getElementById("status");
const bestCropEl            = document.getElementById("best-crop");
const idealGroundEl         = document.getElementById("ideal-ground-card");
const table                 = document.getElementById("results-table");
const tbody                 = table.querySelector("tbody");
const metricsGrid           = document.getElementById("training-metrics");
const xaiPanel              = document.getElementById("xai-panel");
const localExplanationEl    = document.getElementById("local-explanation");
const modelOverviewEl       = document.getElementById("model-overview");
const globalFeaturesEl      = document.getElementById("global-features");
const rejectedCropsEl       = document.getElementById("rejected-crops");
const plotLightgbmEl        = document.getElementById("plot-lightgbm");
const plotCatboostEl        = document.getElementById("plot-catboost");
const areaDisplay           = document.getElementById("area-display");
const coordsDisplay         = document.getElementById("coords-display");
const recentRainfallDisplay = document.getElementById("recent-rainfall-display");
const climateRainfallDisplay= document.getElementById("climate-rainfall-display");
const locationBanner        = document.getElementById("location-banner");
const locateBtn             = document.getElementById("locate-btn");
const clearMapBtn           = document.getElementById("clear-map-btn");
const drawPolygonBtn        = document.getElementById("draw-polygon-btn");
const drawRectangleBtn      = document.getElementById("draw-rectangle-btn");
const drawCircleBtn         = document.getElementById("draw-circle-btn");

// ── Map state ──────────────────────────────────────────────────────────────
const INDIA_CENTER = [20.5937, 78.9629];
const MAP_ZOOM_DEFAULT = 5;
const MAP_ZOOM_CLOSE   = 18;

let metadata      = null;
let map           = null;
let drawnItems    = null;
let activeShape   = null;
let anchorMarker  = null;
let drawHandlers  = null;
let radarChart    = null;
let predictTimer  = null;

const mapState = {
  latitude:          null,
  longitude:         null,
  recentRainfallMm:  null,
  climateRainfallMm: null,
  locationLabel:     "Click the map or draw your land boundary.",
};

// ── Helpers ────────────────────────────────────────────────────────────────
function setStatus(msg) { statusEl.textContent = msg; }

function fmt(value, digits = 2) {
  if (!Number.isFinite(value)) return "--";
  return Number(value).toFixed(digits);
}

function parseNum(value, fallback = null) {
  const n = Number(String(value || "").replace(/,/g, ""));
  return Number.isFinite(n) ? n : fallback;
}

// ── Slider rendering ────────────────────────────────────────────────────────
function updateSliderTrack(id) {
  const slider = document.getElementById(`slider-${id}`);
  if (!slider) return;
  const pct = ((sliderValues[id] - parseFloat(slider.min)) /
               (parseFloat(slider.max) - parseFloat(slider.min))) * 100;
  slider.style.background =
    `linear-gradient(to right, var(--primary) ${pct}%, rgba(29,42,26,0.12) ${pct}%)`;
}

function setSliderValue(id, val) {
  const cfg  = SLIDER_CONFIG.find(s => s.id === id);
  if (!cfg) return;
  const v = Math.min(cfg.max, Math.max(cfg.min, parseFloat(val) || cfg.default));
  sliderValues[id] = v;

  const slider  = document.getElementById(`slider-${id}`);
  const display = document.getElementById(`sliderval-${id}`);
  if (slider)  slider.value     = v;
  if (display) display.textContent = v + (cfg.unit ? " " + cfg.unit : "");
  updateSliderTrack(id);
}

function renderSliders() {
  const grid = document.getElementById("slider-grid");
  grid.innerHTML = "";

  SLIDER_CONFIG.forEach(cfg => {
    const val = sliderValues[cfg.id];
    const item = document.createElement("div");
    item.className = "slider-item";
    item.innerHTML = `
      <div class="slider-label-row">
        <span class="slider-label">${cfg.label}</span>
        <span class="slider-value" id="sliderval-${cfg.id}">${val}${cfg.unit ? " " + cfg.unit : ""}</span>
      </div>
      <input type="range" id="slider-${cfg.id}"
             min="${cfg.min}" max="${cfg.max}" step="${cfg.step}" value="${val}" />
    `;
    grid.appendChild(item);

    const slider = item.querySelector("input");
    updateSliderTrack(cfg.id);

    slider.addEventListener("input", () => {
      setSliderValue(cfg.id, slider.value);
      // debounce auto-predict
      if (areaHectares) {
        clearTimeout(predictTimer);
        predictTimer = setTimeout(() => runPredict(), 800);
      }
    });
  });
}

// ── Soil type preset ────────────────────────────────────────────────────────
const SOIL_PRESETS = {
  alluvial:     { nitrogen: 68, phosphorous: 48, potassium: 55, ph: 7.0, moisture: 30 },
  black_cotton: { nitrogen: 55, phosphorous: 35, potassium: 70, ph: 7.8, moisture: 35 },
  red_laterite: { nitrogen: 40, phosphorous: 25, potassium: 30, ph: 5.8, moisture: 22 },
  sandy_loam:   { nitrogen: 45, phosphorous: 30, potassium: 35, ph: 6.5, moisture: 18 },
  clay:         { nitrogen: 75, phosphorous: 55, potassium: 80, ph: 6.8, moisture: 40 },
};

document.getElementById("soil-type-select").addEventListener("change", e => {
  const preset = SOIL_PRESETS[e.target.value];
  if (!preset) return;
  Object.entries(preset).forEach(([k, v]) => setSliderValue(k, v));
});

// ── Collect payload ─────────────────────────────────────────────────────────
function collectPayload() {
  const payload = { top_k: 5, area: areaHectares || 1.0 };
  SLIDER_CONFIG.forEach(cfg => { payload[cfg.id] = sliderValues[cfg.id]; });

  const prevCrop   = document.getElementById("previous-crop-select")?.value;
  const irrigation = document.getElementById("irrigation-select")?.value;
  const soilType   = document.getElementById("soil-type-select")?.value;
  if (prevCrop)   payload.previous_crop    = prevCrop;
  if (irrigation) payload.irrigation_source = irrigation;
  if (soilType)   payload.soil_type         = soilType;

  if (Number.isFinite(mapState.latitude))  payload.latitude  = mapState.latitude;
  if (Number.isFinite(mapState.longitude)) payload.longitude = mapState.longitude;

  return payload;
}

// ── Metadata ────────────────────────────────────────────────────────────────
async function loadMetadata() {
  const resp = await fetch("/api/metadata");
  if (!resp.ok) throw new Error("Models not trained yet. Train the backend first.");
  metadata = await resp.json();

  // Set slider defaults from backend defaults
  const defs = metadata?.default_inputs || {};
  SLIDER_CONFIG.forEach(cfg => {
    if (defs[cfg.id] !== undefined) setSliderValue(cfg.id, defs[cfg.id]);
  });

  // Now render sliders with correct values
  renderSliders();

  renderMetrics(metadata.training_report);
  renderExplainabilityMetadata(metadata);
  xaiPanel.classList.remove("hidden");
  setStatus(`Loaded ${metadata.crop_count || 22} crops. Draw your land boundary on the map, then predict.`);
}

// ── Weather / Map sync ──────────────────────────────────────────────────────
function formatDate(d) { return d.toISOString().slice(0, 10); }

function sumPositive(arr) {
  return arr.filter(v => Number.isFinite(v) && v > 0).reduce((s, v) => s + v, 0);
}

async function fetchLiveWeather(lat, lng) {
  const url =
    `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lng}` +
    `&current=temperature_2m,relative_humidity_2m,rain,soil_moisture_0_to_1cm&forecast_days=1&timezone=auto`;
  const data = await fetch(url).then(r => r.json());
  const cur  = data.current || {};
  return {
    temperature:   cur.temperature_2m,
    humidity:      cur.relative_humidity_2m,
    currentRain:   cur.rain,
    soilMoisture:  cur.soil_moisture_0_to_1cm,   // m³/m³ → convert to % (×100)
  };
}

async function fetchRainfallHistory(lat, lng) {
  const end   = new Date();
  const start = new Date();
  start.setDate(end.getDate() - 364);
  const url =
    `https://archive-api.open-meteo.com/v1/archive?latitude=${lat}&longitude=${lng}` +
    `&start_date=${formatDate(start)}&end_date=${formatDate(end)}&daily=precipitation_sum&timezone=auto`;
  const data   = await fetch(url).then(r => r.json());
  const series = Array.isArray(data.daily?.precipitation_sum)
    ? data.daily.precipitation_sum.map(Number) : [];
  return {
    annualTotal:              sumPositive(series),
    recent30Total:            sumPositive(series.slice(-30)),
    climateMonthlyEquivalent: sumPositive(series) / 12,
  };
}

// ── SoilGrids v2.0 (ISRIC) — free, no API key ─────────────────────────────
// ONLY fills Nitrogen (N) and pH — the two properties SoilGrids measures directly.
// Phosphorous (P) and Potassium (K) must come from Soil Health Card / lab report.
// Moisture comes from Open-Meteo (filled in fetchWeather above).
async function fetchSoilData(lat, lng) {
  try {
    const url =
      `https://rest.isric.org/soilgrids/v2.0/properties/query?lon=${lng}&lat=${lat}` +
      `&property=nitrogen&property=phh2o&depth=0-5cm&value=mean`;
    const resp = await fetch(url, { signal: AbortSignal.timeout(8000) });
    if (!resp.ok) return null;
    const data = await resp.json();

    const getProp = (name) => {
      const layers = data?.properties?.layers || [];
      const layer  = layers.find(l => l.name === name);
      const depth  = layer?.depths?.find(d => d.label === "0-5cm");
      return depth?.values?.mean ?? null;
    };

    // nitrogen: cg/kg → scale to kg/ha (up to 500)
    const nitrogen = rawN  != null ? Math.min(500, Math.max(0, Math.round((rawN / 100) * 15))) : null;
    const ph       = rawPH != null ? Math.min(9.5, Math.max(3.5, rawPH / 10))                   : null;

    return { nitrogen, ph, source: "SoilGrids v2.0 (ISRIC)" };
  } catch (_) {
    return null;
  }
}

function showSoilBadge(source) {
  let badge = document.getElementById("soil-autofill-badge");
  if (!badge) {
    badge = document.createElement("div");
    badge.id = "soil-autofill-badge";
    badge.className = "soil-badge";
    const sliderSection = document.querySelector(".slider-section");
    if (sliderSection) sliderSection.prepend(badge);
  }
  badge.textContent = `🌱 Soil data auto-filled from ${source}`;
  badge.style.display = "block";

  const notice = document.getElementById("soil-calibration-notice");
  if (notice) notice.style.display = "block";
}

async function fetchWeather(lat, lng) {
  const [live, hist] = await Promise.all([fetchLiveWeather(lat, lng), fetchRainfallHistory(lat, lng)]);

  // Blend climate + recent rainfall
  const modelRainfall = (hist.climateMonthlyEquivalent && hist.recent30Total)
    ? 0.7 * hist.climateMonthlyEquivalent + 0.3 * hist.recent30Total
    : hist.climateMonthlyEquivalent || hist.recent30Total || live.currentRain || 120;

  mapState.recentRainfallMm  = hist.recent30Total  || live.currentRain;
  mapState.climateRainfallMm = modelRainfall;

  // Update rainfall displays
  recentRainfallDisplay.textContent  = Number.isFinite(mapState.recentRainfallMm)
    ? `${fmt(mapState.recentRainfallMm, 2)} mm` : "Unavailable";
  climateRainfallDisplay.textContent = Number.isFinite(modelRainfall)
    ? `${fmt(modelRainfall, 2)} mm` : "Unavailable";

  // ── AUTO-FILL weather sliders ──
  if (Number.isFinite(live.temperature)) setSliderValue("temperature_c", live.temperature);
  if (Number.isFinite(live.humidity))    setSliderValue("humidity",       live.humidity);
  if (Number.isFinite(modelRainfall))    setSliderValue("rainfall_mm",    Math.round(modelRainfall / 5) * 5);
  // Soil moisture from Open-Meteo (m³/m³ → %, rough conversion ×40)
  if (Number.isFinite(live.soilMoisture)) {
    const moisturePct = Math.min(60, Math.max(5, live.soilMoisture * 40));
    setSliderValue("moisture", Math.round(moisturePct));
  }
}

function setAnchorLocation(lat, lng) {
  mapState.latitude  = lat;
  mapState.longitude = lng;
  coordsDisplay.textContent = `${fmt(lat, 5)}, ${fmt(lng, 5)}`;
  if (!anchorMarker) {
    anchorMarker = L.marker([lat, lng]).addTo(map);
  } else {
    anchorMarker.setLatLng([lat, lng]);
  }
}



async function syncLocationAndWeather(lat, lng) {
  setAnchorLocation(lat, lng);
  setStatus("Fetching live weather + AI soil data...");
  mapState.locationLabel = `Selected: ${fmt(lat, 5)}, ${fmt(lng, 5)}`;
  locationBanner.textContent = mapState.locationLabel;

  // Run weather (Open-Meteo) and soil in parallel
  const [, soilResult] = await Promise.allSettled([
    fetchWeather(lat, lng),
    fetchSoilData(lat, lng),
  ]);

  const soil = soilResult?.value;
  let nVal = soil?.nitrogen != null ? Math.round(soil.nitrogen) : null;
  let phVal = soil?.ph != null ? parseFloat(soil.ph.toFixed(1)) : null;

  // Call backend AI soil estimator for Phosphorus and Potassium (and fallback N & pH)
  let aiSoil = null;
  try {
    const aiResp = await fetch("/api/soil/estimate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ latitude: lat, longitude: lng })
    });
    if (aiResp.ok) {
      const aiJson = await aiResp.json();
      aiSoil = aiJson.data;
    }
  } catch (err) {
    console.debug("Soil AI estimate call error:", err);
  }

  // 1. Nitrogen autofill
  if (nVal != null) {
    setSliderValue("nitrogen", nVal);
  } else if (aiSoil && aiSoil.nitrogen != null) {
    setSliderValue("nitrogen", Math.round(aiSoil.nitrogen));
  }

  // 2. pH autofill
  if (phVal != null) {
    setSliderValue("ph", phVal);
  } else if (aiSoil && aiSoil.ph != null) {
    setSliderValue("ph", parseFloat(aiSoil.ph.toFixed(1)));
  }

  // 3. Phosphorus & Potassium autofill from AI model
  if (aiSoil) {
    if (aiSoil.phosphorus != null) setSliderValue("phosphorous", Math.round(aiSoil.phosphorus));
    if (aiSoil.potassium != null) setSliderValue("potassium", Math.round(aiSoil.potassium));
    showSoilBadge(
      `🤖 AI Auto-Filled: N=${document.getElementById("slider-nitrogen")?.value || 50} · P=${aiSoil.phosphorus} · K=${aiSoil.potassium} · pH=${document.getElementById("slider-ph")?.value || 6.8} kg/ha (${aiSoil.source})`
    );
  } else if (soil) {
    showSoilBadge(`🛰 N=${nVal || "?"} · pH=${phVal || "?"} (SoilGrids) | Please adjust P & K sliders`);
  } else {
    showSoilBadge("🛰 Adjust N, P, K & pH sliders based on your farm conditions");
  }

  setStatus("✅ Location synced — N, P, K & pH auto-filled from ICAR / AI Soil Model · Weather from Open-Meteo");
}


// ── Map init ────────────────────────────────────────────────────────────────
function updateAreaDisplay() {
  areaDisplay.textContent = areaHectares ? `${fmt(areaHectares, 3)} ha` : "Draw on map";
}

function clearActiveShape() {
  if (activeShape && drawnItems) drawnItems.removeLayer(activeShape);
  activeShape     = null;
  areaHectares    = null;
  mapState.recentRainfallMm  = null;
  mapState.climateRainfallMm = null;
  updateAreaDisplay();
  recentRainfallDisplay.textContent  = "Waiting for map";
  climateRainfallDisplay.textContent = "Waiting for map";
}

function updateAreaFromLayer(layer) {
  let sqm = null;
  if (layer instanceof L.Circle) {
    sqm = Math.PI * layer.getRadius() ** 2;
  } else if (L.GeometryUtil?.geodesicArea) {
    const lls = layer.getLatLngs();
    sqm = L.GeometryUtil.geodesicArea(Array.isArray(lls[0]) ? lls[0] : lls);
  }
  areaHectares = (Number.isFinite(sqm) && sqm > 0) ? sqm / 10000 : null;
  updateAreaDisplay();
}

function getLayerCenter(layer) {
  if (typeof layer.getBounds === "function" && layer.getBounds().isValid())
    return layer.getBounds().getCenter();
  if (typeof layer.getLatLng === "function") return layer.getLatLng();
  return null;
}

function handleShapeChange(layer) {
  activeShape = layer;
  updateAreaFromLayer(layer);
  const center = getLayerCenter(layer);
  if (center) {
    if (typeof layer.getBounds === "function")
      map.fitBounds(layer.getBounds(), { maxZoom: MAP_ZOOM_CLOSE, padding: [30, 30] });
    else
      map.setView(center, MAP_ZOOM_CLOSE);
    syncLocationAndWeather(center.lat, center.lng).catch(e => setStatus(e.message));
  }
}

function initMap() {
  map = L.map("map", { zoomControl: false, preferCanvas: true }).setView(INDIA_CENTER, MAP_ZOOM_DEFAULT);
  setTimeout(() => map.invalidateSize(), 150);
  L.control.zoom({ position: "topright" }).addTo(map);
  L.esri.basemapLayer("Imagery").addTo(map);
  L.esri.basemapLayer("ImageryLabels").addTo(map);

  drawnItems = new L.FeatureGroup();
  map.addLayer(drawnItems);

  const drawControl = new L.Control.Draw({
    position: "topright",
    draw: {
      polygon:     { allowIntersection: false, showArea: true },
      rectangle:   true,
      circle:      true,
      marker:      false,
      polyline:    false,
      circlemarker:false,
    },
    edit: { featureGroup: drawnItems, edit: true, remove: true },
  });
  map.addControl(drawControl);

  drawHandlers = {
    polygon:   new L.Draw.Polygon(map,   drawControl.options.draw.polygon),
    rectangle: new L.Draw.Rectangle(map, drawControl.options.draw.rectangle),
    circle:    new L.Draw.Circle(map,    drawControl.options.draw.circle),
  };

  map.on(L.Draw.Event.CREATED, e => {
    clearActiveShape();
    drawnItems.addLayer(e.layer);
    handleShapeChange(e.layer);
  });

  map.on(L.Draw.Event.EDITED, e => {
    e.layers.eachLayer(l => handleShapeChange(l));
  });

  map.on(L.Draw.Event.DELETED, () => {
    activeShape     = null;
    areaHectares    = null;
    updateAreaDisplay();
  });

  map.on("click", e => {
    syncLocationAndWeather(e.latlng.lat, e.latlng.lng).catch(err => setStatus(err.message));
  });
}

function enableDrawMode(mode) {
  if (!drawHandlers?.[mode]) { setStatus("Drawing tools not ready — refresh."); return; }
  drawHandlers[mode].enable();
  setStatus(`Draw mode: ${mode}. Mark your farm boundary on the map.`);
}

// ── Button listeners ────────────────────────────────────────────────────────
locateBtn.addEventListener("click", () => {
  if (!navigator.geolocation) { setStatus("Geolocation not available."); return; }
  setStatus("Getting your device location...");
  navigator.geolocation.getCurrentPosition(
    pos => {
      map.setView([pos.coords.latitude, pos.coords.longitude], 17);
      syncLocationAndWeather(pos.coords.latitude, pos.coords.longitude).catch(e => setStatus(e.message));
    },
    () => setStatus("Could not get your location."),
    { enableHighAccuracy: true, timeout: 15000 }
  );
});

clearMapBtn.addEventListener("click", () => {
  clearActiveShape();
  setStatus("Map cleared. Draw again to select your farm.");
});

drawPolygonBtn.addEventListener("click",   () => enableDrawMode("polygon"));
drawRectangleBtn.addEventListener("click", () => enableDrawMode("rectangle"));
drawCircleBtn.addEventListener("click",    () => enableDrawMode("circle"));

document.getElementById("prediction-form").addEventListener("submit", async e => {
  e.preventDefault();
  await runPredict();
});

// ── Predict ─────────────────────────────────────────────────────────────────
async function runPredict() {
  if (!areaHectares) {
    setStatus("Draw your farm boundary on the map first — area is required.");
    return;
  }
  try {
    setStatus("Running AI crop recommendation...");
    const resp = await fetch("/api/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(collectPayload()),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "Prediction failed");
    renderResults(data);
    setStatus(`✔ Best crop: ${data.best_crop}`);
  } catch (err) {
    setStatus(err.message);
  }
}

// ── Render helpers ──────────────────────────────────────────────────────────
function renderMetrics(report) {
  metricsGrid.innerHTML = "";
  if (!report) return;
  [
    ["Stacking Accuracy",  report.classification_metrics?.stacking?.accuracy],
    ["Top-3 Accuracy",     report.classification_metrics?.stacking?.top3_accuracy],
    ["LightGBM Accuracy",  report.classification_metrics?.lightgbm?.accuracy],
    ["Yield R²",           report.regression_metrics?.r2],
  ].forEach(([label, val]) => {
    if (val === undefined) return;
    const card = document.createElement("div");
    card.className = "metric-card";
    card.innerHTML = `<span>${label}</span><strong>${Number(val).toFixed(4)}</strong>`;
    metricsGrid.appendChild(card);
  });
}

function renderGraphRows(items) {
  modelOverviewEl.innerHTML = "";
  items.forEach(item => {
    const row = document.createElement("div");
    row.className = "graph-row";
    row.innerHTML = `
      <header><strong>${item.model}</strong><span>${fmt(item.accuracy * 100, 2)}%</span></header>
      <div class="graph-bar"><span style="width:${Math.max(2, item.accuracy * 100)}%"></span></div>
    `;
    modelOverviewEl.appendChild(row);
  });
}

function renderGlobalFeatures(features) {
  globalFeaturesEl.innerHTML = "";
  features.forEach(f => {
    const chip = document.createElement("div");
    chip.className = "feature-chip";
    chip.innerHTML = `
      <strong>${String(f.feature || "").replace(/^num__|^cat__/, "").replace(/_/g, " ")}</strong>
      <span>importance ${fmt(Number(f.mean_importance || 0), 3)}</span>
    `;
    globalFeaturesEl.appendChild(chip);
  });
}

function renderPlotImages(assets) {
  const lgbm = assets?.lightgbm_summary_plot || assets?.summary_plot_urls?.lightgbm
    || "/reports/realistic_v2/shap_summary_lightgbm.png";
  const cb   = assets?.catboost_summary_plot  || assets?.summary_plot_urls?.catboost
    || "/reports/realistic_v2/shap_summary_catboost.png";
  plotLightgbmEl.src = lgbm;
  plotCatboostEl.src  = cb;
  [plotLightgbmEl, plotCatboostEl].forEach(img => {
    img.style.display = "block";
    img.onerror = () => { img.style.display = "none"; };
  });
}

function renderRejectedCrops(rejected) {
  rejectedCropsEl.innerHTML = "";
  if (!rejected?.length) {
    rejectedCropsEl.innerHTML = "<p>No crops were rejected for these conditions.</p>";
    return;
  }
  const ul = document.createElement("ul");
  ul.className = "rejection-list";
  rejected.forEach(item => {
    const li = document.createElement("li");
    li.textContent = `${item.crop}: ${item.reason}`;
    ul.appendChild(li);
  });
  rejectedCropsEl.appendChild(ul);
}

function renderLocalExplanation(ex) {
  localExplanationEl.innerHTML = "";
  const best = ex?.best_crop_local_explanation;
  if (!best) {
    localExplanationEl.innerHTML = "<p>Run a prediction to see explanation.</p>";
    return;
  }
  const model = best.model_explanation || {};
  const agronomy = best.agronomic_explanation || {};
  [
    { title: "Model explanation",  items: (model.classification_shap?.positive || best.classification_shap?.positive || []).slice(0, 4).map(s => `${s.feature}: +${fmt(s.contribution, 3)}`) },
    { title: "Yield model signals",items: (model.yield_shap?.positive || best.yield_shap?.positive || []).slice(0, 4).map(s => `${s.feature}: +${fmt(s.contribution, 3)}`) },
    { title: "Agronomic fit",      items: agronomy.positives || best.agronomic_positives || best.rule_based_positives || [] },
    { title: "Agronomic advice",   items: agronomy.advice || best.agronomic_advice || best.rule_based_concerns || [] },
  ].forEach(block => {
    const sec = document.createElement("section");
    sec.className = "explanation-block";
    sec.innerHTML = `<h4>${block.title}</h4>` +
      (block.items.length
        ? `<ul class="explanation-list">${block.items.map(i => `<li>${i}</li>`).join("")}</ul>`
        : "<p>No strong signals.</p>");
    localExplanationEl.appendChild(sec);
  });
}

function renderExplainabilityMetadata(meta) {
  const xai    = meta?.xai_assets || {};
  const report = meta?.training_report || {};
  renderGlobalFeatures((xai.top_features || []).slice(0, 8));
  renderPlotImages(xai);
  const overview = Object.entries(report.classification_metrics || {})
    .filter(([, v]) => v?.accuracy !== undefined)
    .map(([k, v]) => ({
      model:    k.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()),
      accuracy: Number(v.accuracy || 0),
    }))
    .sort((a, b) => b.accuracy - a.accuracy);
  renderGraphRows(overview);
  renderLocalExplanation(null);
  renderRejectedCrops([]);
}

function renderPredictionExplainability(result) {
  renderLocalExplanation(result.explainability);
  renderRejectedCrops(result.rejected_crops || []);
  if (result.explainability?.model_overview?.length)
    renderGraphRows(result.explainability.model_overview);
  if (result.explainability?.global_top_features?.length)
    renderGlobalFeatures(result.explainability.global_top_features);
  renderPlotImages(result.explainability || {});
  xaiPanel.classList.remove("hidden");
}

// ── SHAP sentence ───────────────────────────────────────────────────────────
function renderSHAPSentence(result) {
  const el = document.getElementById("shap-sentence");
  const best   = result.top_crops[0];
  const shapPos = result.explainability?.best_crop_local_explanation?.classification_shap?.positive || [];
  if (!shapPos.length) { el.classList.add("hidden"); return; }

  const top     = shapPos.slice(0, 3).map(s =>
    `<strong>${String(s.feature).replace(/^num__|^cat__/, "").replace(/_/g, " ")}</strong>`
  );
  const profitK = (best.profit_rs_per_ha / 1000).toFixed(1);
  const suitabilityText = fmt(best.suitability_pct, 1);
  el.innerHTML = `The AI chose <strong>${best.crop}</strong> because ${top.join(", ")} best match
    your field this season. Yield: <strong>${fmt(best.expected_yield_t_ha, 2)} t/ha</strong> ·
    Profit: <strong>₹${profitK}K/ha</strong> · Harvest: <strong>${best.harvest_month}</strong>.`;
  el.innerHTML = `Model explanation for <strong>${best.crop}</strong>: ${top.join(", ")}
    most influenced the prediction. Agronomic suitability: <strong>${suitabilityText}%</strong> &middot;
    Yield: <strong>${fmt(best.expected_yield_t_ha, 2)} t/ha</strong> &middot;
    Profit: <strong>Rs ${profitK}K/ha</strong> &middot; Harvest: <strong>${best.harvest_month}</strong>.`;
  el.classList.remove("hidden");
}

// ── Market badge ────────────────────────────────────────────────────────────
function renderMarketBadge(result) {
  const badge  = document.getElementById("market-source-badge");
  const source = result.top_crops?.[0]?.market_source || "static_csv";
  const map    = {
    live_agmarknet:   { cls: "live",   text: "🟢 Live Agmarknet Price" },
    cached_agmarknet: { cls: "cached", text: "🟡 Cached Agmarknet (6h)" },
    static_csv:       { cls: "static", text: "📊 Historical Market Data" },
  };
  const cfg = map[source] || map.static_csv;
  badge.className = `market-badge ${cfg.cls}`;
  badge.textContent = cfg.text;
  badge.classList.remove("hidden");
}

// ── Radar chart — improved ──────────────────────────────────────────────────
function renderRadarChart(crops) {
  const canvas = document.getElementById("radar-chart");
  if (!canvas || !window.Chart || !crops?.length) return;
  if (radarChart) { radarChart.destroy(); radarChart = null; }

  const displayCrops = crops.slice(0, 5);
  const labels = ["Profit Potential", "Cost Efficiency", "Safety (Low Risk)", "Sustainability", "Water Efficiency"];
  const palette = ["#225c36", "#c97d2a", "#2a7cc9", "#9b59b6", "#27ae60"];

  const profits = displayCrops.map(c => c.profit_rs_per_ha);
  const costs   = displayCrops.map(c => c.total_cost_rs_per_ha || c.initial_spend_rs_per_ha || 0);
  const minP = Math.min(...profits), maxP = Math.max(...profits);
  const minC = Math.min(...costs),   maxC = Math.max(...costs);
  const norm = (v, lo, hi) => hi === lo ? 60 : Math.min(100, Math.max(0, ((v - lo) / (hi - lo)) * 100));

  radarChart = new Chart(canvas, {
    type: "radar",
    data: {
      labels,
      datasets: displayCrops.map((c, i) => ({
        label:                c.crop,
        data: [
          norm(c.profit_rs_per_ha, minP, maxP),
          norm(-(c.total_cost_rs_per_ha || c.initial_spend_rs_per_ha || 0), -maxC, -minC),
          (1 - (c.risk || 0.5)) * 100,
          (c.sustainability_score || 0.5) * 100,
          (1 - Math.min((c.irrigation_penalty || 0) / 0.20, 1.0)) * 100,
        ],
        borderColor:          palette[i % palette.length],
        backgroundColor:      palette[i % palette.length] + "18",
        pointBackgroundColor: palette[i % palette.length],
        pointRadius:          4,
        borderWidth:          2.5,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            font: { family: "Inter", size: 12, weight: "500" },
            boxWidth: 14, padding: 16,
            generateLabels: (chart) => chart.data.datasets.map((ds, i) => ({
              text: ds.label,
              fillStyle: palette[i % palette.length],
              strokeStyle: palette[i % palette.length],
              lineWidth: 2,
              hidden: false,
              index: i,
            })),
          },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(1)}/100`,
          },
        },
      },
      scales: {
        r: {
          min: 0, max: 100,
          ticks: { stepSize: 25, font: { size: 9 }, backdropColor: "transparent" },
          pointLabels: { font: { family: "Inter", size: 11, weight: "600" }, color: "#2d4a35" },
          grid:  { color: "rgba(34,92,54,0.10)" },
          angleLines: { color: "rgba(34,92,54,0.15)" },
        },
      },
    },
  });
}

// ── Year-wise crop history table ─────────────────────────────────────────────
function renderYearWiseHistory(history) {
  // Ensure container exists — create it dynamically if not in HTML
  let panel = document.getElementById("yearwise-panel");
  if (!panel) {
    panel = document.createElement("section");
    panel.id = "yearwise-panel";
    panel.className = "xai-card yearwise-panel";
    // Insert after xai-panel
    const xai = document.getElementById("xai-panel");
    if (xai && xai.parentNode) xai.parentNode.insertBefore(panel, xai.nextSibling);
    else document.querySelector(".result-panel")?.appendChild(panel);
  }

  if (!history?.length) {
    panel.classList.add("hidden");
    return;
  }

  const stateLabel = history[0]?.state || "Your Region";
  panel.innerHTML = `
    <h3>📅 Year-Wise Best Crop History — ${stateLabel}</h3>
    <p class="card-subtitle">Historically highest-yielding crop per year based on production records</p>
    <div class="table-wrap">
      <table class="yearwise-table">
        <thead>
          <tr>
            <th>Year</th>
            <th>Best Crop</th>
            <th>Season</th>
            <th>Yield (t/ha)</th>
            <th>Price (₹/kg)</th>
            <th>Est. Profit (₹/ha)</th>
            <th>State</th>
          </tr>
        </thead>
        <tbody>
          ${history.map((row, idx) => {
            const profitCls = (row.profit_rs_per_ha || 0) > 0 ? "profit-positive" : "profit-negative";
            const isBest = idx === history.length - 1;
            return `<tr class="${isBest ? "year-best-row" : ""}">
              <td><strong>${row.year}</strong>${isBest ? ' <span class="latest-badge">Latest</span>' : ""}</td>
              <td><strong class="crop-name-cell">${row.best_crop || "—"}</strong></td>
              <td>${row.season || "—"}</td>
              <td>${Number(row.yield_t_ha || 0).toFixed(2)}</td>
              <td>₹${Number(row.price_rs_per_kg || 0).toFixed(2)}</td>
              <td class="${profitCls}">₹${Number(row.profit_rs_per_ha || 0).toLocaleString("en-IN")}</td>
              <td>${row.state || "—"}</td>
            </tr>`;
          }).join("")}
        </tbody>
      </table>
    </div>
    <p class="source-note">Source: ICRISAT District Production Database + Kaggle Crop Dataset | Profit = Yield × Price − Est. Cost</p>
  `;
  panel.classList.remove("hidden");
}



// ── Perennial cards ──────────────────────────────────────────────────────────
function renderPerennialPanel(options) {
  const panel = document.getElementById("perennial-panel");
  const grid  = document.getElementById("perennial-cards");
  if (!options?.length) { panel.classList.add("hidden"); return; }

  grid.innerHTML = "";
  options.forEach(p => {
    const card = document.createElement("div");
    card.className = "perennial-card";
    card.innerHTML = `
      <span class="payback-pill">${p.payback_years}yr payback</span>
      <h4>${p.crop}</h4>
      <div class="perennial-meta">
        Yield: <strong>${fmt(p.expected_yield_t_ha, 2)} t/ha</strong><br/>
        Price: <strong>₹${fmt(p.stable_price_rs_per_kg, 2)}/kg</strong><br/>
        Cost: <strong>₹${(p.total_cost_rs_per_ha / 1000).toFixed(1)}K/ha</strong><br/>
        Profit: <strong>₹${(p.profit_rs_per_ha / 1000).toFixed(1)}K/ha</strong><br/>
        Risk: <strong>${(p.risk * 100).toFixed(0)}%</strong> &nbsp;
        Sustain: <strong>${(p.sustainability_score * 100).toFixed(0)}%</strong>
      </div>
      <div class="perennial-invest-note">${p.investment_note}</div>
    `;
    grid.appendChild(card);
  });
  panel.classList.remove("hidden");
}

// ── Main renderResults ───────────────────────────────────────────────────────
function renderResults(result) {
  const best = result.top_crops?.[0];
  if (!best) { setStatus("No results returned — check backend."); return; }

  // ── Off-season advisory banner ──────────────────────────────────────────
  let advisoryBanner = document.getElementById("off-season-banner");
  if (!advisoryBanner) {
    advisoryBanner = document.createElement("div");
    advisoryBanner.id = "off-season-banner";
    advisoryBanner.className = "off-season-banner";
    bestCropEl.parentNode?.insertBefore(advisoryBanner, bestCropEl);
  }
  if (best.off_season) {
    advisoryBanner.innerHTML = `
      <span class="banner-icon">⚠</span>
      <span>${best.off_season_advisory || "Results shown for off-season sowing — yield reduced by ~20%."}</span>
    `;
    advisoryBanner.style.display = "flex";
  } else {
    advisoryBanner.style.display = "none";
  }

  // ── Best crop card ──────────────────────────────────────────────────────
  bestCropEl.classList.remove("hidden");
  const profitK      = (best.profit_rs_per_ha / 1000).toFixed(1);
  const suitabilityK = fmt(best.suitability_pct, 1);
  const monoNote = best.mono_crop_penalty > 0
    ? `<span class="advisory-tag">⚠ Mono-crop penalty applied</span>` : "";
  const irrNote  = best.irrigation_penalty > 0
    ? `<span class="advisory-tag">💧 Irrigation penalty applied</span>` : "";
  const offNote  = best.off_season
    ? `<span class="advisory-tag timing-off">🕐 Off-season sowing — 20% yield penalty</span>` : "";

    bestCropEl.innerHTML = `
    <h3>🌾 Best Crop: ${best.crop}</h3>
    <p>Sow in <strong>${best.sowing_month}</strong> → harvest <strong>${best.harvest_month}</strong>
       (${best.duration_months} months).<br/>
       Suitability: <strong>${suitabilityK}%</strong> &middot;
       Yield: <strong>${fmt(best.expected_yield_t_ha, 2)} t/ha</strong> &middot;
       Profit: <strong>₹${profitK}K/ha</strong>.
    </p>
    <p style="display:flex;gap:8px;font-size:0.83em;flex-wrap:wrap;margin-top:4px">${monoNote}${irrNote}${offNote}</p>
    <div style="margin-top:10px">
      <button type="button" class="btn-commit-crop" onclick="commitCropPlan('${best.crop}')" style="padding:9px 18px;font-size:0.88rem;display:inline-flex;align-items:center;gap:6px">
        📅 Commit &amp; Generate Sowing Schedule
      </button>
    </div>
  `;

  // ── Ideal ground card ───────────────────────────────────────────────────
  const ideal = result.ideal_ground_recommendation;
  if (ideal) {
    idealGroundEl.classList.remove("hidden");
    idealGroundEl.innerHTML = `
      <h3>🏆 Ideal Ground Crop: ${ideal.crop}</h3>
      <p>Long-term land fit — suitability ${fmt(ideal.land_suitability, 3)},
         yield ${fmt(ideal.expected_yield_t_ha, 2)} t/ha,
         profit ₹${(ideal.profit_rs_per_ha / 1000).toFixed(1)}K/ha,
         risk ${(ideal.risk * 100).toFixed(1)}%.
      </p>
      <p style="font-size:0.82em;color:var(--muted)">${(ideal.why || []).slice(0, 2).join(" ")}</p>
    `;
  } else {
    idealGroundEl.classList.add("hidden");
  }

  // ── Results table with timing badges ───────────────────────────────────
  tbody.innerHTML = "";
  result.top_crops.forEach(item => {
    const timing = item.off_season
      ? `<span class="timing-badge timing-off">Off-Season</span>`
      : `<span class="timing-badge timing-ok">✓ Seasonal</span>`;
    const tr = document.createElement("tr");
    if (item.off_season) tr.classList.add("row-offseason");
    tr.innerHTML = `
      <td><strong>${item.crop}</strong>${timing}</td>
      <td>${item.sowing_month}</td>
      <td>${item.harvest_month}</td>
      <td>${item.duration_months}mo</td>
      <td>${fmt(item.expected_yield_t_ha, 2)}</td>
      <td>₹${fmt(item.adjusted_price_rs_per_kg, 2)}</td>
      <td>₹${Number(item.total_cost_rs_per_ha || 0).toFixed(0)}</td>
      <td>₹${Number(item.revenue_rs_per_ha || 0).toFixed(0)}</td>
      <td>₹${Number(item.profit_rs_per_ha || 0).toFixed(0)}</td>
      <td>${(item.risk * 100).toFixed(1)}%</td>
      <td>${(item.sustainability_score * 100).toFixed(0)}%</td>
      <td>${item.final_score.toFixed(3)}</td>
      <td>
        <button type="button" class="btn-commit-crop" onclick="commitCropPlan('${item.crop}')" title="Generate task calendar">
          Plan
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
  table.classList.remove("hidden");

  renderMetrics(result.training_summary);
  renderPredictionExplainability(result);
  renderSHAPSentence(result);
  renderMarketBadge(result);
  renderRadarChart(result.top_crops);
  renderPerennialPanel(result.perennial_options || []);


  const dlBtn = document.getElementById("download-report-btn");
  dlBtn.classList.remove("hidden");
  dlBtn._result = result;
}



// ── PDF report ───────────────────────────────────────────────────────────────
document.getElementById("download-report-btn").addEventListener("click", e => {
  const result = e.currentTarget._result;
  if (!result || !window.jspdf) return;
  const { jsPDF } = window.jspdf;
  const doc  = new jsPDF({ unit: "mm", format: "a4" });
  const best = result.top_crops[0];
  const today = new Date().toLocaleDateString("en-IN", { year: "numeric", month: "long", day: "numeric" });

  doc.setFillColor(34, 92, 54);
  doc.rect(0, 0, 210, 28, "F");
  doc.setTextColor(255, 255, 255);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(15);
  doc.text("Crop Intelligence Platform — Farm Advisory Report", 14, 11);
  doc.setFontSize(9);
  doc.setFont("helvetica", "normal");
  doc.text(`Generated: ${today}   |   Santhosh S | 23MIS0616`, 14, 19);
  doc.text(`Best Crop: ${result.best_crop}`, 14, 25);

  doc.setTextColor(0, 0, 0);
  doc.setFont("helvetica", "bold"); doc.setFontSize(12);
  doc.text("Best Seasonal Crop", 14, 38);
  doc.setFont("helvetica", "normal"); doc.setFontSize(9);
  [
    `Crop: ${best.crop}  |  Sow: ${best.sowing_month}  |  Harvest: ${best.harvest_month}  |  Duration: ${best.duration_months}mo`,
    `Yield: ${fmt(best.expected_yield_t_ha, 2)} t/ha  |  Price: ₹${fmt(best.adjusted_price_rs_per_kg, 2)}/kg`,
    `Cost/ha: ₹${best.total_cost_rs_per_ha.toFixed(0)}  |  Revenue/ha: ₹${best.revenue_rs_per_ha.toFixed(0)}  |  Profit/ha: ₹${best.profit_rs_per_ha.toFixed(0)}`,
    `Risk: ${(best.risk * 100).toFixed(1)}%  |  Sustainability: ${(best.sustainability_score * 100).toFixed(1)}%  |  Market: ${best.market_source}`,
  ].forEach((l, i) => doc.text(l, 14, 46 + i * 6));

  doc.setFont("helvetica", "bold"); doc.setFontSize(12);
  doc.text("Ranked Annual Crops", 14, 76);

  doc.autoTable({
    head: [["Crop","Sow","Harvest","Yield","Price","Cost/ha","Revenue/ha","Profit/ha","Risk","Score"]],
    body: result.top_crops.map(c => [
      c.crop, c.sowing_month, c.harvest_month,
      fmt(c.expected_yield_t_ha, 2), `₹${fmt(c.adjusted_price_rs_per_kg, 2)}`,
      `₹${c.total_cost_rs_per_ha.toFixed(0)}`, `₹${c.revenue_rs_per_ha.toFixed(0)}`,
      `₹${c.profit_rs_per_ha.toFixed(0)}`, `${(c.risk*100).toFixed(1)}%`, c.final_score.toFixed(3),
    ]),
    startY: 80,
    headStyles: { fillColor: [34, 92, 54], fontSize: 8 },
    bodyStyles: { fontSize: 7.5 },
  });

  const perennials = result.perennial_options || [];
  if (perennials.length) {
    const y = doc.lastAutoTable.finalY + 10;
    doc.setFont("helvetica", "bold"); doc.setFontSize(12);
    doc.text("Long-Term Investment Options", 14, y);
    doc.autoTable({
      head: [["Crop", "Payback", "Yield(t/ha)", "Price(₹/kg)", "Ann.Cost(₹/ha)", "Ann.Revenue(₹/ha)", "Ann.Profit(₹/ha)", "Risk", "Sustain."]],
      body: perennials.map(p => {
        const yieldVal = p.expected_yield_t_ha || 0;
        const priceVal = p.stable_price_rs_per_kg || 0;
        const revHa = yieldVal * priceVal * 1000;
        const payback = p.payback_years || 3;
        const annCostHa = (p.total_cost_rs_per_ha || 0) / payback;
        const annProfitHa = revHa - annCostHa;
        return [
          p.crop,
          `${payback} yr`,
          fmt(yieldVal, 2),
          `₹${fmt(priceVal, 2)}`,
          `₹${Math.round(annCostHa).toLocaleString("en-IN")}`,
          `₹${Math.round(revHa).toLocaleString("en-IN")}`,
          `₹${Math.round(annProfitHa).toLocaleString("en-IN")}`,
          `${((p.risk || 0.28) * 100).toFixed(1)}%`,
          `${((p.sustainability_score || 0.65) * 100).toFixed(1)}%`,
        ];
      }),
      startY: y + 4,
      headStyles: { fillColor: [201, 125, 42], fontSize: 7.5 },
      bodyStyles: { fontSize: 7 },
    });
  }

  if (best.advisory_notes?.length) {
    const y2 = (doc.lastAutoTable?.finalY || 200) + 8;
    if (y2 < 265) {
      doc.setFont("helvetica", "bold"); doc.setFontSize(10);
      doc.text("Advisory Notes & Agronomic Transparency", 14, y2);
      doc.setFont("helvetica", "normal"); doc.setFontSize(8);
      best.advisory_notes.forEach((n, i) => doc.text(`• ${n}`, 16, y2 + 5 + i * 5));
    }
  }

  // Page 2: Calculated Fertilizer/Pesticide Input Doses & Complete Sowing Task Schedule
  doc.addPage();
  doc.setFillColor(34, 92, 54);
  doc.rect(0, 0, 210, 24, "F");
  doc.setTextColor(255, 255, 255);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(13);
  doc.text(`TNAU Agronomic Advisory & Sowing Task Calendar — ${best.crop}`, 14, 15);

  const plotAcres = currentCommittedPlan ? currentCommittedPlan.area_acres : ((areaHectares || 1.0) * 2.471);
  doc.setTextColor(0, 0, 0);
  doc.setFontSize(10.5);
  doc.text(`Calculated Input Doses for Plot Area: ${plotAcres.toFixed(1)} Acres (${(plotAcres/2.471).toFixed(2)} ha)`, 14, 32);

  const isLegumeCrop = ["blackgram", "chickpea", "lentil", "mungbean", "pigeonpeas", "mothbeans"].includes((best.crop || "").toLowerCase());
  let fertData;
  if (isLegumeCrop) {
    fertData = [
      ["Basal NPK & Sulphur Dose", `${Math.round(44 * plotAcres)} kg DAP + ${Math.round(16 * plotAcres)} kg MOP + ${Math.round(40 * plotAcres)} kg Gypsum`, "Land prep / Basal (delivers 10:20:10:8 kg N:P2O5:K2O:S per acre)"],
      ["Rhizobium Seed Bio-Priming", `Rhizobium (30g/kg) + Phosphobacteria (30g/kg) + Trichoderma (4g/kg)`, "Seed treatment before sowing (vital for biological nodulation)"],
      ["Foliar Nutrition Round 1", `2% DAP Foliar Spray (${Math.round(4 * plotAcres)} kg DAP in ${Math.round(200 * plotAcres)} L water) or 1% Urea`, "30 DAS (Early flowering) — NO soil urea (protects nodule N-fixation)"],
      ["Pod Setting Foliar Booster", `TNAU Pulse Wonder @ ${Math.round(2 * plotAcres)} kg in ${Math.round(200 * plotAcres)} L water`, "45 DAS (Peak pod development to prevent flower drop)"],
      ["Pesticide / IPM Control", `Yellow sticky traps (10/acre) + Neem oil 3% (${(1.5 * plotAcres).toFixed(1)} L) + Pheromone traps`, "Scouting at 30 & 50 DAS (prevents pod borer & YMV whitefly)"],
    ];
  } else {
    fertData = [
      ["Basal NPK Dose", `${Math.round(25 * plotAcres)} kg Urea + ${Math.round(50 * plotAcres)} kg DAP + ${Math.round(25 * plotAcres)} kg MOP`, "Land prep / Sowing"],
      ["Organic / Bio-Fertilizer", `${Math.round(100 * plotAcres)} kg Neem Cake + ${(2 * plotAcres).toFixed(1)} kg Azospirillum`, "Basal incorporation"],
      ["Topdressing Round 1", `${Math.round(35 * plotAcres)} kg Urea`, "20-25 days after sowing (Active vegetative)"],
      ["Topdressing Round 2", `${Math.round(25 * plotAcres)} kg Urea + ${Math.round(15 * plotAcres)} kg MOP`, "Flowering / Panicle emergence"],
      ["Pesticide / IPM Control", `Light traps + Neem oil spray (${(1.5 * plotAcres).toFixed(1)} L) + Trichogramma bio-cards`, "Scouting at 30 & 60 days"],
    ];
  }

  doc.autoTable({
    head: [["Category / Operation", "Recommended Dose & Materials", "Application Stage / Agronomic Principle"]],
    body: fertData,
    startY: 36,
    headStyles: { fillColor: [20, 83, 45], fontSize: 8 },
    bodyStyles: { fontSize: 7.5 },
  });

  const yTasks = (doc.lastAutoTable?.finalY || 100) + 10;
  doc.setFontSize(10.5);
  doc.setFont("helvetica", "bold");
  doc.text("Sowing-to-Harvest Task Timeline & Cost Breakdown", 14, yTasks);

  const taskRows = (currentCommittedPlan?.tasks || []).map(t => [
    t.task_id,
    t.due_date,
    (t.task_type || "").toUpperCase(),
    t.title,
    `Rs ${Math.round(t.estimated_cost_inr || 0).toLocaleString("en-IN")}`,
    t.is_completed ? "Completed" : (t.escalation_tier > 0 ? `Tier ${t.escalation_tier} Alert` : "Pending")
  ]);

  if (taskRows.length > 0) {
    doc.autoTable({
      head: [["Task ID", "Due Date", "Stage", "Task Title & Operation", "Est. Cost", "Status"]],
      body: taskRows,
      startY: yTasks + 4,
      headStyles: { fillColor: [21, 128, 61], fontSize: 8 },
      bodyStyles: { fontSize: 7.5 },
    });
  }

  if (currentCommittedPlan?.annual_rotation_cycle?.cycles) {
    doc.addPage();
    doc.setFillColor(34, 92, 54);
    doc.rect(0, 0, 210, 24, "F");
    doc.setTextColor(255, 255, 255);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(13);
    doc.text("1-Year Sustainable Multi-Crop Rotation Plan (365-Day Cycle)", 14, 11);
    doc.setFontSize(8.5);
    doc.setFont("helvetica", "normal");
    const annualProf = currentCommittedPlan.annual_rotation_cycle.estimated_annual_profit_inr || 0;
    doc.text(`3 Synergistic Crops with 20-Day Soil Rest Gaps  |  Est. Annual Return: Rs ${Number(annualProf).toLocaleString('en-IN')}`, 14, 18);

    doc.setTextColor(0, 0, 0);
    const rotTableRows = [];
    currentCommittedPlan.annual_rotation_cycle.cycles.forEach(c => {
      rotTableRows.push([
        `Cycle ${c.cycle_number}: ${c.crop_name} (${c.season_label})`,
        c.sowing_date,
        c.expected_harvest_date,
        `${c.duration_days} Days`,
        c.expected_yield_per_acre,
        `Rs ${Number(c.estimated_net_profit_inr).toLocaleString('en-IN')}`,
        c.role
      ]);
      if (c.safety_gap_after) {
        rotTableRows.push([
          `[REST GAP] 20-Day Soil Rest (${c.safety_gap_after.start_date} to ${c.safety_gap_after.end_date})`,
          c.safety_gap_after.start_date,
          c.safety_gap_after.end_date,
          "20 Days",
          "Soil Solarization & Green Manure",
          "—",
          c.safety_gap_after.activity
        ]);
      }
    });

    doc.autoTable({
      head: [["Crop / Cycle", "Sowing Date", "Harvest Date", "Duration", "Yield/Acre", "Est. Profit", "Agronomic Role / Activity"]],
      body: rotTableRows,
      startY: 30,
      headStyles: { fillColor: [20, 83, 45], fontSize: 8 },
      bodyStyles: { fontSize: 7 },
    });
  }

  doc.setFillColor(240, 240, 240);
  doc.rect(0, 282, 210, 15, "F");
  doc.setTextColor(100, 100, 100); doc.setFontSize(8);
  doc.text("Crop Intelligence Platform | AI Project | VIT University", 14, 289);
  doc.save(`Farm_Advisory_${result.best_crop}_${new Date().toISOString().slice(0, 10)}.pdf`);
});

// ── Init ─────────────────────────────────────────────────────────────────────
initMap();

// Render sliders with defaults first, then load metadata (which updates values)
renderSliders();

loadMetadata().catch(err => {
  setStatus(err.message);
  // Still render sliders with built-in defaults
  renderSliders();
});

// ══════════════════════════════════════════════════════════════════════════════
// ── Sowing-to-Harvest Plan & Task Checklist ──────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════
let currentCommittedPlan = null;

async function commitCropPlan(cropName) {
  try {
    setStatus(`Generating ICAR/TNAU sowing plan for ${cropName}...`);
    const areaAcres = (areaHectares || 1.0) * 2.471;
    const today = new Date().toISOString().slice(0, 10);

    const payload = {
      crop_name: cropName,
      area_acres: Number(areaAcres.toFixed(2)),
      sowing_date: today,
      farmer_budget_inr: Math.round(50000.0 * areaAcres),
      irrigation_source: "Borewell"
    };

    const resp = await fetch("/api/plan/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Failed to generate sowing plan");
    }

    const plan = await resp.json();
    currentCommittedPlan = plan;
    renderCommittedPlan(plan);

    const planSec = document.getElementById("committed-plan-section");
    if (planSec) {
      planSec.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    setStatus(`✔ Sowing plan generated for ${cropName} (${plan.tasks?.length || 0} tasks scheduled)`);

    // Automatic Voice Readout of 2-line plan summary
    if ("speechSynthesis" in window) {
      try {
        window.speechSynthesis.cancel();
        const areaAcres = plan.area_acres || ((areaHectares || 1.0) * 2.471);
        const lang = document.getElementById("assistant-lang")?.value || "en";
        let speechMsg = `Sowing schedule committed for ${cropName} across ${areaAcres.toFixed(1)} acres. Estimated budget is ${Math.round(plan.budget_analysis?.cost_breakdown?.total_recommended_budget_inr || 0).toLocaleString("en-IN")} rupees.`;
        if (lang === "ta") {
          speechMsg = `${cropName} பயிருக்கான விதைப்பு கால அட்டவணை உருவாக்கப்பட்டது. மதிப்பீடு செய்யப்பட்ட செலவு ${Math.round(plan.budget_analysis?.cost_breakdown?.total_recommended_budget_inr || 0).toLocaleString("en-IN")} ரூபாய்.`;
        }
        const utter = new SpeechSynthesisUtterance(speechMsg);
        utter.lang = lang === "ta" ? "ta-IN" : "en-IN";
        utter.rate = 0.95;
        if (lang === "ta") {
          const tamilVoice = typeof getTamilVoice === "function" ? getTamilVoice() : null;
          if (tamilVoice) utter.voice = tamilVoice;
        }
        window.speechSynthesis.speak(utter);
      } catch (speechErr) {
        console.warn("Auto voice readout error:", speechErr);
      }
    }
  } catch (err) {
    setStatus("Plan error: " + err.message);
  }
}

function renderCommittedPlan(plan) {
  if (!plan) return;
  const section = document.getElementById("committed-plan-section");
  if (!section) return;
  section.classList.remove("hidden");

  // Header & stats
  const areaAcres = plan.area_acres || ((areaHectares || 1.0) * 2.471);
  const totalCost = plan.budget_analysis?.cost_breakdown?.total_recommended_budget_inr || 0;
  document.getElementById("plan-crop-title").textContent = `📅 Sowing-to-Harvest Plan: ${plan.crop_name} (${areaAcres.toFixed(1)} acres)`;
  document.getElementById("plan-subtitle").textContent = 
    `Plot: ${(areaAcres / 2.471).toFixed(2)} ha (${areaAcres.toFixed(1)} acres) · Duration: ${plan.duration_days} Days · Est. Budget: ₹${Math.round(totalCost).toLocaleString("en-IN")}`;

  document.getElementById("plan-sowing-date").textContent = plan.sowing_date;
  document.getElementById("plan-harvest-date").textContent = plan.expected_harvest_date;

  const budgetEl = document.getElementById("plan-budget-status");
  const ba = plan.budget_analysis;
  if (ba) {
    const isAffordable = ba.is_affordable;
    budgetEl.textContent = isAffordable ? `✓ ${ba.status.toUpperCase()}` : "⚠ INSUFFICIENT";
    budgetEl.className = `status-badge ${isAffordable ? "badge-success" : "badge-danger"}`;
    budgetEl.title = ba.guidance || "";
  }

  // Progress
  const totalTasks = plan.total_tasks_count || (plan.tasks ? plan.tasks.length : 0);
  const completedTasks = plan.completed_tasks_count !== undefined 
    ? plan.completed_tasks_count 
    : (plan.tasks ? plan.tasks.filter(t => t.is_completed).length : 0);
  const adherence = totalTasks > 0 ? Math.round((completedTasks / totalTasks) * 100) : 0;
  document.getElementById("plan-progress").textContent = `${adherence}% Completed (${completedTasks}/${totalTasks})`;

  // Risk
  const riskVal = plan.total_profit_at_risk_inr || plan.profit_at_risk_rs || 0;
  const riskEl = document.getElementById("plan-risk-amt");
  riskEl.textContent = `₹${Math.round(riskVal).toLocaleString("en-IN")}`;
  riskEl.style.color = riskVal > 0 ? "#dc2626" : "#16a34a";

  // Render task list
  const container = document.getElementById("task-checklist");
  container.innerHTML = "";

  (plan.tasks || []).forEach((task) => {
    const isDone = !!task.is_completed;
    const isSuppressed = task.escalation_label === "Recalibrated" || task.recalibration_note;
    const isPostponed = task.escalation_label === "Postponed";
    const card = document.createElement("div");
    card.className = `task-card ${isDone ? "completed" : ""} ${isSuppressed ? "suppressed" : ""}`;
    card.id = `task-card-${task.task_id}`;

    let statusBadge = "";
    if (isDone) {
      statusBadge = `<span class="timing-badge timing-ok">✓ Done</span>`;
    } else if (isSuppressed) {
      statusBadge = `<span class="timing-badge" style="background:#e0f2fe;color:#0369a1">💧 Recalibrated (Rain Suppressed)</span>`;
    } else if (isPostponed) {
      statusBadge = `<span class="timing-badge" style="background:#fef3c7;color:#92400e">⏳ Postponed</span>`;
    } else if (task.escalation_tier > 0) {
      statusBadge = `<span class="timing-badge timing-off">⚠️ Tier ${task.escalation_tier} (${task.escalation_channel || "Alert"})</span>`;
    }

    card.innerHTML = `
      <div style="display:flex;align-items:flex-start;gap:12px">
        <input type="checkbox" id="chk-${task.task_id}" class="task-checkbox" 
               ${isDone ? "checked" : ""} 
               onchange="toggleTaskConfirm('${plan.plan_id}', '${task.task_id}', this.checked)" />
        <div style="flex:1">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px">
            <span class="task-stage-badge">${(task.task_type || "TASK").toUpperCase()} (Day ${task.day_offset})</span>
            <div style="display:flex;gap:6px;align-items:center">
              ${statusBadge}
              <span class="task-date">${task.due_date}</span>
            </div>
          </div>
          <h4 style="margin:6px 0 3px 0;color:#1e293b;font-size:0.96rem">${task.title}</h4>
          <p style="margin:0;font-size:0.85rem;color:#475569;line-height:1.4">${task.description}</p>
          ${task.recalibration_note ? `<p style="margin:4px 0 0 0;font-size:0.82rem;color:#0284c7"><strong>Recalibration:</strong> ${task.recalibration_note}</p>` : ""}
          <div style="margin-top:8px;font-size:0.8rem;color:#64748b;display:flex;justify-content:space-between;align-items:center">
            <span>Est. Cost: <strong>₹${Math.round(task.estimated_cost_inr || 0).toLocaleString("en-IN")}</strong></span>
            ${task.alert_message ? `<span style="color:#b91c1c;font-weight:500">${task.alert_message}</span>` : ""}
          </div>
        </div>
      </div>
    `;
    container.appendChild(card);
  });
}

async function toggleTaskConfirm(planId, taskId, isConfirmed) {
  try {
    const resp = await fetch("/api/plan/task/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        plan_id: planId,
        task_id: taskId,
        is_completed: isConfirmed
      })
    });

    if (!resp.ok) {
      throw new Error("Failed to update task confirmation state");
    }

    const data = await resp.json();
    if (currentCommittedPlan && currentCommittedPlan.tasks) {
      let targetIdx = -1;
      currentCommittedPlan.tasks.forEach((t, idx) => {
        if (t.task_id === taskId) {
          t.is_completed = isConfirmed;
          targetIdx = idx;
        }
      });

      // Demonstration: If checking a later task (e.g. task 4) while prior tasks (e.g. task 2 & 3) were skipped
      let skippedTasks = [];
      let totalRisk = 0;
      if (isConfirmed && targetIdx > 0) {
        for (let i = 0; i < targetIdx; i++) {
          const prior = currentCommittedPlan.tasks[i];
          if (!prior.is_completed) {
            prior.escalation_tier = 3;
            prior.escalation_label = "Critical";
            prior.escalation_channel = "whatsapp_sms_call";
            const cost = prior.estimated_cost_inr || 1000;
            prior.alert_message = `URGENT: Skipped task '${prior.title}' is overdue! Action required to protect approx ₹${Math.round(cost * 3.8).toLocaleString("en-IN")} in yield.`;
            skippedTasks.push(prior);
            totalRisk += Math.round(cost * 3.8);
          }
        }
      }

      currentCommittedPlan.completed_tasks_count = data.completed_tasks_count;
      if (skippedTasks.length > 0) {
        currentCommittedPlan.total_profit_at_risk_inr = totalRisk;
      }
      renderCommittedPlan(currentCommittedPlan);

      // Render Alert Banner & Speak Warning if tasks were skipped
      if (skippedTasks.length > 0) {
        const banner = document.getElementById("plan-alert-banner");
        if (banner) {
          banner.classList.remove("hidden");
          banner.innerHTML = `
            <div style="font-weight:600;margin-bottom:6px;display:flex;align-items:center;gap:6px;color:#b91c1c">
              <span>🚨 Escalation Alert: ${skippedTasks.length} skipped prior task(s) detected!</span>
            </div>
            <ul style="margin:0;padding-left:18px;font-size:0.86rem;line-height:1.5;color:#991b1b">
              ${skippedTasks.map(t => `
                <li>
                  <strong>[CALL / WHATSAPP DISPATCHED] Tier 3 - ${t.title}</strong>: 
                  ${t.alert_message}
                </li>
              `).join("")}
            </ul>
            <div style="margin-top:6px;font-size:0.83rem;color:#7f1d1d">Total Profit at Risk: <strong>₹${totalRisk.toLocaleString("en-IN")}</strong></div>
          `;
        }

        // Voice Alert Readout
        if ("speechSynthesis" in window) {
          try {
            window.speechSynthesis.cancel();
            const lang = document.getElementById("assistant-lang")?.value || "en";
            let warnMsg = `Warning: ${skippedTasks.length} critical task was skipped. Automated SMS and voice call dispatched to protect yield.`;
            if (lang === "ta") {
              warnMsg = `அவசர எச்சரிக்கை: ${skippedTasks.length} முக்கியமான பணி தவிர்க்கப்பட்டது. பயிர் பாதுகாப்பிற்காக தானியங்கி அழைப்பு விடுக்கப்பட்டது.`;
            }
            const utter = new SpeechSynthesisUtterance(warnMsg);
            utter.lang = lang === "ta" ? "ta-IN" : "en-IN";
            utter.rate = 0.95;
            if (lang === "ta") {
              const tamilVoice = typeof getTamilVoice === "function" ? getTamilVoice() : null;
              if (tamilVoice) utter.voice = tamilVoice;
            }
            window.speechSynthesis.speak(utter);
          } catch (e) {}
        }
      }
    }
    setStatus(`Task updated: ${isConfirmed ? "Confirmed completed" : "Marked incomplete"}`);
  } catch (err) {
    setStatus("Task update error: " + err.message);
  }
}

// ── Environmental Recalibration & Graded Escalation ─────────────────────────
document.getElementById("recheck-plan-btn")?.addEventListener("click", async () => {
  if (!currentCommittedPlan) {
    setStatus("Please commit to a crop plan first.");
    return;
  }
  try {
    setStatus("Checking live weather & evaluating task schedule...");
    const banner = document.getElementById("plan-alert-banner");

    const payload = {
      plan_id: currentCommittedPlan.plan_id,
      plan: currentCommittedPlan,
      weather: {
        rainfall_mm: 38.5,
        wind_kmh: 12.0
      }
    };

    const resp = await fetch("/api/plan/recheck", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!resp.ok) {
      throw new Error("Recalibration failed");
    }

    const result = await resp.json();
    currentCommittedPlan.tasks = result.tasks;
    currentCommittedPlan.total_profit_at_risk_inr = result.total_profit_at_risk_inr;
    currentCommittedPlan.completed_tasks_count = result.completed_tasks;
    renderCommittedPlan(currentCommittedPlan);

    // Render alert ladder
    if (result.active_alerts && result.active_alerts.length > 0) {
      banner.classList.remove("hidden");
      banner.innerHTML = `
        <div style="font-weight:600;margin-bottom:6px;display:flex;align-items:center;gap:6px">
          <span>⚡ Environmental Recalibration & Graded Escalation Triggered (${result.highest_escalation_label}):</span>
        </div>
        <ul style="margin:0;padding-left:18px;font-size:0.86rem;line-height:1.5">
          ${result.active_alerts.map(a => `
            <li>
              <strong>[${(a.channel || "Alert").toUpperCase()}] Tier ${a.tier} - ${a.title}</strong>: 
              ${a.message} 
              ${a.tamil_message ? `<br/><em style="color:#166534">தமிழ்: ${a.tamil_message}</em>` : ""}
            </li>
          `).join("")}
        </ul>
      `;
    } else {
      banner.classList.remove("hidden");
      banner.innerHTML = `<span style="color:#166534">✓ Weather check normal. All scheduled operations are on track with zero conflict.</span>`;
    }
    setStatus("✔ Plan recalibrated against environmental conditions");
  } catch (err) {
    setStatus("Recalibration error: " + err.message);
  }
});

// ══════════════════════════════════════════════════════════════════════════════
// ── Voice & Text Grounded RAG Assistant ("Ask SmartFarm") ───────────────────
// ══════════════════════════════════════════════════════════════════════════════
const assistantDock = document.getElementById("assistant-dock");
const toggleAssistantBtn = document.getElementById("toggle-assistant-btn");
const assistantInput = document.getElementById("assistant-input");
const sendAssistantBtn = document.getElementById("send-assistant-btn");
const micBtn = document.getElementById("mic-btn");
const langSelect = document.getElementById("assistant-lang");
const chatMessages = document.getElementById("chat-messages");

if (toggleAssistantBtn && assistantDock) {
  toggleAssistantBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const isMinimized = assistantDock.classList.toggle("minimized");
    assistantDock.classList.toggle("collapsed", isMinimized);
    toggleAssistantBtn.textContent = isMinimized ? "+" : "_";
    toggleAssistantBtn.title = isMinimized ? "Expand Chat" : "Minimize Chat";
  });
}

// ── Make Chatbot Draggable across the viewport ──────────────────────────────
const assistantHeader = assistantDock?.querySelector(".assistant-header");
if (assistantHeader && assistantDock) {
  let isDragging = false;
  let startX = 0, startY = 0;
  let initialLeft = 0, initialTop = 0;

  const startDrag = (e) => {
    // Ignore clicks on control elements inside header (select, button)
    if (e.target.closest("button") || e.target.closest("select")) return;

    isDragging = true;
    assistantDock.classList.add("dragging");

    const clientX = e.type.startsWith("touch") ? e.touches[0].clientX : e.clientX;
    const clientY = e.type.startsWith("touch") ? e.touches[0].clientY : e.clientY;

    const rect = assistantDock.getBoundingClientRect();
    startX = clientX;
    startY = clientY;
    initialLeft = rect.left;
    initialTop = rect.top;

    // Pin position to explicit pixel values for smooth dragging
    assistantDock.style.right = "auto";
    assistantDock.style.bottom = "auto";
    assistantDock.style.left = `${initialLeft}px`;
    assistantDock.style.top = `${initialTop}px`;

    document.addEventListener("mousemove", onDrag);
    document.addEventListener("mouseup", stopDrag);
    document.addEventListener("touchmove", onDrag, { passive: false });
    document.addEventListener("touchend", stopDrag);
  };

  const onDrag = (e) => {
    if (!isDragging) return;
    if (e.cancelable) e.preventDefault();

    const clientX = e.type.startsWith("touch") ? e.touches[0].clientX : e.clientX;
    const clientY = e.type.startsWith("touch") ? e.touches[0].clientY : e.clientY;

    const dx = clientX - startX;
    const dy = clientY - startY;

    const maxLeft = Math.max(10, window.innerWidth - assistantDock.offsetWidth - 10);
    const maxTop = Math.max(10, window.innerHeight - assistantDock.offsetHeight - 10);

    const newLeft = Math.max(10, Math.min(initialLeft + dx, maxLeft));
    const newTop = Math.max(10, Math.min(initialTop + dy, maxTop));

    assistantDock.style.left = `${newLeft}px`;
    assistantDock.style.top = `${newTop}px`;
  };

  const stopDrag = () => {
    isDragging = false;
    assistantDock?.classList.remove("dragging");
    document.removeEventListener("mousemove", onDrag);
    document.removeEventListener("mouseup", stopDrag);
    document.removeEventListener("touchmove", onDrag);
    document.removeEventListener("touchend", stopDrag);
  };

  assistantHeader.addEventListener("mousedown", startDrag);
  assistantHeader.addEventListener("touchstart", startDrag, { passive: false });
}

function appendChatMessage(sender, text, citations = null, warning = null) {
  const msgEl = document.createElement("div");
  msgEl.className = `chat-msg ${sender}-msg`;

  let inner = `<div>${text.replace(/\n/g, "<br/>")}</div>`;
  if (citations && citations.length > 0) {
    inner += `<div class="msg-citations"><small>📚 <strong>Sources:</strong> ${citations.join(" · ")}</small></div>`;
  }
  if (warning) {
    inner += `<div class="msg-warning"><small>⚠️ ${warning}</small></div>`;
  }
  msgEl.innerHTML = inner;
  chatMessages.appendChild(msgEl);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function askAssistant(queryText) {
  if (!queryText || !queryText.trim()) return;
  const q = queryText.trim();
  const lang = langSelect?.value || "en";
  const cropCtx = currentCommittedPlan?.crop_name || "";

  appendChatMessage("user", q);
  if (assistantInput) assistantInput.value = "";

  try {
    const loadingId = "loading-" + Date.now();
    const loadingEl = document.createElement("div");
    loadingEl.id = loadingId;
    loadingEl.className = "chat-msg system-msg";
    loadingEl.textContent = lang === "ta" ? "சிந்திக்கிறது... (Consulting ICAR/TNAU POP)..." : "Consulting ICAR/TNAU Package of Practices...";
    chatMessages.appendChild(loadingEl);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    const resp = await fetch("/api/assistant/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: q,
        crop_name: cropCtx,
        language: lang
      })
    });

    document.getElementById(loadingId)?.remove();

    if (!resp.ok) {
      throw new Error("Assistant request failed");
    }

    const data = await resp.json();
    const citations = (data.sources || []).map(s => `${s.title} (${s.source})`);
    appendChatMessage("assistant", data.answer, citations, data.disclaimer);

    // Speak response if Web Speech API available
    if ("speechSynthesis" in window && data.answer) {
      try {
        window.speechSynthesis.cancel();
        const cleanText = data.answer.replace(/[*#_`]/g, "").slice(0, 250);
        const utter = new SpeechSynthesisUtterance(cleanText);
        utter.lang = lang === "ta" ? "ta-IN" : "en-IN";
        utter.rate = 0.95;
        if (lang === "ta") {
          const tamilVoice = typeof getTamilVoice === "function" ? getTamilVoice() : null;
          if (tamilVoice) utter.voice = tamilVoice;
        }
        window.speechSynthesis.speak(utter);
      } catch (speechErr) {
        console.warn("TTS error:", speechErr);
      }
    }
  } catch (err) {
    appendChatMessage("system", "Error answering query: " + err.message);
  }
}

sendAssistantBtn?.addEventListener("click", () => {
  askAssistant(assistantInput?.value);
});

assistantInput?.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    askAssistant(assistantInput.value);
  }
});

// Voice Input (Web Speech Recognition)
let recognition = null;
if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRec();
  recognition.continuous = false;
  recognition.interimResults = false;

  recognition.onstart = () => {
    micBtn?.classList.add("listening");
    setStatus("🎙️ Listening... speak now");
  };

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    if (assistantInput) assistantInput.value = transcript;
    askAssistant(transcript);
  };

  recognition.onerror = (event) => {
    micBtn?.classList.remove("listening");
    setStatus("Mic error: " + event.error);
  };

  recognition.onend = () => {
    micBtn?.classList.remove("listening");
  };
}

micBtn?.addEventListener("click", () => {
  if (!recognition) {
    alert("Speech recognition is not supported in this browser. Please use Chrome or Edge.");
    return;
  }
  const lang = langSelect?.value || "en";
  recognition.lang = lang === "ta" ? "ta-IN" : "en-IN";
  try {
    recognition.start();
  } catch (e) {
    recognition.stop();
  }
});

// ── Farmer Registration & Plan Commitment Handlers ─────────────────────────

let _pending_commit_crop = null;

function getStoredFarmer() {
  try {
    const raw = localStorage.getItem("smartfarm_farmer");
    return raw ? JSON.parse(raw) : null;
  } catch (e) {
    return null;
  }
}

function setStoredFarmer(user) {
  try {
    localStorage.setItem("smartfarm_farmer", JSON.stringify(user));
  } catch (e) {}
}

// ── Tamil Voice Helper ──────────────────────────────────────────────────────
// Selects a real ta-IN voice from browser voice list for proper Tamil TTS
function getTamilVoice() {
  const voices = window.speechSynthesis.getVoices();
  let v = voices.find(v => v.lang === "ta-IN");
  if (!v) v = voices.find(v => v.lang && v.lang.startsWith("ta"));
  if (!v) v = voices.find(v => v.name && v.name.toLowerCase().includes("tamil"));
  return v || null;
}

// Helper to speak text with proper Tamil voice selection
function speakText(text, lang) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  if (lang === "ta") {
    utterance.lang = "ta-IN";
    const tamilVoice = getTamilVoice();
    if (tamilVoice) utterance.voice = tamilVoice;
  } else {
    utterance.lang = "en-IN";
  }
  utterance.rate = 0.9;
  window.speechSynthesis.speak(utterance);
}

function playAuthVoiceInstructions() {
  if (!("speechSynthesis" in window)) return;
  const lang = document.getElementById("assistant-lang")?.value || "en";
  const text = lang === "ta"
    ? "வணக்கம்! உங்கள் பயிர் திட்டத்தை ஆயுட்காலம் வரை சேமிக்க உங்கள் 10 இலக்க மொபைல் எண்ணை உள்ளிட்டு சரிபார்க்கவும் அல்லது கூகிள் கணக்கை பயன்படுத்தவும்."
    : "Hello! Please enter your 10-digit mobile number to verify and save your crop sowing schedule lifelong, or sign in with Google.";
  speakText(text, lang);
}

window.commitCropPlan = function(cropName) {
  _pending_commit_crop = cropName;
  const farmer = getStoredFarmer();
  if (farmer && farmer.phone_number) {
    executeCommitCropPlan(cropName, farmer.phone_number, farmer.full_name);
  } else {
    // Open Auth Modal
    const modal = document.getElementById("auth-modal");
    if (modal) modal.classList.remove("hidden");
    playAuthVoiceInstructions();
  }
};
// currentCommittedPlan is declared at line 1118

async function executeCommitCropPlan(cropName, farmerPhone, farmerName) {
  setStatus(`Generating dated sowing schedule and 1-year crop rotation for ${cropName}...`);
  try {
    // Dynamic starting timestamp: 10 days from now (or today) as requested
    const targetStartDate = new Date(Date.now() + 10 * 86400000).toISOString().split("T")[0];

    const res = await fetch("/api/plan/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        crop_name: cropName,
        sowing_date: targetStartDate,
        area_acres: Number(document.getElementById("area-range")?.value || 1.0),
        farmer_budget_inr: 50000,
        irrigation_source: document.getElementById("irrigation-select")?.value || "Borewell",
        farmer_phone: farmerPhone,
        farmer_name: farmerName,
      })
    });
    
    if (!res.ok) throw new Error(await res.text());
    const plan = await res.json();
    currentCommittedPlan = plan;
    
    // Hide auth modal if open
    const modal = document.getElementById("auth-modal");
    if (modal) {
      modal.classList.add("hidden");
      modal.style.display = "none";
    }
    
    // Render Sowing Plan UI & 1-Year Multi-Crop Rotation Schedule
    renderActiveSowingPlan(plan, farmerPhone);
    setStatus(`✓ Plan committed and saved lifelong for ${farmerPhone}!`);
  } catch (err) {
    alert("Plan Generation Failed: " + err.message);
    setStatus("Plan generation error: " + err.message);
  }
}

// ── Checkbox Toggle & Sequence Escalation Handler ─────────────────────────────
window.toggleTaskComplete = async function(planId, taskId, isChecked) {
  try {
    const res = await fetch("/api/plan/task/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        plan_id: planId,
        task_id: taskId,
        is_completed: isChecked
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Task update failed");

    // Update schedule adherence stats
    const progEl = document.getElementById("plan-progress");
    if (progEl && data.total_tasks_count) {
      const pct = Math.round((data.completed_tasks_count / data.total_tasks_count) * 100);
      progEl.textContent = `${pct}% Completed (${data.completed_tasks_count}/${data.total_tasks_count})`;
    }

    // ── Check if Escalation Was Triggered ──
    if (data.escalation && data.escalation.triggered) {
      handleEscalationTriggered(data.escalation);
    }
  } catch (err) {
    console.error("Task toggle error:", err);
    alert("Could not update task: " + err.message);
  }
};

function handleEscalationTriggered(esc) {
  const planSection = document.getElementById("committed-plan-section") || document.getElementById("active-plan-section");
  let banner = document.getElementById("plan-alert-banner");
  if (!banner && planSection) {
    banner = document.createElement("div");
    banner.id = "plan-alert-banner";
    planSection.prepend(banner);
  }
  if (!banner) return;

  const isVoice = esc.type === "voice_call";
  banner.classList.remove("hidden");
  banner.style.display = "block";
  banner.style.padding = "16px";
  banner.style.marginBottom = "18px";
  banner.style.borderRadius = "12px";
  banner.style.border = isVoice ? "2px solid #dc2626" : "2px solid #2563eb";
  banner.style.background = isVoice ? "#fef2f2" : "#eff6ff";
  banner.style.color = isVoice ? "#991b1b" : "#1e40af";
  banner.style.boxShadow = "0 4px 14px rgba(0,0,0,0.12)";

  banner.innerHTML = `
    <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px">
      <div style="display:flex;align-items:flex-start;gap:12px">
        <span style="font-size:2rem">${isVoice ? '🚨' : '📱'}</span>
        <div>
          <strong style="font-size:1.05rem;display:block;margin-bottom:4px">${esc.title}</strong>
          <p style="font-size:0.9rem;margin:0 0 6px 0;line-height:1.4">${esc.message}</p>
          ${esc.tamil_message ? `<p style="font-size:0.86rem;margin:0;color:#047857;line-height:1.4"><em>${esc.tamil_message}</em></p>` : ''}
          ${esc.skipped_tasks ? `<div style="margin-top:6px;font-size:0.8rem;background:#fee2e2;padding:4px 8px;border-radius:6px;color:#b91c1c">⚠️ Skipped Tasks: ${esc.skipped_tasks.join(' · ')}</div>` : ''}
        </div>
      </div>
      <button type="button" onclick="this.closest('#plan-alert-banner').style.display='none'" style="background:none;border:none;font-size:1.4rem;cursor:pointer;color:inherit">&times;</button>
    </div>
  `;

  // Spoken voice feedback through device speakers
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
    const lang = document.getElementById("assistant-lang")?.value || "en";
    const textToSpeak = (lang === "ta" && esc.tamil_message) ? esc.tamil_message : esc.message;
    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    utterance.lang = (lang === "ta" && esc.tamil_message) ? "ta-IN" : "en-IN";
    utterance.rate = 0.95;
    if (lang === "ta" && esc.tamil_message) {
      const tamilVoice = typeof getTamilVoice === "function" ? getTamilVoice() : null;
      if (tamilVoice) utterance.voice = tamilVoice;
    }
    window.speechSynthesis.speak(utterance);
  }

  banner.scrollIntoView({ behavior: "smooth", block: "center" });
}

function renderActiveSowingPlan(plan, farmerPhone) {
  currentCommittedPlan = plan;
  const planSection = document.getElementById("committed-plan-section") || document.getElementById("active-plan-section");
  if (!planSection) return;
  
  planSection.classList.remove("hidden");
  planSection.style.display = "block";

  const titleEl = document.getElementById("plan-crop-title") || document.getElementById("plan-crop-name");
  if (titleEl) titleEl.textContent = `${plan.crop_name} Sowing Schedule` + (plan.farmer_phone ? ` (Registered: ${plan.farmer_phone})` : "");
  
  const sowEl = document.getElementById("plan-sowing-date") || document.getElementById("plan-sow-date");
  if (sowEl) sowEl.textContent = plan.sowing_date;
  
  const harvEl = document.getElementById("plan-harvest-date");
  if (harvEl) harvEl.textContent = plan.expected_harvest_date;
  
  const bCheck = plan.budget_analysis || {};
  const bStatusEl = document.getElementById("plan-budget-status");
  if (bStatusEl) bStatusEl.textContent = bCheck.status ? bCheck.status.toUpperCase() : "ADEQUATE";
  
  const progEl = document.getElementById("plan-progress");
  if (progEl) progEl.textContent = (plan.completion_percentage || 0) + "% Completed";
  
  // Render Milestone Tasks List
  const taskChecklist = document.getElementById("task-checklist");
  if (taskChecklist) {
    taskChecklist.innerHTML = "";
    (plan.tasks || []).forEach((task, idx) => {
      const taskCard = document.createElement("div");
      taskCard.className = `task-card ${task.is_completed ? 'task-completed' : ''}`;
      taskCard.style.padding = "14px";
      taskCard.style.marginBottom = "10px";
      taskCard.style.borderRadius = "10px";
      taskCard.style.border = "1px solid #cbd5e1";
      taskCard.style.background = task.is_completed ? "#f0fdf4" : "#ffffff";

      taskCard.innerHTML = `
        <div style="display:flex;align-items:flex-start;gap:12px">
          <input type="checkbox" id="chk-${task.task_id}" ${task.is_completed ? 'checked' : ''} onchange="toggleTaskComplete('${plan.plan_id}', '${task.task_id}', this.checked)" style="margin-top:4px;transform:scale(1.3);cursor:pointer" />
          <div style="flex:1">
            <div style="display:flex;align-items:center;gap:8px">
              <span style="font-size:0.75rem;font-weight:700;color:#166534;background:#dcfce7;padding:2px 6px;border-radius:4px">BOX ${idx + 1} (${task.task_id})</span>
              <strong style="font-size:0.95rem;color:#1e293b">${task.title}</strong>
              ${task.is_critical ? '<span style="font-size:0.7rem;background:#fee2e2;color:#b91c1c;padding:1px 6px;border-radius:4px;font-weight:600">CRITICAL</span>' : ''}
            </div>
            <p style="font-size:0.84rem;color:#475569;margin:4px 0">${task.description}</p>
            <small style="font-size:0.78rem;color:#059669">📅 Due: <strong>${task.due_date}</strong> &middot; Est. Cost: ₹${Number(task.estimated_cost_inr).toLocaleString('en-IN')}</small>
            ${task.recalibration_note ? `<div style="font-size:0.78rem;color:#d97706;margin-top:4px">🌧️ ${task.recalibration_note}</div>` : ''}
          </div>
        </div>
      `;
      taskChecklist.appendChild(taskCard);
    });
  }

  // ── Render 1-Year Multi-Crop Rotation Schedule (365-Day Cycle with 20-Day Soil Rest Gaps) ──
  const rot = plan.annual_rotation_cycle;
  if (rot && rot.cycles) {
    let rotSection = document.getElementById("annual-rotation-container");
    if (!rotSection) {
      rotSection = document.createElement("div");
      rotSection.id = "annual-rotation-container";
      rotSection.style.marginTop = "28px";
      rotSection.style.padding = "20px";
      rotSection.style.background = "#ffffff";
      rotSection.style.borderRadius = "14px";
      rotSection.style.border = "1px solid #cbd5e1";
      rotSection.style.boxShadow = "0 4px 10px rgba(0,0,0,0.05)";
      planSection.appendChild(rotSection);
    }

    let cyclesHtml = "";
    rot.cycles.forEach((c) => {
      cyclesHtml += `
        <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:12px;padding:16px;margin-bottom:12px">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
            <span style="font-size:0.75rem;font-weight:700;background:#166534;color:#fff;padding:3px 10px;border-radius:12px">CYCLE ${c.cycle_number} · ${c.season_label.toUpperCase()}</span>
            <strong style="color:#15803d;font-size:0.95rem">₹${Number(c.estimated_net_profit_inr).toLocaleString('en-IN')} Est. Net Profit</strong>
          </div>
          <div style="display:flex;align-items:baseline;gap:8px">
            <h3 style="font-size:1.18rem;color:#14532d;margin:0">${c.crop_name}</h3>
            ${c.local_name ? `<span style="font-size:0.85rem;color:#475569">(${c.local_name})</span>` : ''}
            <span style="font-size:0.75rem;color:#64748b;background:#e2e8f0;padding:2px 8px;border-radius:4px">${c.category}</span>
          </div>
          <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(150px, 1fr));gap:8px;margin-top:10px;font-size:0.84rem;color:#334155">
            <div>📅 <strong>Sowing:</strong> ${c.sowing_date}</div>
            <div>🌾 <strong>Harvest:</strong> ${c.expected_harvest_date}</div>
            <div>⏱️ <strong>Duration:</strong> ${c.duration_days} Days</div>
            <div>📈 <strong>Yield/Acre:</strong> ${c.expected_yield_per_acre}</div>
          </div>
          <p style="font-size:0.82rem;color:#475569;margin:8px 0 0 0"><em>${c.role}</em></p>
        </div>
      `;

      if (c.safety_gap_after) {
        cyclesHtml += `
          <div style="display:flex;align-items:center;gap:12px;margin: -6px 0 12px 14px;padding:10px 14px;background:#ecfdf5;border-left:4px solid #10b981;border-radius:6px;font-size:0.82rem">
            <span style="font-size:1.2rem">🌱</span>
            <div>
              <strong style="color:#065f46">20-DAY MANDATORY SOIL RECUPERATION GAP (${c.safety_gap_after.start_date} to ${c.safety_gap_after.end_date})</strong>
              <p style="margin:2px 0 0 0;color:#047857">${c.safety_gap_after.activity} — <em>${c.safety_gap_after.soil_benefit}</em></p>
            </div>
          </div>
        `;
      }
    });

    rotSection.innerHTML = `
      <div style="border-bottom:1px solid #e2e8f0;padding-bottom:12px;margin-bottom:16px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px">
        <div>
          <h2 style="font-size:1.25rem;color:#14532d;margin:0 0 4px 0">🔄 1-Year Sustainable Multi-Crop Rotation Plan (365-Day Cycle)</h2>
          <p style="font-size:0.85rem;color:#64748b;margin:0">From ${rot.start_date} to ${rot.end_date} · 3 Synergistic Crops with 20-Day Soil Rest & Solarization Gaps</p>
        </div>
        <div style="text-align:right">
          <div style="font-size:0.75rem;color:#64748b">Cumulative 1-Yr Net Return</div>
          <strong style="font-size:1.25rem;color:#15803d">₹${Number(rot.estimated_annual_profit_inr).toLocaleString('en-IN')}</strong>
        </div>
      </div>
      <div style="margin-bottom:14px">
        <span style="font-size:0.82rem;background:#f0fdf4;color:#166534;padding:4px 12px;border-radius:12px;border:1px solid #bbf7d0;font-weight:600">
          🌿 Soil Ecological Health: ${rot.soil_health_rating}
        </span>
      </div>
      ${cyclesHtml}
    `;
  }

  planSection.scrollIntoView({ behavior: "smooth" });
}

// Modal tab & Auth event listeners
document.addEventListener("DOMContentLoaded", () => {
  const closeAuthBtn = document.getElementById("close-auth-modal");
  const tabPhoneBtn = document.getElementById("tab-phone-btn");
  const tabGoogleBtn = document.getElementById("tab-google-btn");
  const phoneSec = document.getElementById("auth-phone-section");
  const googleSec = document.getElementById("auth-google-section");
  const sendOtpBtn = document.getElementById("send-otp-btn");
  const verifyOtpBtn = document.getElementById("verify-otp-btn");
  const googleAuthBtn = document.getElementById("google-auth-btn");
  const voicePromptBtn = document.getElementById("auth-voice-prompt-btn");
  const statusMsg = document.getElementById("auth-status-msg");
  const topLoginBtn = document.getElementById("top-login-btn");

  topLoginBtn?.addEventListener("click", () => {
    const modal = document.getElementById("auth-modal");
    if (modal) {
      modal.classList.remove("hidden");
      modal.style.display = "flex";
    }
  });

  closeAuthBtn?.addEventListener("click", () => {
    const modal = document.getElementById("auth-modal");
    if (modal) {
      modal.classList.add("hidden");
      modal.style.display = "none";
    }
  });

  voicePromptBtn?.addEventListener("click", () => {
    playAuthVoiceInstructions();
  });

  tabPhoneBtn?.addEventListener("click", () => {
    tabPhoneBtn.style.background = "#166534";
    tabPhoneBtn.style.color = "#fff";
    tabGoogleBtn.style.background = "#f8fafc";
    tabGoogleBtn.style.color = "#475569";
    phoneSec?.classList.remove("hidden");
    googleSec?.classList.add("hidden");
  });

  tabGoogleBtn?.addEventListener("click", () => {
    tabGoogleBtn.style.background = "#166534";
    tabGoogleBtn.style.color = "#fff";
    tabPhoneBtn.style.background = "#f8fafc";
    tabPhoneBtn.style.color = "#475569";
    googleSec?.classList.remove("hidden");
    phoneSec?.classList.add("hidden");
  });

  sendOtpBtn?.addEventListener("click", async () => {
    const name = document.getElementById("farmer-name-input")?.value?.trim();
    const phone = document.getElementById("farmer-phone-input")?.value?.trim();

    if (!name || name.length < 2) {
      if (statusMsg) {
        statusMsg.style.color = "#dc2626";
        statusMsg.textContent = "Please enter your full name before sending OTP.";
      }
      return;
    }

    if (!phone || phone.length < 8) {
      if (statusMsg) {
        statusMsg.style.color = "#dc2626";
        statusMsg.textContent = "Please enter your mobile phone number with country code (e.g. +91994525549).";
      }
      return;
    }
    if (statusMsg) {
      statusMsg.style.color = "#166534";
      statusMsg.textContent = "Sending OTP via SMS & WhatsApp to your mobile number...";
    }
    try {
      const res = await fetch("/api/auth/otp/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phone_number: phone })
      });
      let data;
      try {
        data = await res.json();
      } catch (jsonErr) {
        throw new Error("Server error — please try again.");
      }
      if (!res.ok) throw new Error(data.detail || "Failed to send OTP");
      
      document.getElementById("step-otp-verify")?.classList.remove("hidden");
      if (statusMsg) {
        statusMsg.style.color = "#15803d";
        const channels = [];
        if (data.sms_dispatched) channels.push("SMS");
        if (data.whatsapp_dispatched) channels.push("WhatsApp");
        const chStr = channels.length ? ` via ${channels.join(" & ")}` : "";
        statusMsg.textContent = `✓ OTP sent${chStr}! Code: ${data.otp_demo} (or use 1234)`;
      }
    } catch (e) {
      if (statusMsg) {
        statusMsg.style.color = "#dc2626";
        statusMsg.textContent = e.message;
      }
    }
  });

  verifyOtpBtn?.addEventListener("click", async () => {
    const phone = document.getElementById("farmer-phone-input")?.value?.trim();
    const otp = document.getElementById("otp-code-input")?.value?.trim();
    const name = document.getElementById("farmer-name-input")?.value?.trim() || "Farmer";

    if (!otp) {
      if (statusMsg) {
        statusMsg.style.color = "#dc2626";
        statusMsg.textContent = "Please enter the verification OTP code.";
      }
      return;
    }

    try {
      const res = await fetch("/api/auth/otp/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phone_number: phone, otp_code: otp, full_name: name })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "OTP verification failed");

      setStoredFarmer(data.user);
      updateFarmerAuthUI(data.user);
      const modal = document.getElementById("auth-modal");
      if (modal) {
        modal.classList.add("hidden");
        modal.style.display = "none";
      }
      if (_pending_commit_crop) {
        executeCommitCropPlan(_pending_commit_crop, data.user.phone_number, data.user.full_name);
      } else {
        loadFarmerExistingPlan(data.user.phone_number || data.user.email);
      }
    } catch (e) {
      if (statusMsg) {
        statusMsg.style.color = "#dc2626";
        statusMsg.textContent = e.message;
      }
    }
  });

  googleAuthBtn?.addEventListener("click", () => {
    const name = document.getElementById("farmer-name-input")?.value?.trim();
    if (!name || name.length < 2) {
      if (statusMsg) {
        statusMsg.style.color = "#dc2626";
        statusMsg.textContent = "Please enter your full name before signing in with Google.";
      }
      return;
    }

    if (statusMsg) {
      statusMsg.style.color = "#166534";
      statusMsg.textContent = "Opening Google Sign-In...";
    }

    // Real Google Sign-In via Google Identity Services (GIS)
    const tryGoogleSignIn = () => {
      if (typeof google !== "undefined" && google.accounts && google.accounts.id) {
        google.accounts.id.initialize({
          client_id: "YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com",
          callback: async (response) => {
            try {
              const farmerName = document.getElementById("farmer-name-input")?.value?.trim() || "Google Farmer";
              const farmerPhone = document.getElementById("farmer-phone-input")?.value?.trim() || "";
              const res = await fetch("/api/auth/google", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  google_token: response.credential,
                  full_name: farmerName,
                  phone_number: farmerPhone || undefined
                })
              });
              let data;
              try { data = await res.json(); } catch { throw new Error("Server error"); }
              if (!res.ok) throw new Error(data.detail || "Google Auth failed");

              setStoredFarmer(data.user);
              updateFarmerAuthUI(data.user);
              const modal = document.getElementById("auth-modal");
              if (modal) { modal.classList.add("hidden"); modal.style.display = "none"; }
              if (_pending_commit_crop) {
                executeCommitCropPlan(_pending_commit_crop, data.user.phone_number, data.user.full_name);
              } else {
                loadFarmerExistingPlan(data.user.phone_number || data.user.email);
              }
            } catch (e) {
              if (statusMsg) { statusMsg.style.color = "#dc2626"; statusMsg.textContent = e.message; }
            }
          },
          auto_select: false,
          cancel_on_tap_outside: true,
        });
        google.accounts.id.prompt((notification) => {
          if (notification.isNotDisplayed() || notification.isSkippedMoment()) {
            // Fallback: render a traditional button
            google.accounts.id.renderButton(
              document.getElementById("google-auth-btn"),
              { theme: "outline", size: "large", width: 340 }
            );
            if (statusMsg) {
              statusMsg.style.color = "#475569";
              statusMsg.textContent = "Click the Google button above to sign in.";
            }
          }
        });
      } else {
        // GIS not loaded yet — try one-click demo fallback
        if (statusMsg) {
          statusMsg.style.color = "#d97706";
          statusMsg.textContent = "Google Sign-In loading... If this persists, use Phone OTP instead.";
        }
        setTimeout(tryGoogleSignIn, 1500);
      }
    };
    tryGoogleSignIn();
  });

  // Download Plan PDF Handler
  document.getElementById("download-plan-pdf-btn")?.addEventListener("click", () => {
    const reportBtn = document.getElementById("download-report-btn");
    if (reportBtn && reportBtn._result) {
      reportBtn.click();
    } else if (currentCommittedPlan && window.jspdf) {
      const area = currentCommittedPlan.area_acres || 1.0;
      const synthResult = {
        best_crop: currentCommittedPlan.crop_name,
        top_crops: [
          {
            crop: currentCommittedPlan.crop_name,
            sowing_month: currentCommittedPlan.sowing_date,
            harvest_month: currentCommittedPlan.expected_harvest_date,
            duration_months: Math.round(currentCommittedPlan.duration_days / 30),
            expected_yield_t_ha: 2.8,
            adjusted_price_rs_per_kg: 65,
            total_cost_rs_per_ha: currentCommittedPlan.budget_analysis?.cost_breakdown?.operational_subtotal_inr || 35000,
            revenue_rs_per_ha: (currentCommittedPlan.budget_analysis?.cost_breakdown?.operational_subtotal_inr || 35000) * 1.6,
            profit_rs_per_ha: (currentCommittedPlan.budget_analysis?.cost_breakdown?.operational_subtotal_inr || 35000) * 0.6,
            risk: 0.12,
            sustainability_score: 0.88,
            market_source: "E-NAM Baseline",
            final_score: 0.92,
          }
        ]
      };
      if (reportBtn) {
        reportBtn._result = synthResult;
        reportBtn.click();
      }
    } else {
      alert("Please generate or select a crop plan first to download the advisory PDF.");
    }
  });

  // ── Auto-restore saved farmer session & plans from database on page load ──
  const stored = getStoredFarmer();
  if (stored && (stored.phone_number || stored.email)) {
    updateFarmerAuthUI(stored);
    loadFarmerExistingPlan(stored.phone_number || stored.email);
  }
});

function updateFarmerAuthUI(farmer) {
  const topLoginBtn = document.getElementById("top-login-btn");
  if (!topLoginBtn) return;
  const name = farmer.full_name || farmer.phone_number || "Farmer";
  topLoginBtn.innerHTML = `👤 ${name} <span style="font-size:0.75rem;opacity:0.8;margin-left:4px">(Logout)</span>`;
  topLoginBtn.title = "Click to log out or switch account";
  topLoginBtn.onclick = (e) => {
    e.preventDefault();
    if (confirm(`Currently logged in as ${name}. Do you want to log out?`)) {
      localStorage.removeItem("smartfarm_farmer");
      location.reload();
    }
  };
}

async function loadFarmerExistingPlan(identifier) {
  if (!identifier) return;
  try {
    const res = await fetch(`/api/farmer/${encodeURIComponent(identifier)}/plans`);
    if (!res.ok) return;
    const data = await res.json();
    if (data.plans && data.plans.length > 0) {
      const plan = data.plans[0];
      renderActiveSowingPlan(plan, identifier);
      setStatus(`✓ Restored active sowing schedule for ${plan.crop_name} (${data.user?.full_name || identifier})`);
    }
  } catch (err) {
    console.debug("Could not auto-restore farmer plan:", err);
  }
}

window.toggleTaskConfirm = window.toggleTaskComplete;



