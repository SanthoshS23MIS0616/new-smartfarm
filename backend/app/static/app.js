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
  setStatus("Fetching live weather + soil data...");
  mapState.locationLabel = `Selected: ${fmt(lat, 5)}, ${fmt(lng, 5)}`;
  locationBanner.textContent = mapState.locationLabel;

  // Run weather (Open-Meteo) and soil (SoilGrids) in parallel
  const [, soilResult] = await Promise.allSettled([
    fetchWeather(lat, lng),
    fetchSoilData(lat, lng),
  ]);

  // ── Auto-fill ONLY N and pH from SoilGrids ─────────────────────────────
  // P and K are NOT filled — SoilGrids does not measure them accurately.
  // User must enter P and K from their Soil Health Card or lab report.
  const soil = soilResult?.value;
  if (soil) {
    if (soil.nitrogen != null) setSliderValue("nitrogen", Math.round(soil.nitrogen));
    if (soil.ph       != null) setSliderValue("ph",       parseFloat(soil.ph.toFixed(1)));
    showSoilBadge(
      `🛰 Auto-filled: N=${soil.nitrogen != null ? Math.round(soil.nitrogen) : "?"} kg/ha · pH=${soil.ph != null ? soil.ph.toFixed(1) : "?"} (SoilGrids ISRIC)` +
      `  |  ⚠ Enter Phosphorus & Potassium manually from your Soil Health Card`
    );
  } else {
    showSoilBadge("🛰 Soil auto-fill unavailable — enter N, P, K, pH manually from your Soil Health Card");
  }

  setStatus("✅ Location synced — N/pH from SoilGrids · Temperature/Humidity/Rainfall/Moisture from Open-Meteo · Enter P & K from Soil Health Card");
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
      head: [["Crop","Payback(yr)","Yield","Price","Cost/ha","Profit/ha","Sustainability"]],
      body: perennials.map(p => [
        p.crop, p.payback_years, fmt(p.expected_yield_t_ha, 2),
        `₹${fmt(p.stable_price_rs_per_kg, 2)}`, `₹${p.total_cost_rs_per_ha.toFixed(0)}`,
        `₹${p.profit_rs_per_ha.toFixed(0)}`, `${(p.sustainability_score*100).toFixed(1)}%`,
      ]),
      startY: y + 4,
      headStyles: { fillColor: [201, 125, 42], fontSize: 8 },
      bodyStyles: { fontSize: 7.5 },
    });
  }

  if (best.advisory_notes?.length) {
    const y2 = (doc.lastAutoTable?.finalY || 200) + 10;
    if (y2 < 260) {
      doc.setFont("helvetica", "bold"); doc.setFontSize(10);
      doc.text("Advisory Notes", 14, y2);
      doc.setFont("helvetica", "normal"); doc.setFontSize(8.5);
      best.advisory_notes.forEach((n, i) => doc.text(`• ${n}`, 16, y2 + 6 + i * 5.5));
    }
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
