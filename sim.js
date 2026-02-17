// ═══════════════════════════════════════════════════════════════
//  EVOLUTION SIMULATION ENGINE — Main Orchestrator
//  Integrates: NeuralNetwork, Genetics, Ecosystem, Analytics
//  Python ML backend (FastAPI + NumPy + scikit-learn) handles:
//    - Genome creation, GA reproduction, k-means, statistics
//  JS handles: per-frame rendering, NN forward pass, ecosystem
// ═══════════════════════════════════════════════════════════════

(() => {
  "use strict";

  // ── Giant Trait Pool (40 traits) ──────────────────────────
  const TRAIT_POOL = [
    { name: "Heat Resistance",       color: "#ef4444", desc: "Survives extreme temperatures" },
    { name: "Cold Tolerance",        color: "#38bdf8", desc: "Thrives in freezing conditions" },
    { name: "UV Shield",             color: "#fbbf24", desc: "Blocks ultraviolet radiation" },
    { name: "Acid Resistance",       color: "#a3e635", desc: "Survives in acidic environments" },
    { name: "Alkaline Tolerance",    color: "#818cf8", desc: "Thrives in basic pH levels" },
    { name: "Thick Cell Wall",       color: "#f97316", desc: "Extra structural protection" },
    { name: "Rapid Division",        color: "#f472b6", desc: "Doubles faster than peers" },
    { name: "Spore Formation",       color: "#a78bfa", desc: "Can enter dormant state" },
    { name: "Biofilm Producer",      color: "#34d399", desc: "Creates protective colonies" },
    { name: "Antibiotic Resistance", color: "#fb923c", desc: "Survives antibiotic exposure" },
    { name: "Toxin Secretion",       color: "#c084fc", desc: "Produces harmful compounds" },
    { name: "Chemotaxis",            color: "#22d3ee", desc: "Moves toward nutrients" },
    { name: "Photosynthesis",        color: "#4ade80", desc: "Harvests light energy" },
    { name: "Bioluminescence",       color: "#e879f9", desc: "Produces visible light" },
    { name: "Magnetotaxis",          color: "#60a5fa", desc: "Navigates via magnetic fields" },
    { name: "Radiation Resistance",  color: "#fcd34d", desc: "Withstands ionizing radiation" },
    { name: "Desiccation Tolerance", color: "#d4a76a", desc: "Survives dehydration" },
    { name: "Pressure Resistance",   color: "#6ee7b7", desc: "Survives deep-sea pressures" },
    { name: "Oxygen Independence",   color: "#94a3b8", desc: "Lives without oxygen" },
    { name: "Metal Resistance",      color: "#cbd5e1", desc: "Tolerates heavy metals" },
    { name: "Salt Tolerance",        color: "#fda4af", desc: "Thrives in high salinity" },
    { name: "Elastic Membrane",      color: "#86efac", desc: "Flexible cell boundary" },
    { name: "Quorum Sensing",        color: "#c4b5fd", desc: "Communicates with neighbors" },
    { name: "Gene Transfer",         color: "#fdba74", desc: "Shares DNA with others" },
    { name: "Camouflage",            color: "#475569", desc: "Evades immune detection" },
    { name: "Fast Metabolism",       color: "#f43f5e", desc: "Processes energy quickly" },
    { name: "Slow Metabolism",       color: "#64748b", desc: "Conserves energy reserves" },
    { name: "Enzyme Secretion",      color: "#2dd4bf", desc: "Breaks down complex molecules" },
    { name: "Pili Attachment",       color: "#fb7185", desc: "Clings to surfaces" },
    { name: "Flagella Motility",     color: "#38bdf8", desc: "Swims with flagella" },
    { name: "Capsule Coating",       color: "#a3e635", desc: "Slimy protective layer" },
    { name: "Iron Scavenging",       color: "#d97706", desc: "Extracts iron efficiently" },
    { name: "Nitrogen Fixation",     color: "#059669", desc: "Converts atmospheric nitrogen" },
    { name: "Sulfur Metabolism",     color: "#eab308", desc: "Uses sulfur compounds" },
    { name: "Symbiosis",            color: "#8b5cf6", desc: "Benefits from host organisms" },
    { name: "Predatory Behavior",   color: "#dc2626", desc: "Consumes other bacteria" },
    { name: "DNA Repair",           color: "#14b8a6", desc: "Fixes mutations efficiently" },
    { name: "Membrane Pumps",       color: "#7c3aed", desc: "Expels toxins actively" },
    { name: "Endospore Armor",      color: "#b45309", desc: "Nearly indestructible shell" },
    { name: "Phage Immunity",       color: "#0ea5e9", desc: "Resists viral infection" },
  ];

  // ── Species colours for k-means clusters ──────────────────
  const SPECIES_COLORS = ["#ef4444", "#38bdf8", "#fbbf24", "#34d399", "#a78bfa"];

  // ── Simulation state ──────────────────────────────────────
  let bacteria = [];
  let deadBacteria = [];
  let generation = 0;
  let diedThisRound = 0;
  let selectionHistory = [];
  let simSpeed = 1;
  let paused = false;
  let initialPop = 200;
  let traitsPerBacterium = 4;
  let selecting = false;
  let inspectedBacterium = null;
  let bacteriaIdCounter = 0;

  // GA config (sync'd from sliders)
  let mutationRate = 0.05;
  let mutationSigma = 0.20;
  let elitismPct = 10;
  let crossoverMethod = "uniform";
  let tournamentSize = 3;

  // ── DOM references ────────────────────────────────────────
  const canvas        = document.getElementById("simCanvas");
  const ctx           = canvas.getContext("2d");
  const flashOverlay  = document.getElementById("flashOverlay");
  const selAnnounce   = document.getElementById("selAnnounce");
  const selTraitName  = document.getElementById("selTraitName");
  const selTraitDesc  = document.getElementById("selTraitDesc");
  const pauseOverlay  = document.getElementById("pauseOverlay");
  const btnSelect     = document.getElementById("btnSelect");
  const btnReset      = document.getElementById("btnReset");
  const btnPause      = document.getElementById("btnPause");
  const popSlider     = document.getElementById("popSlider");
  const popVal        = document.getElementById("popVal");
  const traitSlider   = document.getElementById("traitCountSlider");
  const traitCountVal = document.getElementById("traitCountVal");
  const statAlive     = document.getElementById("statAlive");
  const statGen       = document.getElementById("statGen");
  const statDied      = document.getElementById("statDied");
  const statTraits    = document.getElementById("statTraits");
  const statMeanFit   = document.getElementById("statMeanFit");
  const statDiversity = document.getElementById("statDiversity");
  const genBadge      = document.getElementById("genBadge");
  const hsPop         = document.getElementById("hsPop");
  const hsFit         = document.getElementById("hsFit");
  const traitLog      = document.getElementById("traitLog");
  const traitLegend   = document.getElementById("traitLegend");
  const speedBtns     = document.getElementById("speedBtns");
  const hwTable       = document.getElementById("hwTable");

  // GA sliders
  const mutRateSlider   = document.getElementById("mutRateSlider");
  const mutRateVal      = document.getElementById("mutRateVal");
  const mutSigmaSlider  = document.getElementById("mutSigmaSlider");
  const mutSigmaVal     = document.getElementById("mutSigmaVal");
  const elitismSlider   = document.getElementById("elitismSlider");
  const elitismVal      = document.getElementById("elitismVal");
  const crossoverSelect = document.getElementById("crossoverSelect");
  const tournamentSlider= document.getElementById("tournamentSlider");
  const tournamentVal   = document.getElementById("tournamentVal");

  // Ecosystem sliders
  const foodRateSlider = document.getElementById("foodRateSlider");
  const foodRateVal    = document.getElementById("foodRateVal");
  const maxFoodSlider  = document.getElementById("maxFoodSlider");
  const maxFoodVal     = document.getElementById("maxFoodVal");
  const metaCostSlider = document.getElementById("metaCostSlider");
  const metaCostVal    = document.getElementById("metaCostVal");

  // Inspector
  const inspectorEmpty   = document.getElementById("inspectorEmpty");
  const inspectorContent = document.getElementById("inspectorContent");
  const inspectorTitle   = document.getElementById("inspectorTitle");
  const inspectorStats   = document.getElementById("inspectorStats");
  const genomeViz        = document.getElementById("genomeViz");
  const inspectorTraits  = document.getElementById("inspectorTraits");
  const nnCanvas         = document.getElementById("nnCanvas");
  const nnCtx            = nnCanvas.getContext("2d");

  // Analytics
  const analyticsPanel  = document.getElementById("analyticsPanel");
  const analyticsToggle = document.getElementById("analyticsToggle");

  // Event feed
  const eventFeedEl = document.getElementById("eventFeed");

  // ── Event Feed System ────────────────────────────────────
  const EVENT_FEED_MAX = 8;
  const eventCooldowns = {};
  const EVENT_COOLDOWN = 1500; // ms per event-type

  function addEvent(type, emoji, text, color) {
    const now = performance.now();
    if (eventCooldowns[type] && now - eventCooldowns[type] < EVENT_COOLDOWN) return;
    eventCooldowns[type] = now;

    const div = document.createElement("div");
    div.className = "event-item";
    div.style.borderLeftColor = color;
    div.textContent = `${emoji}  ${text}`;
    eventFeedEl.prepend(div);

    while (eventFeedEl.children.length > EVENT_FEED_MAX) {
      eventFeedEl.removeChild(eventFeedEl.lastChild);
    }
    div.addEventListener("animationend", () => div.remove());
  }

  // ── Budding (asexual reproduction) config ─────────────────
  const BUDDING_ENERGY_THRESHOLD = 100;  // energy needed to bud (was 120 — unreachable)
  const BUDDING_CHANCE = 0.006;          // per tick per eligible bacterium (was 0.003)
  const MAX_POPULATION = 600;            // hard cap

  // ── Utilities ──────────────────────────────────────────────
  const rand  = (min, max) => Math.random() * (max - min) + min;
  const randInt = (min, max) => Math.floor(rand(min, max + 1));
  const pick  = (arr) => arr[Math.floor(Math.random() * arr.length)];
  const hexToRGBA = (hex, a) => {
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    return `rgba(${r},${g},${b},${a})`;
  };

  // ── Canvas resize ─────────────────────────────────────────
  function resize() {
    const wrap = canvas.parentElement;
    canvas.width  = wrap.clientWidth * devicePixelRatio;
    canvas.height = wrap.clientHeight * devicePixelRatio;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  }
  window.addEventListener("resize", resize);
  resize();

  function canvasW() { return canvas.width / devicePixelRatio; }
  function canvasH() { return canvas.height / devicePixelRatio; }

  // ── Create a bacterium ────────────────────────────────────
  function createBacterium(x, y, genome) {
    const w = canvasW();
    const h = canvasH();

    // Genome: either provided (offspring) or fresh random
    const g = genome || Genetics.createGenome();

    // If no genome provided, set initial trait bias
    if (!genome) {
      // Make sure the bacterium has roughly `traitsPerBacterium` traits
      // by biasing those genes above 0.5
      const indices = [];
      for (let i = 0; i < Genetics.TRAIT_GENE_COUNT; i++) indices.push(i);
      // Shuffle and pick
      for (let i = indices.length - 1; i > 0; i--) {
        const j = randInt(0, i);
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
      // Set first N above 0.5, rest below
      for (let i = 0; i < Genetics.TRAIT_GENE_COUNT; i++) {
        if (i < traitsPerBacterium) {
          g[indices[i]] = 0.5 + Math.random() * 0.5; // [0.5, 1.0]
        } else {
          g[indices[i]] = Math.random() * 0.45; // [0, 0.45]
        }
      }
    }

    // Derive traits from genome
    const traits = Genetics.expressTraits(g, TRAIT_POOL);
    const brain  = Genetics.buildBrain(g);
    const dominantTrait = traits.length > 0 ? traits[0] : TRAIT_POOL[0];
    const baseSize = 3 + traits.length * 0.4;
    const isPredator = traits.some(t => t.name === "Predatory Behavior");

    return {
      id: ++bacteriaIdCounter,
      x: x ?? rand(20, w - 20),
      y: y ?? rand(20, h - 20),
      vx: rand(-0.6, 0.6),
      vy: rand(-0.6, 0.6),
      size: baseSize + rand(-0.3, 0.8),
      genome: g,
      traits,
      brain,
      color: dominantTrait.color,
      alive: true,
      alpha: 1,
      pulse: rand(0, Math.PI * 2),
      wobble: rand(0.3, 1.0),
      age: 0,
      energy: Ecosystem.config.startEnergy,
      fitness: 0,
      species: 0,
      foodEaten: 0,
      energyCollected: 0,
      survivalTime: 0,
      generation: generation,
      isPredator,
      deathTime: 0,
      deathCause: "",
    };
  }

  // ── Population init ───────────────────────────────────────
  async function initPopulation() {
    bacteria = [];
    deadBacteria = [];
    generation = 0;
    diedThisRound = 0;
    selectionHistory = [];
    inspectedBacterium = null;
    bacteriaIdCounter = 0;
    traitLog.innerHTML = "";

    const w = canvasW();
    const h = canvasH();

    // Try Python backend for genome creation (NumPy)
    const apiResult = await MLApi.initPopulation(initialPop, traitsPerBacterium);

    if (apiResult) {
      console.log("[Python] Population created (NumPy)");
      for (let i = 0; i < apiResult.genomes.length; i++) {
        const genome = new Float64Array(apiResult.genomes[i]);
        bacteria.push(createBacterium(null, null, genome));
      }
    } else {
      // JS fallback
      for (let i = 0; i < initialPop; i++) {
        bacteria.push(createBacterium());
      }
    }

    Ecosystem.reset(w, h);
    Analytics.reset();

    // Record generation 0 — try Python stats first
    bacteria.forEach(b => { b.fitness = Genetics.computeFitness(b); });
    const stats = await MLApi.computeStats(bacteria);
    if (stats) {
      console.log("[Python] Gen 0 stats computed (scikit-learn k-means)");
    }

    Analytics.recordGeneration(0, bacteria, TRAIT_POOL);
    Analytics.updateCharts(TRAIT_POOL);

    updateStats();
    buildTraitLegend();
    updateInspector();
    updateBackendBadge();
  }

  // ── Trait legend ──────────────────────────────────────────
  function buildTraitLegend() {
    traitLegend.innerHTML = "";
    const present = new Set();
    bacteria.filter(b => b.alive).forEach(b => b.traits.forEach(t => present.add(t.name)));
    TRAIT_POOL.forEach(t => {
      if (!present.has(t.name)) return;
      const pill = document.createElement("span");
      pill.className = "trait-pill";
      pill.textContent = t.name;
      pill.style.background = hexToRGBA(t.color, 0.15);
      pill.style.color = t.color;
      pill.style.border = `1px solid ${hexToRGBA(t.color, 0.25)}`;
      pill.title = t.desc;
      traitLegend.appendChild(pill);
    });
  }

  // ── Stats update ──────────────────────────────────────────
  function updateStats() {
    const alive = bacteria.filter(b => b.alive);
    const presentTraits = new Set();
    alive.forEach(b => b.traits.forEach(t => presentTraits.add(t.name)));

    const stats = Genetics.populationStats(alive);

    statAlive.textContent    = alive.length;
    statGen.textContent      = generation;
    statDied.textContent     = diedThisRound;
    statTraits.textContent   = presentTraits.size;
    statMeanFit.textContent  = stats.meanFitness.toFixed(2);
    statDiversity.textContent= stats.diversity.toFixed(2);
    genBadge.textContent     = generation;
    hsPop.textContent        = alive.length;
    hsFit.textContent        = stats.meanFitness.toFixed(2);

    // Hardy-Weinberg table
    const hw = Genetics.hardyWeinberg(alive, TRAIT_POOL);
    if (hw.length > 0) {
      const top = hw.sort((a, b) => b.departure - a.departure).slice(0, 10);
      hwTable.innerHTML = `<table style="width:100%;border-collapse:collapse;">
        <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
          <th style="text-align:left;padding:3px 4px;color:#777;">Trait</th>
          <th style="text-align:right;padding:3px 4px;color:#777;">Freq</th>
          <th style="text-align:right;padding:3px 4px;color:#777;">HW Dep.</th>
        </tr>
        ${top.map(r => `<tr>
          <td style="padding:2px 4px;">${r.trait}</td>
          <td style="text-align:right;padding:2px 4px;font-family:'JetBrains Mono',monospace;">${r.observed.toFixed(3)}</td>
          <td style="text-align:right;padding:2px 4px;font-family:'JetBrains Mono',monospace;color:${r.departure > 0.3 ? '#f87171' : '#34d399'};">${r.departure.toFixed(3)}</td>
        </tr>`).join("")}
      </table>`;
    }
  }

  // ═══════════════════════════════════════════════════════════
  //  NATURAL SELECTION
  // ═══════════════════════════════════════════════════════════
  function triggerSelection() {
    if (selecting) return;
    const alive = bacteria.filter(b => b.alive);
    if (alive.length === 0) return;

    selecting = true;
    generation++;

    // Compute fitness for all alive before selection
    alive.forEach(b => { b.fitness = Genetics.computeFitness(b); });

    // Pick random trait from FULL pool
    const selectedTrait = pick(TRAIT_POOL);
    const traitIndex = TRAIT_POOL.indexOf(selectedTrait);

    // Announcement
    selTraitName.textContent = `"${selectedTrait.name}"`;
    selTraitName.style.color = selectedTrait.color;
    selTraitDesc.textContent = `Only bacteria with this trait survive!`;
    selAnnounce.classList.remove("visible");
    void selAnnounce.offsetWidth;
    selAnnounce.classList.add("visible");

    // Flash
    flashOverlay.style.background = `radial-gradient(ellipse at center, ${hexToRGBA(selectedTrait.color, 0.4)}, transparent 70%)`;
    flashOverlay.classList.remove("active");
    void flashOverlay.offsetWidth;
    flashOverlay.classList.add("active");

    // Kill bacteria without the trait
    let died = 0;
    alive.forEach(b => {
      const hasTrait = b.genome[traitIndex] > 0.5;
      if (!hasTrait) {
        b.alive = false;
        b.deathTime = performance.now();
        b.deathCause = "selection";
        deadBacteria.push(b);
        died++;
      }
    });

    diedThisRound = died;

    // Log selection event to feed
    addEvent("selection", "[SELECT]", `Selection killed ${died} bacteria!`, "#a78bfa");

    // Remove dead from main array so they don't linger
    bacteria = bacteria.filter(b => b.alive);
    const survivors = bacteria;

    // Log
    const entry = document.createElement("div");
    entry.className = "trait-entry";
    entry.innerHTML = `
      <span class="round">#${generation}</span>
      <span class="name" style="color:${selectedTrait.color}">${selectedTrait.name}</span>
      <span class="survived">${survivors.length} lived</span>
      <span class="died">${died} died</span>
    `;
    traitLog.prepend(entry);
    selectionHistory.push({ trait: selectedTrait, survived: survivors.length, died });

    // Also add a hazard zone for ongoing pressure
    Ecosystem.addHazardZone(canvasW(), canvasH());

    // Reproduce with GA after delay
    setTimeout(async () => {
      await reproduceGA(survivors);
      selecting = false;

      // Record analytics — try Python stats
      const nowAlive = bacteria.filter(b => b.alive);
      nowAlive.forEach(b => { b.fitness = Genetics.computeFitness(b); });

      const pyStats = await MLApi.computeStats(bacteria);
      if (pyStats) {
        console.log(`[Python] Gen ${generation} stats: diversity=${pyStats.diversity}, species=${pyStats.species_count}`);
      }

      Analytics.recordGeneration(generation, bacteria, TRAIT_POOL);
      Analytics.updateCharts(TRAIT_POOL);

      updateStats();
      buildTraitLegend();
      updateBackendBadge();
    }, 1200);

    updateStats();
  }

  // ═══════════════════════════════════════════════════════════
  //  GA REPRODUCTION (Python NumPy or JS fallback)
  // ═══════════════════════════════════════════════════════════
  async function reproduceGA(survivors) {
    if (survivors.length === 0) return;

    // Compute fitness
    survivors.forEach(b => { b.fitness = Genetics.computeFitness(b); });

    // ─── KEY FIX: Don't refill to initialPop ───
    // Survivors breed a modest number of offspring (30% of current pop).
    // This means each selection event permanently reduces the population.
    // The population slowly recovers but never instantly resets.
    const offspringCount = Math.max(1, Math.floor(survivors.length * 0.3));
    const targetPop = survivors.length + offspringCount;

    // Try Python backend for reproduction (NumPy GA)
    const apiResult = await MLApi.reproduce(survivors, targetPop, {
      mutationRate,
      mutationSigma,
      elitismPct,
      crossoverMethod,
      tournamentSize,
    });

    const newBacteria = [];

    if (apiResult) {
      console.log(`[Python] GA reproduction — ${apiResult.offspring_genomes.length} offspring (pop: ${survivors.length} -> ${targetPop})`);

      // Keep all survivors
      for (const s of survivors) {
        newBacteria.push(s);
      }

      // Add Python-created offspring
      for (const genomeArr of apiResult.offspring_genomes) {
        const genome = new Float64Array(genomeArr);
        const parentRef = survivors[Math.floor(Math.random() * survivors.length)];
        const offspring = createBacterium(
          parentRef.x + rand(-25, 25),
          parentRef.y + rand(-25, 25),
          genome
        );
        offspring.generation = generation;
        newBacteria.push(offspring);
      }
    } else {
      // JS fallback — same logic: only breed 30% more
      for (const s of survivors) {
        newBacteria.push(s);
      }

      for (let i = 0; i < offspringCount; i++) {
        const parentA = Genetics.tournamentSelect(survivors, tournamentSize);
        const parentB = Genetics.tournamentSelect(survivors, tournamentSize);
        let childGenome = Genetics.crossover(parentA.genome, parentB.genome, crossoverMethod);
        childGenome = Genetics.mutate(childGenome, mutationRate, mutationSigma);
        const offspring = createBacterium(
          parentA.x + rand(-25, 25),
          parentA.y + rand(-25, 25),
          childGenome
        );
        offspring.generation = generation;
        newBacteria.push(offspring);
      }
    }

    // Replace bacteria array
    bacteria = newBacteria.filter(b => b.alive);
  }

  // ═══════════════════════════════════════════════════════════
  //  MAIN SIMULATION LOOP
  // ═══════════════════════════════════════════════════════════
  let lastTime = performance.now();

  function update(now) {
    requestAnimationFrame(update);

    if (paused) return;

    const dt = Math.min((now - lastTime) / 16.67, 3) * simSpeed;
    lastTime = now;

    const w = canvasW();
    const h = canvasH();

    ctx.clearRect(0, 0, w, h);
    drawGrid(w, h);

    // ── Update ecosystem (food, hazards, energy) ──
    Ecosystem.update(dt, bacteria, w, h);

    // ── Detect newly dead bacteria & log events ──
    const frameDeath = {};
    for (let i = bacteria.length - 1; i >= 0; i--) {
      const b = bacteria[i];
      if (!b.alive && !b.deathTime) {
        b.deathTime = now;
        deadBacteria.push(b);
        bacteria.splice(i, 1);
        // Categorise the death
        let cause = b.deathCause || "unknown";
        if (cause === "starvation" && b.lastDamageSource) cause = b.lastDamageSource;
        frameDeath[cause] = (frameDeath[cause] || 0) + 1;
      }
    }
    // Push death events to the live feed
    for (const [cause, count] of Object.entries(frameDeath)) {
      const n = count > 1 ? `${count} bacteria` : "A bacterium";
      switch (cause) {
        case "predation":
          addEvent("predation", "[HUNT]", `Predator hunted ${count > 1 ? count + " prey" : "prey"}`, "#ef4444");
          break;
        case "starvation":
          addEvent("starvation", "[STARVE]", `${n} starved`, "#f87171");
          break;
        case "old age":
          addEvent("old_age", "[AGE]", `${n} died of old age`, "#94a3b8");
          break;
        default:
          if (cause.includes("Zone")) {
            addEvent(cause, "[HAZARD]", `${n} died in ${cause}`, "#fbbf24");
          }
          break;
      }
    }

    // ── Budding (asexual reproduction) ──
    if (bacteria.length < MAX_POPULATION && !selecting) {
      const budders = [];
      for (const b of bacteria) {
        if (!b.alive) continue;
        if (b.energy >= BUDDING_ENERGY_THRESHOLD) {
          let chance = BUDDING_CHANCE;
          if (b.traits.some(t => t.name === "Rapid Division")) chance *= 2.5;
          if (Math.random() < chance * dt) budders.push(b);
        }
      }
      for (const parent of budders) {
        if (bacteria.length >= MAX_POPULATION) break;
        // Slightly mutated clone
        const childGenome = Genetics.mutate(
          new Float64Array(parent.genome),
          mutationRate * 0.3,
          mutationSigma * 0.3
        );
        const offspring = createBacterium(
          parent.x + rand(-15, 15),
          parent.y + rand(-15, 15),
          childGenome
        );
        offspring.generation = generation;
        offspring.energy = parent.energy * 0.4;
        parent.energy *= 0.6; // parent splits energy with child
        bacteria.push(offspring);
        addEvent("budding", "[BUD]", `#${parent.id} budded -> #${offspring.id}`, "#4ade80");
      }
    }

    // ── Draw hazard zones (behind everything) ──
    Ecosystem.drawHazardZones(ctx);

    // ── Draw food ──
    Ecosystem.drawFood(ctx);

    // ── Draw dead bacteria (fade out) ──
    for (let i = deadBacteria.length - 1; i >= 0; i--) {
      const b = deadBacteria[i];
      const elapsed = (now - b.deathTime) / 1000;
      b.alpha = Math.max(0, 1 - elapsed / 1.5);
      if (b.alpha <= 0) {
        deadBacteria.splice(i, 1);
        continue;
      }
      drawBacterium(b, b.alpha * 0.4, b.size * b.alpha, true);
    }

    // ── Update & draw alive bacteria ──
    for (const b of bacteria) {
      if (!b.alive) continue;

      b.age += dt;
      b.pulse += 0.03 * dt;

      // ── Old age death ──
      // Bacteria have a natural lifespan — older ones die more often
      const maxAge = 1800 + (b.traits.some(t => t.name === "Spore Formation") ? 600 : 0)
                          + (b.traits.some(t => t.name === "Endospore Armor") ? 400 : 0);
      if (b.age > maxAge) {
        const agePastMax = b.age - maxAge;
        if (Math.random() < (agePastMax / 600) * 0.02 * dt) {
          b.alive = false;
          b.deathCause = "old age";
          continue; // dead detection loop handles deathTime, deadBacteria, splice & event
        }
      }

      // ── Neural network driven movement ──
      const inputs = Ecosystem.computeNNInputs(b, bacteria, w, h);
      const [turnDelta, speedMult] = b.brain.forward(inputs);

      // Convert NN output to velocity adjustments
      const currentAngle = Math.atan2(b.vy, b.vx);
      const newAngle = currentAngle + turnDelta * 0.3 * dt;
      const baseSpeed = 0.6 + (b.traits.some(t => t.name === "Fast Metabolism") ? 0.4 : 0)
                            - (b.traits.some(t => t.name === "Slow Metabolism") ? 0.2 : 0);
      const speed = baseSpeed * (0.5 + (speedMult + 1) * 0.3);

      b.vx += Math.cos(newAngle) * speed * 0.1 * dt;
      b.vy += Math.sin(newAngle) * speed * 0.1 * dt;

      // Add some wobble for organic feel
      b.vx += rand(-0.05, 0.05) * b.wobble * dt;
      b.vy += rand(-0.05, 0.05) * b.wobble * dt;

      // Damping
      b.vx *= 0.97;
      b.vy *= 0.97;

      // Clamp
      const maxV = 1.8;
      b.vx = Math.max(-maxV, Math.min(maxV, b.vx));
      b.vy = Math.max(-maxV, Math.min(maxV, b.vy));

      b.x += b.vx * dt;
      b.y += b.vy * dt;

      // Bounce
      if (b.x < b.size)     { b.x = b.size; b.vx *= -0.7; }
      if (b.x > w - b.size) { b.x = w - b.size; b.vx *= -0.7; }
      if (b.y < b.size)     { b.y = b.size; b.vy *= -0.7; }
      if (b.y > h - b.size) { b.y = h - b.size; b.vy *= -0.7; }

      const pulseSize = b.size + Math.sin(b.pulse) * 0.5;
      drawBacterium(b, 1, pulseSize, false);

      // Draw energy bar
      Ecosystem.drawEnergyBar(ctx, b);
    }

    // Highlight inspected bacterium
    if (inspectedBacterium && inspectedBacterium.alive) {
      const ib = inspectedBacterium;
      ctx.strokeStyle = "rgba(255,255,255,0.6)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(ib.x, ib.y, ib.size + 8, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    lastTime = now;
  }

  // ── Draw a single bacterium ───────────────────────────────
  function drawBacterium(b, alpha, size, isDead) {
    // Species outline ring
    if (!isDead && b.species !== undefined) {
      const specColor = SPECIES_COLORS[b.species % SPECIES_COLORS.length];
      ctx.strokeStyle = hexToRGBA(specColor, 0.3 * alpha);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(b.x, b.y, size + 2, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Glow
    const gradient = ctx.createRadialGradient(b.x, b.y, 0, b.x, b.y, size * 3.5);
    gradient.addColorStop(0, hexToRGBA(b.color, 0.1 * alpha));
    gradient.addColorStop(1, "transparent");
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(b.x, b.y, size * 3.5, 0, Math.PI * 2);
    ctx.fill();

    // Body
    const bodyAlpha = isDead ? 0.35 : (b.isPredator ? 0.95 : 0.8);
    ctx.fillStyle = hexToRGBA(b.color, bodyAlpha * alpha);
    ctx.beginPath();
    ctx.arc(b.x, b.y, size, 0, Math.PI * 2);
    ctx.fill();

    // Predator indicator (spiky border)
    if (b.isPredator && !isDead) {
      ctx.strokeStyle = hexToRGBA("#dc2626", 0.5 * alpha);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      const spikes = 8;
      for (let i = 0; i <= spikes * 2; i++) {
        const angle = (Math.PI * 2 / (spikes * 2)) * i + b.pulse * 0.2;
        const r = i % 2 === 0 ? size + 3 : size + 1;
        if (i === 0) ctx.moveTo(b.x + Math.cos(angle) * r, b.y + Math.sin(angle) * r);
        else ctx.lineTo(b.x + Math.cos(angle) * r, b.y + Math.sin(angle) * r);
      }
      ctx.closePath();
      ctx.stroke();
    }

    // Bright core
    ctx.fillStyle = hexToRGBA("#ffffff", 0.3 * alpha);
    ctx.beginPath();
    ctx.arc(b.x - size * 0.2, b.y - size * 0.2, size * 0.3, 0, Math.PI * 2);
    ctx.fill();

    // Death cross
    if (isDead && alpha > 0.1) {
      ctx.strokeStyle = hexToRGBA("#ff4444", alpha * 0.7);
      ctx.lineWidth = 1.5;
      const s = size * 1.1;
      ctx.beginPath();
      ctx.moveTo(b.x - s, b.y - s); ctx.lineTo(b.x + s, b.y + s);
      ctx.moveTo(b.x + s, b.y - s); ctx.lineTo(b.x - s, b.y + s);
      ctx.stroke();
    }

    // Trait ring dots
    if (!isDead && alpha > 0.5 && size > 3 && b.traits.length > 0) {
      const tc = b.traits.length;
      b.traits.forEach((t, i) => {
        const angle = (Math.PI * 2 / tc) * i + b.pulse * 0.25;
        const ringR = size + 5;
        const tx = b.x + Math.cos(angle) * ringR;
        const ty = b.y + Math.sin(angle) * ringR;
        ctx.fillStyle = hexToRGBA(t.color, 0.55 * alpha);
        ctx.beginPath();
        ctx.arc(tx, ty, 1, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }

  // ── Background grid ───────────────────────────────────────
  function drawGrid(w, h) {
    ctx.strokeStyle = "rgba(255,255,255,0.012)";
    ctx.lineWidth = 1;
    const step = 40;
    for (let x = 0; x < w; x += step) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += step) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
  }

  // ═══════════════════════════════════════════════════════════
  //  INSPECTOR
  // ═══════════════════════════════════════════════════════════
  function updateInspector() {
    if (!inspectedBacterium || !inspectedBacterium.alive) {
      inspectorEmpty.style.display = "block";
      inspectorContent.style.display = "none";
      return;
    }

    const b = inspectedBacterium;
    inspectorEmpty.style.display = "none";
    inspectorContent.style.display = "flex";

    inspectorTitle.textContent = `Bacterium #${b.id}`;

    // Stats
    inspectorStats.innerHTML = `
      <div class="inspector-stat"><div class="is-label">Energy</div><div class="is-value" style="color:${b.energy > 50 ? '#34d399' : '#f87171'}">${b.energy.toFixed(1)}</div></div>
      <div class="inspector-stat"><div class="is-label">Fitness</div><div class="is-value" style="color:#c084fc">${b.fitness.toFixed(2)}</div></div>
      <div class="inspector-stat"><div class="is-label">Food Eaten</div><div class="is-value">${b.foodEaten}</div></div>
      <div class="inspector-stat"><div class="is-label">Species</div><div class="is-value" style="color:${SPECIES_COLORS[b.species % SPECIES_COLORS.length]}">${b.species}</div></div>
      <div class="inspector-stat"><div class="is-label">Generation</div><div class="is-value">${b.generation}</div></div>
      <div class="inspector-stat"><div class="is-label">Traits</div><div class="is-value">${b.traits.length}</div></div>
    `;

    // Genome grid (40 trait genes)
    genomeViz.innerHTML = "";
    for (let i = 0; i < Genetics.TRAIT_GENE_COUNT && i < TRAIT_POOL.length; i++) {
      const val = b.genome[i];
      const expressed = val > 0.5;
      const trait = TRAIT_POOL[i];
      const cell = document.createElement("div");
      cell.className = "genome-cell";
      cell.style.background = expressed
        ? hexToRGBA(trait.color, 0.5 + val * 0.4)
        : `rgba(255,255,255,${0.02 + val * 0.06})`;
      cell.innerHTML = `<span class="tooltip">${trait.name}: ${val.toFixed(2)}</span>`;
      genomeViz.appendChild(cell);
    }

    // Expressed traits
    inspectorTraits.innerHTML = "";
    b.traits.forEach(t => {
      const pill = document.createElement("span");
      pill.className = "trait-pill";
      pill.textContent = t.name;
      pill.style.background = hexToRGBA(t.color, 0.15);
      pill.style.color = t.color;
      pill.style.border = `1px solid ${hexToRGBA(t.color, 0.3)}`;
      inspectorTraits.appendChild(pill);
    });

    // NN visualisation
    drawNNViz(b.brain);
  }

  function drawNNViz(brain) {
    const c = nnCtx;
    const w = nnCanvas.width;
    const h = nnCanvas.height;
    c.clearRect(0, 0, w, h);

    const layers = brain.topology;
    const layerX = [];
    const padding = 30;
    for (let l = 0; l < layers.length; l++) {
      layerX.push(padding + (w - padding * 2) * (l / (layers.length - 1)));
    }

    // Compute node positions
    const nodePos = [];
    for (let l = 0; l < layers.length; l++) {
      const nodes = [];
      const n = layers[l];
      for (let i = 0; i < n; i++) {
        const y = padding + (h - padding * 2) * (i / Math.max(1, n - 1));
        nodes.push({ x: layerX[l], y });
      }
      nodePos.push(nodes);
    }

    // Draw connections
    let wi = 0;
    for (let l = 1; l < layers.length; l++) {
      for (let j = 0; j < layers[l]; j++) {
        for (let i = 0; i < layers[l - 1]; i++) {
          const weight = brain.weights[wi++];
          const intensity = Math.min(1, Math.abs(weight));
          const color = weight > 0
            ? `rgba(74,222,128,${0.1 + intensity * 0.4})`
            : `rgba(248,113,113,${0.1 + intensity * 0.4})`;
          c.strokeStyle = color;
          c.lineWidth = 0.5 + intensity;
          c.beginPath();
          c.moveTo(nodePos[l - 1][i].x, nodePos[l - 1][i].y);
          c.lineTo(nodePos[l][j].x, nodePos[l][j].y);
          c.stroke();
        }
        wi++; // skip bias
      }
    }

    // Draw nodes
    const labels = [
      ["Food∠", "Food𝑑", "Thr∠", "Thr𝑑", "E", "v"],
      null,
      ["Δθ", "|v|"],
    ];
    for (let l = 0; l < layers.length; l++) {
      for (let i = 0; i < layers[l]; i++) {
        const { x, y } = nodePos[l][i];
        c.fillStyle = l === 0 ? "rgba(56,189,248,0.7)" : l === layers.length - 1 ? "rgba(251,191,36,0.7)" : "rgba(167,139,250,0.5)";
        c.beginPath();
        c.arc(x, y, 4, 0, Math.PI * 2);
        c.fill();
        // Label
        if (labels[l] && labels[l][i]) {
          c.fillStyle = "rgba(255,255,255,0.5)";
          c.font = "7px 'JetBrains Mono', monospace";
          c.textAlign = l === 0 ? "right" : "left";
          c.fillText(labels[l][i], l === 0 ? x - 7 : x + 7, y + 2);
        }
      }
    }
  }

  // ═══════════════════════════════════════════════════════════
  //  EVENT WIRING
  // ═══════════════════════════════════════════════════════════

  // Tab switching
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    });
  });

  // Analytics panel toggle
  analyticsToggle.addEventListener("click", () => {
    analyticsPanel.classList.toggle("collapsed");
  });

  // Selection button
  btnSelect.addEventListener("click", triggerSelection);

  // Reset
  btnReset.addEventListener("click", () => { initPopulation(); });

  // Pause
  btnPause.addEventListener("click", () => {
    paused = !paused;
    btnPause.textContent = paused ? "Play" : "Pause";
    pauseOverlay.classList.toggle("visible", paused);
  });

  // Speed buttons
  speedBtns.querySelectorAll("button").forEach(btn => {
    btn.addEventListener("click", () => {
      speedBtns.querySelectorAll("button").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      simSpeed = parseFloat(btn.dataset.speed);
    });
  });

  // Population sliders
  popSlider.addEventListener("input", () => { initialPop = parseInt(popSlider.value); popVal.textContent = initialPop; });
  traitSlider.addEventListener("input", () => { traitsPerBacterium = parseInt(traitSlider.value); traitCountVal.textContent = traitsPerBacterium; });

  // GA sliders
  mutRateSlider.addEventListener("input", () => { mutationRate = parseFloat(mutRateSlider.value); mutRateVal.textContent = mutationRate.toFixed(2); });
  mutSigmaSlider.addEventListener("input", () => { mutationSigma = parseFloat(mutSigmaSlider.value); mutSigmaVal.textContent = mutationSigma.toFixed(2); });
  elitismSlider.addEventListener("input", () => { elitismPct = parseInt(elitismSlider.value); elitismVal.textContent = elitismPct; });
  crossoverSelect.addEventListener("change", () => { crossoverMethod = crossoverSelect.value; });
  tournamentSlider.addEventListener("input", () => { tournamentSize = parseInt(tournamentSlider.value); tournamentVal.textContent = tournamentSize; });

  // Ecosystem sliders
  foodRateSlider.addEventListener("input", () => { Ecosystem.config.foodSpawnRate = parseFloat(foodRateSlider.value); foodRateVal.textContent = Ecosystem.config.foodSpawnRate.toFixed(1); });
  maxFoodSlider.addEventListener("input", () => { Ecosystem.config.maxFood = parseInt(maxFoodSlider.value); maxFoodVal.textContent = Ecosystem.config.maxFood; });
  metaCostSlider.addEventListener("input", () => { Ecosystem.config.metabolicCost = parseFloat(metaCostSlider.value); metaCostVal.textContent = Ecosystem.config.metabolicCost.toFixed(2); });

  // Canvas click → inspect bacterium
  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left);
    const my = (e.clientY - rect.top);

    let closest = null;
    let closestDist = Infinity;
    for (const b of bacteria) {
      if (!b.alive) continue;
      const dx = b.x - mx;
      const dy = b.y - my;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < b.size + 10 && d < closestDist) {
        closestDist = d;
        closest = b;
      }
    }

    inspectedBacterium = closest;
    updateInspector();

    // Switch to inspector tab
    if (closest) {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      document.querySelector('.tab-btn[data-tab="inspector"]').classList.add("active");
      document.getElementById("tab-inspector").classList.add("active");
    }
  });

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    switch (e.code) {
      case "Space":
        e.preventDefault();
        triggerSelection();
        break;
      case "KeyP":
        e.preventDefault();
        btnPause.click();
        break;
      case "KeyR":
        e.preventDefault();
        initPopulation();
        break;
    }
  });

  // Periodically update inspector + live stats
  setInterval(() => {
    if (inspectedBacterium) updateInspector();
    updateStats(); // keep alive count accurate as bacteria die naturally
  }, 500);

  // ── Backend status badge ────────────────────────────────────
  function updateBackendBadge() {
    const badge = document.getElementById("backendBadge");
    if (!badge) return;
    if (MLApi.connected) {
      badge.textContent = "Python ML";
      badge.style.color = "#4ade80";
      badge.title = "Connected to Python backend (FastAPI + NumPy + scikit-learn)";
    } else {
      badge.textContent = "JS Fallback";
      badge.style.color = "#f87171";
      badge.title = "Python backend not available — using JavaScript fallback";
    }
  }

  // ── Initialise ────────────────────────────────────────────
  (async () => {
    // Check if Python backend is available
    await MLApi.checkConnection();
    updateBackendBadge();

    Analytics.init();
    await initPopulation();
    requestAnimationFrame(update);
  })();

})();
