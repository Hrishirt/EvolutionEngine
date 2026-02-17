// ═══════════════════════════════════════════════════════════════
//  ECOSYSTEM — Food, energy, hazard zones, predator-prey
// ═══════════════════════════════════════════════════════════════

const Ecosystem = {

  // ── Configuration (modified from sidebar) ─────────────────
  // ── Energy-economy balance (math below) ─────────────────
  //  target equilibrium ≈ 200 bacteria
  //  drain/sec = pop × metaCost × 60  → 200×0.04×60 = 480
  //  food/sec  = spawnRate×60×energy   → 0.6×60×25   = 900
  //  ~60 % utilisation → 540 in, 480 out  ≈ stable
  config: {
    foodSpawnRate: 0.6,        // food particles per frame (was 0.3)
    maxFood: 150,              // more food on screen (was 80)
    foodEnergy: 25,            // energy per food (was 20)
    foodSize: 3,
    metabolicCost: 0.04,       // energy per tick (was 0.15 — way too high)
    fastMetabolicMultiplier: 1.8,
    slowMetabolicMultiplier: 0.5,
    startEnergy: 100,
    maxEnergy: 150,
    overcrowdingThreshold: 180, // kicks in closer to target pop (was 100)
    overcrowdingMultiplier: 2.0,// strong push-back when overcrowded (was 1.5)
    predatorEnergyGain: 40,
    predatorCatchRange: 2.5,
    hazardDamage: 0.15,        // less instantly lethal (was 0.3)
    hazardDriftSpeed: 0.1,
  },

  // ── State ─────────────────────────────────────────────────
  food: [],          // { x, y, pulse, age }
  hazardZones: [],   // { x, y, radius, type, traitIndex, color, vx, vy }
  foodAccumulator: 0,

  // Hazard zone definitions — map to TRAIT_POOL indices
  HAZARD_TYPES: [
    { name: "Heat Zone",      traitIndex: 0,  color: "#ef4444" }, // Heat Resistance
    { name: "Cold Zone",      traitIndex: 1,  color: "#38bdf8" }, // Cold Tolerance
    { name: "UV Zone",        traitIndex: 2,  color: "#fbbf24" }, // UV Shield
    { name: "Acid Zone",      traitIndex: 3,  color: "#a3e635" }, // Acid Resistance
    { name: "Radiation Zone", traitIndex: 15, color: "#fcd34d" }, // Radiation Resistance
    { name: "Pressure Zone",  traitIndex: 17, color: "#6ee7b7" }, // Pressure Resistance
    { name: "Metal Zone",     traitIndex: 19, color: "#cbd5e1" }, // Metal Resistance
    { name: "Salt Zone",      traitIndex: 20, color: "#fda4af" }, // Salt Tolerance
  ],

  // ── Initialise / Reset ────────────────────────────────────
  reset(canvasW, canvasH) {
    this.food = [];
    this.hazardZones = [];
    this.foodAccumulator = 0;
    // Spawn generous initial food so population doesn't starve immediately
    for (let i = 0; i < 80; i++) {
      this.food.push(Ecosystem.createFood(canvasW, canvasH));
    }
  },

  createFood(canvasW, canvasH) {
    return {
      x: Math.random() * (canvasW - 40) + 20,
      y: Math.random() * (canvasH - 40) + 20,
      pulse: Math.random() * Math.PI * 2,
      age: 0,
    };
  },

  // ── Add a random hazard zone ──────────────────────────────
  addHazardZone(canvasW, canvasH) {
    const type = this.HAZARD_TYPES[Math.floor(Math.random() * this.HAZARD_TYPES.length)];
    const radius = 40 + Math.random() * 80;
    this.hazardZones.push({
      x: Math.random() * (canvasW - radius * 2) + radius,
      y: Math.random() * (canvasH - radius * 2) + radius,
      radius,
      type: type.name,
      traitIndex: type.traitIndex,
      color: type.color,
      vx: (Math.random() - 0.5) * this.config.hazardDriftSpeed,
      vy: (Math.random() - 0.5) * this.config.hazardDriftSpeed,
      pulse: 0,
      age: 0,
    });
    // Cap at 5 hazard zones
    if (this.hazardZones.length > 5) {
      this.hazardZones.shift();
    }
  },

  // ── Per-frame update ──────────────────────────────────────
  update(dt, bacteria, canvasW, canvasH) {
    const cfg = this.config;

    // Spawn food
    this.foodAccumulator += cfg.foodSpawnRate * dt;
    while (this.foodAccumulator >= 1 && this.food.length < cfg.maxFood) {
      this.food.push(Ecosystem.createFood(canvasW, canvasH));
      this.foodAccumulator -= 1;
    }

    // Update food pulses
    for (const f of this.food) {
      f.pulse += 0.04 * dt;
      f.age += dt;
    }

    // Update hazard zones (drift)
    for (const hz of this.hazardZones) {
      hz.x += hz.vx * dt;
      hz.y += hz.vy * dt;
      hz.pulse += 0.02 * dt;
      hz.age += dt;
      // Bounce off walls
      if (hz.x - hz.radius < 0 || hz.x + hz.radius > canvasW) hz.vx *= -1;
      if (hz.y - hz.radius < 0 || hz.y + hz.radius > canvasH) hz.vy *= -1;
      hz.x = Math.max(hz.radius, Math.min(canvasW - hz.radius, hz.x));
      hz.y = Math.max(hz.radius, Math.min(canvasH - hz.radius, hz.y));
    }

    // Process each living bacterium
    const alive = bacteria.filter(b => b.alive);
    const popCount = alive.length;

    // Overcrowding multiplier — when population exceeds threshold, metabolism spikes
    const overcrowdingFactor = popCount > cfg.overcrowdingThreshold
      ? 1 + (popCount - cfg.overcrowdingThreshold) / cfg.overcrowdingThreshold * (cfg.overcrowdingMultiplier - 1)
      : 1;

    for (const b of alive) {
      // ── Metabolic cost (with overcrowding pressure) ──
      let metaRate = cfg.metabolicCost * overcrowdingFactor;
      if (b.traits.some(t => t.name === "Fast Metabolism")) {
        metaRate *= cfg.fastMetabolicMultiplier;
      } else if (b.traits.some(t => t.name === "Slow Metabolism")) {
        metaRate *= cfg.slowMetabolicMultiplier;
      }
      b.energy -= metaRate * dt;

      // ── Eating food ──
      for (let i = this.food.length - 1; i >= 0; i--) {
        const f = this.food[i];
        const dx = b.x - f.x;
        const dy = b.y - f.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < b.size + cfg.foodSize) {
          b.energy = Math.min(cfg.maxEnergy, b.energy + cfg.foodEnergy);
          b.foodEaten = (b.foodEaten || 0) + 1;
          b.energyCollected = (b.energyCollected || 0) + cfg.foodEnergy;
          this.food.splice(i, 1);
          break; // one food per tick
        }
      }

      // ── Hazard damage ──
      for (const hz of this.hazardZones) {
        const dx = b.x - hz.x;
        const dy = b.y - hz.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < hz.radius) {
          const hasResistance = b.genome[hz.traitIndex] > 0.5;
          if (!hasResistance) {
            b.energy -= cfg.hazardDamage * dt;
            b.lastDamageSource = hz.type; // Track what's hurting this bacterium
          }
        }
      }

      // ── Predator-prey (with wider catch range) ──
      if (b.traits.some(t => t.name === "Predatory Behavior")) {
        for (const prey of alive) {
          if (prey === b || !prey.alive) continue;
          if (prey.traits.some(t => t.name === "Predatory Behavior")) continue;
          const dx = b.x - prey.x;
          const dy = b.y - prey.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          // Catch range: predator size * catchRange multiplier
          if (dist < b.size * cfg.predatorCatchRange) {
            prey.energy = 0;
            prey.alive = false;
            prey.deathCause = "predation";
            b.energy = Math.min(cfg.maxEnergy, b.energy + cfg.predatorEnergyGain);
            b.foodEaten = (b.foodEaten || 0) + 1;
            break;
          }
        }
      }

      // ── Starvation death ──
      if (b.energy <= 0) {
        b.energy = 0;
        b.alive = false;
        b.deathCause = "starvation";
      }

      // Track survival time
      b.survivalTime = (b.survivalTime || 0) + dt * 0.01;
    }
  },

  // ── Neural network input computation ──────────────────────
  /**
   * Compute the 6 NN inputs for a given bacterium.
   * Returns [foodAngle, foodDist, threatAngle, threatDist, energy, speed]
   */
  computeNNInputs(b, bacteria, canvasW, canvasH) {
    const cfg = this.config;

    // Find nearest food
    let nearestFood = null;
    let nearestFoodDist = Infinity;
    for (const f of this.food) {
      const dx = f.x - b.x;
      const dy = f.y - b.y;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < nearestFoodDist) {
        nearestFoodDist = d;
        nearestFood = f;
      }
    }

    // Find nearest threat (predator or hazard)
    let nearestThreatDist = Infinity;
    let nearestThreatAngle = 0;

    // Check predators
    for (const other of bacteria) {
      if (other === b || !other.alive) continue;
      if (other.traits.some(t => t.name === "Predatory Behavior") && !b.traits.some(t => t.name === "Predatory Behavior")) {
        const dx = other.x - b.x;
        const dy = other.y - b.y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < nearestThreatDist) {
          nearestThreatDist = d;
          nearestThreatAngle = Math.atan2(dy, dx);
        }
      }
    }

    // Check hazard zones
    for (const hz of this.hazardZones) {
      const hasResistance = b.genome[hz.traitIndex] > 0.5;
      if (hasResistance) continue;
      const dx = hz.x - b.x;
      const dy = hz.y - b.y;
      const d = Math.sqrt(dx * dx + dy * dy) - hz.radius;
      if (d < nearestThreatDist) {
        nearestThreatDist = d;
        nearestThreatAngle = Math.atan2(dy, dx);
      }
    }

    // Normalise
    const maxDist = Math.sqrt(canvasW * canvasW + canvasH * canvasH);
    const foodAngle = nearestFood ? Math.atan2(nearestFood.y - b.y, nearestFood.x - b.x) / Math.PI : 0;
    const foodDist = nearestFood ? Math.min(nearestFoodDist / maxDist, 1) : 1;
    const threatAngle = nearestThreatAngle / Math.PI;
    const threatDist = Math.min(nearestThreatDist / maxDist, 1);
    const energy = (b.energy || 0) / cfg.maxEnergy;
    const speed = Math.sqrt(b.vx * b.vx + b.vy * b.vy) / 2;

    return [foodAngle, foodDist, threatAngle, threatDist, energy, speed];
  },

  // ── Drawing ───────────────────────────────────────────────
  drawFood(ctx) {
    const cfg = this.config;
    for (const f of this.food) {
      const s = cfg.foodSize + Math.sin(f.pulse) * 0.8;
      // Glow
      const grad = ctx.createRadialGradient(f.x, f.y, 0, f.x, f.y, s * 4);
      grad.addColorStop(0, "rgba(74, 222, 128, 0.15)");
      grad.addColorStop(1, "transparent");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(f.x, f.y, s * 4, 0, Math.PI * 2);
      ctx.fill();

      // Body
      ctx.fillStyle = "rgba(74, 222, 128, 0.8)";
      ctx.beginPath();
      ctx.arc(f.x, f.y, s, 0, Math.PI * 2);
      ctx.fill();

      // Core
      ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
      ctx.beginPath();
      ctx.arc(f.x - s * 0.2, f.y - s * 0.2, s * 0.3, 0, Math.PI * 2);
      ctx.fill();
    }
  },

  drawHazardZones(ctx) {
    for (const hz of this.hazardZones) {
      const pulseR = hz.radius + Math.sin(hz.pulse) * 4;

      // Fill
      const r = parseInt(hz.color.slice(1, 3), 16);
      const g = parseInt(hz.color.slice(3, 5), 16);
      const bl = parseInt(hz.color.slice(5, 7), 16);

      const grad = ctx.createRadialGradient(hz.x, hz.y, 0, hz.x, hz.y, pulseR);
      grad.addColorStop(0, `rgba(${r},${g},${bl},0.08)`);
      grad.addColorStop(0.7, `rgba(${r},${g},${bl},0.05)`);
      grad.addColorStop(1, `rgba(${r},${g},${bl},0)`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(hz.x, hz.y, pulseR, 0, Math.PI * 2);
      ctx.fill();

      // Border
      ctx.strokeStyle = `rgba(${r},${g},${bl},${0.25 + Math.sin(hz.pulse) * 0.1})`;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(hz.x, hz.y, pulseR, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      // Label
      ctx.fillStyle = `rgba(${r},${g},${bl},0.5)`;
      ctx.font = "9px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(hz.type, hz.x, hz.y - pulseR - 6);
    }
  },

  /**
   * Draw energy bar above a bacterium.
   */
  drawEnergyBar(ctx, b) {
    if (!b.alive) return;
    const barW = 12;
    const barH = 2;
    const x = b.x - barW / 2;
    const y = b.y - b.size - 6;
    const pct = Math.max(0, Math.min(1, b.energy / this.config.maxEnergy));

    // Background
    ctx.fillStyle = "rgba(255,255,255,0.1)";
    ctx.fillRect(x, y, barW, barH);

    // Fill
    const r = Math.round(255 * (1 - pct));
    const g = Math.round(200 * pct);
    ctx.fillStyle = `rgba(${r},${g},80,0.7)`;
    ctx.fillRect(x, y, barW * pct, barH);
  },
};
