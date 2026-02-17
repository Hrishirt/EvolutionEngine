// ═══════════════════════════════════════════════════════════════
//  API CLIENT — Bridge between JS frontend and Python ML backend
//  Calls FastAPI endpoints for GA, stats, and NN operations
// ═══════════════════════════════════════════════════════════════

const MLApi = {

  BASE_URL: window.location.origin,
  connected: false,

  // ── Health check ─────────────────────────────────────────────
  async checkConnection() {
    try {
      const res = await fetch(`${this.BASE_URL}/api/health`);
      if (res.ok) {
        this.connected = true;
        const data = await res.json();
        console.log("[API] Python ML backend connected:", data);
        return data;
      }
    } catch (e) {
      this.connected = false;
      console.warn("[API] Python ML backend not available -- using JS fallback");
    }
    return null;
  },

  // ── Create initial population genomes (NumPy) ───────────────
  async initPopulation(popSize, traitsPerBacterium) {
    if (!this.connected) return null;
    try {
      const res = await fetch(`${this.BASE_URL}/api/init`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pop_size: popSize,
          traits_per_bacterium: traitsPerBacterium,
        }),
      });
      if (res.ok) return await res.json();
    } catch (e) {
      console.warn("API init failed, using JS fallback:", e.message);
    }
    return null;
  },

  // ── GA reproduction (NumPy crossover + mutation) ─────────────
  async reproduce(survivors, targetPop, config) {
    if (!this.connected) return null;
    try {
      const survivorData = survivors.map(b => ({
        genome: Array.from(b.genome),
        fitness: b.fitness || 0,
      }));

      const res = await fetch(`${this.BASE_URL}/api/reproduce`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          survivors: survivorData,
          target_pop: targetPop,
          mutation_rate: config.mutationRate,
          mutation_sigma: config.mutationSigma,
          elitism_pct: config.elitismPct,
          crossover_method: config.crossoverMethod,
          tournament_size: config.tournamentSize,
        }),
      });
      if (res.ok) return await res.json();
    } catch (e) {
      console.warn("API reproduce failed, using JS fallback:", e.message);
    }
    return null;
  },

  // ── Population statistics (NumPy + scikit-learn) ─────────────
  async computeStats(population, kSpecies = 4) {
    if (!this.connected) return null;
    try {
      const alive = population.filter(b => b.alive);
      if (alive.length === 0) return null;

      const genomes = alive.map(b => Array.from(b.genome));
      const fitnesses = alive.map(b => b.fitness || 0);

      const res = await fetch(`${this.BASE_URL}/api/stats`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          genomes,
          fitnesses,
          k_species: kSpecies,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        // Apply species assignments back to bacteria
        for (let i = 0; i < alive.length; i++) {
          alive[i].species = data.species_assignments[i] || 0;
        }
        return data;
      }
    } catch (e) {
      console.warn("API stats failed, using JS fallback:", e.message);
    }
    return null;
  },

  // ── Batch NN forward pass (NumPy) ───────────────────────────
  // Used for non-real-time bulk evaluation; real-time stays in JS
  async batchForward(nnWeights, inputs) {
    if (!this.connected) return null;
    try {
      const res = await fetch(`${this.BASE_URL}/api/nn/forward`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          nn_weights: nnWeights,
          inputs,
        }),
      });
      if (res.ok) return await res.json();
    } catch (e) {
      console.warn("API nn/forward failed:", e.message);
    }
    return null;
  },
};
