// ═══════════════════════════════════════════════════════════════
//  ANALYTICS — Chart.js real-time graphs and statistical computations
// ═══════════════════════════════════════════════════════════════

const Analytics = {

  // Chart instances
  popChart: null,
  fitnessChart: null,
  traitChart: null,
  diversityChart: null,

  // Data history (per generation)
  history: {
    generations: [],
    population: [],
    predators: [],
    prey: [],
    meanFitness: [],
    maxFitness: [],
    minFitness: [],
    stdFitness: [],
    diversity: [],
    speciesCount: [],
    traitFrequencies: {},  // traitName -> [freq, freq, ...]
  },

  // Colour palette for trait chart (taken from top-10 traits by variance)
  SPECIES_COLORS: [
    "#ef4444", "#38bdf8", "#fbbf24", "#a78bfa", "#34d399",
    "#f472b6", "#fb923c", "#22d3ee", "#8b5cf6", "#4ade80",
  ],

  // ── Initialise all 4 Chart.js charts ──────────────────────
  init() {
    const baseOpts = {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: {
          labels: { color: "#888", font: { size: 10, family: "Inter" } },
          display: true,
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,0.04)" },
          ticks: { color: "#666", font: { size: 9 } },
          title: { display: true, text: "Generation", color: "#666", font: { size: 10 } },
        },
        y: {
          grid: { color: "rgba(255,255,255,0.04)" },
          ticks: { color: "#666", font: { size: 9 } },
          beginAtZero: true,
        },
      },
    };

    // 1. Population chart
    const popCtx = document.getElementById("popChart").getContext("2d");
    this.popChart = new Chart(popCtx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Total",
            data: [],
            borderColor: "#34d399",
            backgroundColor: "rgba(52,211,153,0.1)",
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 2,
          },
          {
            label: "Predators",
            data: [],
            borderColor: "#ef4444",
            backgroundColor: "rgba(239,68,68,0.08)",
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 1.5,
          },
          {
            label: "Prey",
            data: [],
            borderColor: "#38bdf8",
            backgroundColor: "rgba(56,189,248,0.08)",
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 1.5,
          },
        ],
      },
      options: {
        ...baseOpts,
        plugins: { ...baseOpts.plugins, title: { display: true, text: "Population Dynamics", color: "#aaa", font: { size: 11 } } },
      },
    });

    // 2. Fitness chart
    const fitCtx = document.getElementById("fitnessChart").getContext("2d");
    this.fitnessChart = new Chart(fitCtx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Mean",
            data: [],
            borderColor: "#a78bfa",
            backgroundColor: "rgba(167,139,250,0.1)",
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 2,
          },
          {
            label: "Max",
            data: [],
            borderColor: "#fbbf24",
            borderDash: [3, 3],
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 1.5,
            fill: false,
          },
          {
            label: "Min",
            data: [],
            borderColor: "#f87171",
            borderDash: [3, 3],
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 1.5,
            fill: false,
          },
        ],
      },
      options: {
        ...baseOpts,
        plugins: { ...baseOpts.plugins, title: { display: true, text: "Fitness Over Generations", color: "#aaa", font: { size: 11 } } },
      },
    });

    // 3. Trait frequency chart (stacked area)
    const traitCtx = document.getElementById("traitChart").getContext("2d");
    this.traitChart = new Chart(traitCtx, {
      type: "line",
      data: { labels: [], datasets: [] },
      options: {
        ...baseOpts,
        plugins: {
          ...baseOpts.plugins,
          title: { display: true, text: "Trait Allele Frequencies", color: "#aaa", font: { size: 11 } },
          legend: { display: false },
        },
        scales: {
          ...baseOpts.scales,
          y: { ...baseOpts.scales.y, max: 1, title: { display: true, text: "Frequency", color: "#666", font: { size: 10 } } },
        },
      },
    });

    // 4. Diversity chart
    const divCtx = document.getElementById("diversityChart").getContext("2d");
    this.diversityChart = new Chart(divCtx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Shannon Entropy",
            data: [],
            borderColor: "#22d3ee",
            backgroundColor: "rgba(34,211,238,0.1)",
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 2,
            yAxisID: "y",
          },
          {
            label: "Species (k-means)",
            data: [],
            borderColor: "#fbbf24",
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 1.5,
            fill: false,
            yAxisID: "y1",
          },
        ],
      },
      options: {
        ...baseOpts,
        plugins: { ...baseOpts.plugins, title: { display: true, text: "Genetic Diversity", color: "#aaa", font: { size: 11 } } },
        scales: {
          ...baseOpts.scales,
          y: { ...baseOpts.scales.y, position: "left", title: { display: true, text: "Shannon H", color: "#666", font: { size: 10 } } },
          y1: {
            position: "right",
            grid: { drawOnChartArea: false },
            ticks: { color: "#666", font: { size: 9 } },
            beginAtZero: true,
            title: { display: true, text: "Species", color: "#666", font: { size: 10 } },
          },
        },
      },
    });
  },

  // ── Record a generation snapshot ──────────────────────────
  recordGeneration(generation, population, traitPool) {
    const h = this.history;
    const alive = population.filter(b => b.alive);

    h.generations.push(generation);

    // Population
    const total = alive.length;
    const predators = alive.filter(b => b.traits.some(t => t.name === "Predatory Behavior")).length;
    const prey = total - predators;
    h.population.push(total);
    h.predators.push(predators);
    h.prey.push(prey);

    // Fitness
    const stats = Genetics.populationStats(alive);
    h.meanFitness.push(stats.meanFitness);
    h.maxFitness.push(stats.maxFitness);
    h.minFitness.push(stats.minFitness);
    h.stdFitness.push(stats.stdFitness);

    // Diversity
    h.diversity.push(stats.diversity);

    // Species via k-means
    if (alive.length > 0) {
      const assignments = Genetics.kMeansClustering(alive, 4, 8);
      const uniqueSpecies = new Set(assignments).size;
      h.speciesCount.push(uniqueSpecies);
      // Assign species to bacteria
      for (let i = 0; i < alive.length; i++) {
        alive[i].species = assignments[i];
      }
    } else {
      h.speciesCount.push(0);
    }

    // Trait frequencies
    const n = alive.length || 1;
    for (let i = 0; i < Math.min(Genetics.TRAIT_GENE_COUNT, traitPool.length); i++) {
      const tName = traitPool[i].name;
      if (!h.traitFrequencies[tName]) h.traitFrequencies[tName] = [];
      let count = 0;
      for (const b of alive) {
        if (b.genome[i] > 0.5) count++;
      }
      h.traitFrequencies[tName].push(count / n);
    }
  },

  // ── Update all charts with latest data ────────────────────
  updateCharts(traitPool) {
    const h = this.history;

    // Pop chart
    this.popChart.data.labels = h.generations;
    this.popChart.data.datasets[0].data = h.population;
    this.popChart.data.datasets[1].data = h.predators;
    this.popChart.data.datasets[2].data = h.prey;
    this.popChart.update("none");

    // Fitness chart
    this.fitnessChart.data.labels = h.generations;
    this.fitnessChart.data.datasets[0].data = h.meanFitness;
    this.fitnessChart.data.datasets[1].data = h.maxFitness;
    this.fitnessChart.data.datasets[2].data = h.minFitness;
    this.fitnessChart.update("none");

    // Trait chart — show top 10 traits by variance
    const traitEntries = Object.entries(h.traitFrequencies)
      .filter(([, arr]) => arr.length > 0)
      .map(([name, arr]) => {
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        const variance = arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
        return { name, arr, variance };
      })
      .sort((a, b) => b.variance - a.variance)
      .slice(0, 10);

    this.traitChart.data.labels = h.generations;
    this.traitChart.data.datasets = traitEntries.map((entry, idx) => {
      const traitObj = traitPool.find(t => t.name === entry.name);
      const color = traitObj ? traitObj.color : this.SPECIES_COLORS[idx % this.SPECIES_COLORS.length];
      return {
        label: entry.name,
        data: entry.arr,
        borderColor: color,
        backgroundColor: color + "18",
        fill: false,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 1.5,
      };
    });
    this.traitChart.update("none");

    // Diversity chart
    this.diversityChart.data.labels = h.generations;
    this.diversityChart.data.datasets[0].data = h.diversity;
    this.diversityChart.data.datasets[1].data = h.speciesCount;
    this.diversityChart.update("none");
  },

  // ── Reset ─────────────────────────────────────────────────
  reset() {
    this.history = {
      generations: [],
      population: [],
      predators: [],
      prey: [],
      meanFitness: [],
      maxFitness: [],
      minFitness: [],
      stdFitness: [],
      diversity: [],
      speciesCount: [],
      traitFrequencies: {},
    };
    if (this.popChart) {
      this.popChart.data.labels = [];
      this.popChart.data.datasets.forEach(d => d.data = []);
      this.popChart.update("none");
    }
    if (this.fitnessChart) {
      this.fitnessChart.data.labels = [];
      this.fitnessChart.data.datasets.forEach(d => d.data = []);
      this.fitnessChart.update("none");
    }
    if (this.traitChart) {
      this.traitChart.data.labels = [];
      this.traitChart.data.datasets = [];
      this.traitChart.update("none");
    }
    if (this.diversityChart) {
      this.diversityChart.data.labels = [];
      this.diversityChart.data.datasets.forEach(d => d.data = []);
      this.diversityChart.update("none");
    }
  },
};
