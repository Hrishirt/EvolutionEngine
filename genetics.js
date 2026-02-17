// ═══════════════════════════════════════════════════════════════
//  GENETICS — Formal Genetic Algorithm operators
//  Genome = Float64Array: 40 trait genes + NN weight genes
// ═══════════════════════════════════════════════════════════════

const TRAIT_GENE_COUNT = 40;    // one float per trait
const NN_TOPOLOGY = [6, 8, 2]; // must match neural.js
const NN_WEIGHT_COUNT = NeuralNetwork.countWeights(NN_TOPOLOGY);
const GENOME_LENGTH = TRAIT_GENE_COUNT + NN_WEIGHT_COUNT; // ~106

const Genetics = {

  TRAIT_GENE_COUNT,
  NN_WEIGHT_COUNT,
  GENOME_LENGTH,
  NN_TOPOLOGY,

  // ── Genome creation ──────────────────────────────────────
  /**
   * Create a random genome.
   * Trait genes: uniform [0,1].  NN weights: Xavier-ish [-0.8, 0.8].
   */
  createGenome() {
    const g = new Float64Array(GENOME_LENGTH);
    // Trait genes
    for (let i = 0; i < TRAIT_GENE_COUNT; i++) {
      g[i] = Math.random();
    }
    // NN weight genes
    for (let i = TRAIT_GENE_COUNT; i < GENOME_LENGTH; i++) {
      g[i] = (Math.random() * 2 - 1) * 0.8;
    }
    return g;
  },

  /**
   * Extract expressed traits from genome (gene > 0.5 = expressed).
   * @param {Float64Array} genome
   * @param {object[]} traitPool — the TRAIT_POOL constant
   * @returns {object[]} array of expressed trait objects
   */
  expressTraits(genome, traitPool) {
    const expressed = [];
    for (let i = 0; i < TRAIT_GENE_COUNT && i < traitPool.length; i++) {
      if (genome[i] > 0.5) {
        expressed.push(traitPool[i]);
      }
    }
    return expressed;
  },

  /**
   * Build a NeuralNetwork from the NN portion of the genome.
   */
  buildBrain(genome) {
    const nnWeights = genome.slice(TRAIT_GENE_COUNT, GENOME_LENGTH);
    return new NeuralNetwork(NN_TOPOLOGY, nnWeights);
  },

  /**
   * Write NN weights back into genome (after mutation of brain directly).
   */
  writeBrainToGenome(genome, brain) {
    const w = brain.getWeights();
    for (let i = 0; i < NN_WEIGHT_COUNT; i++) {
      genome[TRAIT_GENE_COUNT + i] = w[i];
    }
  },

  // ── Fitness ──────────────────────────────────────────────
  /**
   * Compute fitness for a bacterium.
   */
  computeFitness(bacterium) {
    const e = Math.max(0, bacterium.energyCollected || 0);
    const s = Math.max(0, bacterium.survivalTime || 0);
    const f = Math.max(0, bacterium.foodEaten || 0);
    return e * 0.4 + s * 0.3 + f * 0.3;
  },

  // ── Selection ────────────────────────────────────────────
  /**
   * Tournament selection — pick k random, return the fittest.
   */
  tournamentSelect(population, k = 3) {
    let best = null;
    for (let i = 0; i < k; i++) {
      const candidate = population[Math.floor(Math.random() * population.length)];
      if (!best || candidate.fitness > best.fitness) {
        best = candidate;
      }
    }
    return best;
  },

  /**
   * Roulette wheel selection (fitness-proportionate).
   */
  rouletteSelect(population) {
    const totalFitness = population.reduce((s, b) => s + Math.max(0.01, b.fitness), 0);
    let r = Math.random() * totalFitness;
    for (const b of population) {
      r -= Math.max(0.01, b.fitness);
      if (r <= 0) return b;
    }
    return population[population.length - 1];
  },

  // ── Crossover operators ──────────────────────────────────
  /**
   * Uniform crossover on genomes.
   */
  crossoverUniform(genomeA, genomeB) {
    const child = new Float64Array(GENOME_LENGTH);
    for (let i = 0; i < GENOME_LENGTH; i++) {
      child[i] = Math.random() < 0.5 ? genomeA[i] : genomeB[i];
    }
    return child;
  },

  /**
   * Single-point crossover.
   */
  crossoverSinglePoint(genomeA, genomeB) {
    const point = Math.floor(Math.random() * GENOME_LENGTH);
    const child = new Float64Array(GENOME_LENGTH);
    for (let i = 0; i < GENOME_LENGTH; i++) {
      child[i] = i < point ? genomeA[i] : genomeB[i];
    }
    return child;
  },

  /**
   * Two-point crossover.
   */
  crossoverTwoPoint(genomeA, genomeB) {
    let p1 = Math.floor(Math.random() * GENOME_LENGTH);
    let p2 = Math.floor(Math.random() * GENOME_LENGTH);
    if (p1 > p2) [p1, p2] = [p2, p1];
    const child = new Float64Array(GENOME_LENGTH);
    for (let i = 0; i < GENOME_LENGTH; i++) {
      child[i] = (i >= p1 && i < p2) ? genomeB[i] : genomeA[i];
    }
    return child;
  },

  /**
   * Dispatch crossover by name.
   */
  crossover(genomeA, genomeB, method = "uniform") {
    switch (method) {
      case "single": return Genetics.crossoverSinglePoint(genomeA, genomeB);
      case "two":    return Genetics.crossoverTwoPoint(genomeA, genomeB);
      default:       return Genetics.crossoverUniform(genomeA, genomeB);
    }
  },

  // ── Mutation ─────────────────────────────────────────────
  /**
   * Gaussian mutation on entire genome.
   * @param {Float64Array} genome
   * @param {number} rate — probability per gene
   * @param {number} sigma — std-dev of perturbation
   */
  mutate(genome, rate = 0.05, sigma = 0.2) {
    for (let i = 0; i < GENOME_LENGTH; i++) {
      if (Math.random() < rate) {
        genome[i] += NeuralNetwork.gaussianRandom() * sigma;
        // Clamp trait genes to [0, 1]
        if (i < TRAIT_GENE_COUNT) {
          genome[i] = Math.max(0, Math.min(1, genome[i]));
        } else {
          // Clamp NN weights to [-3, 3]
          genome[i] = Math.max(-3, Math.min(3, genome[i]));
        }
      }
    }
    return genome;
  },

  // ── Elitism ──────────────────────────────────────────────
  /**
   * Return the top n individuals sorted by fitness (descending).
   */
  getElites(population, n) {
    return [...population]
      .sort((a, b) => b.fitness - a.fitness)
      .slice(0, n);
  },

  // ── Statistics ───────────────────────────────────────────
  /**
   * Shannon diversity index over trait frequencies.
   */
  shannonDiversity(population, traitCount = TRAIT_GENE_COUNT) {
    if (population.length === 0) return 0;
    const freq = new Float64Array(traitCount);
    for (const b of population) {
      for (let i = 0; i < traitCount; i++) {
        if (b.genome[i] > 0.5) freq[i]++;
      }
    }
    let H = 0;
    const n = population.length;
    for (let i = 0; i < traitCount; i++) {
      const p = freq[i] / n;
      if (p > 0 && p < 1) {
        H -= p * Math.log(p);
      }
    }
    return H;
  },

  /**
   * Simple k-means clustering on genome vectors.
   * Returns cluster assignments array.
   */
  kMeansClustering(population, k = 4, iterations = 10) {
    if (population.length === 0) return [];
    const n = population.length;
    const dim = GENOME_LENGTH;

    // Initialise centroids from random individuals
    const centroids = [];
    const used = new Set();
    for (let c = 0; c < k; c++) {
      let idx;
      do { idx = Math.floor(Math.random() * n); } while (used.has(idx) && used.size < n);
      used.add(idx);
      centroids.push(new Float64Array(population[idx].genome));
    }

    const assignments = new Int32Array(n);

    for (let iter = 0; iter < iterations; iter++) {
      // Assign each point to nearest centroid
      for (let i = 0; i < n; i++) {
        let bestDist = Infinity;
        let bestC = 0;
        for (let c = 0; c < k; c++) {
          let dist = 0;
          for (let d = 0; d < dim; d++) {
            const diff = population[i].genome[d] - centroids[c][d];
            dist += diff * diff;
          }
          if (dist < bestDist) {
            bestDist = dist;
            bestC = c;
          }
        }
        assignments[i] = bestC;
      }

      // Recompute centroids
      const counts = new Float64Array(k);
      const sums = [];
      for (let c = 0; c < k; c++) sums.push(new Float64Array(dim));

      for (let i = 0; i < n; i++) {
        const c = assignments[i];
        counts[c]++;
        for (let d = 0; d < dim; d++) {
          sums[c][d] += population[i].genome[d];
        }
      }

      for (let c = 0; c < k; c++) {
        if (counts[c] > 0) {
          for (let d = 0; d < dim; d++) {
            centroids[c][d] = sums[c][d] / counts[c];
          }
        }
      }
    }

    return assignments;
  },

  /**
   * Compute Hardy-Weinberg expected vs observed frequencies.
   * Returns array of { trait, observed, expected, departure } for each trait.
   */
  hardyWeinberg(population, traitPool) {
    if (population.length === 0) return [];
    const n = population.length;
    const results = [];

    for (let i = 0; i < Math.min(TRAIT_GENE_COUNT, traitPool.length); i++) {
      let count = 0;
      for (const b of population) {
        if (b.genome[i] > 0.5) count++;
      }
      const p = count / n;           // frequency of expressed allele
      const q = 1 - p;               // frequency of non-expressed
      const observed = p;
      const expectedHet = 2 * p * q; // expected heterozygosity
      const departure = Math.abs(p * p + 2 * p * q + q * q - 1); // always ~0 for validation
      results.push({
        trait: traitPool[i].name,
        observed: p,
        expectedHet,
        departure: Math.abs(observed - 0.5) // simple deviation from equilibrium
      });
    }

    return results;
  },

  /**
   * Population-level statistics summary.
   */
  populationStats(population) {
    if (population.length === 0) {
      return { meanFitness: 0, maxFitness: 0, minFitness: 0, stdFitness: 0, diversity: 0 };
    }
    const fitnesses = population.map(b => b.fitness);
    const mean = fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length;
    const max = Math.max(...fitnesses);
    const min = Math.min(...fitnesses);
    const variance = fitnesses.reduce((s, f) => s + (f - mean) ** 2, 0) / fitnesses.length;
    const std = Math.sqrt(variance);
    const diversity = Genetics.shannonDiversity(population);

    return { meanFitness: mean, maxFitness: max, minFitness: min, stdFitness: std, diversity };
  }
};
