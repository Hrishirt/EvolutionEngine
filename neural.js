// ═══════════════════════════════════════════════════════════════
//  NEURAL NETWORK — Feedforward NN from scratch
//  Topology: 6 inputs → 8 hidden (tanh) → 2 outputs (tanh)
//  ~66 learnable parameters per bacterium brain.
// ═══════════════════════════════════════════════════════════════

class NeuralNetwork {
  /**
   * @param {number[]} topology — e.g. [6, 8, 2]
   * @param {Float64Array} [weights] — flat weight array to initialise from
   */
  constructor(topology = [6, 8, 2], weights = null) {
    this.topology = topology;
    this.weightCount = NeuralNetwork.countWeights(topology);

    if (weights) {
      this.weights = new Float64Array(weights);
    } else {
      // Xavier-ish random init
      this.weights = new Float64Array(this.weightCount);
      for (let i = 0; i < this.weightCount; i++) {
        this.weights[i] = (Math.random() * 2 - 1) * 0.8;
      }
    }
  }

  /**
   * Count total weights + biases for a given topology.
   */
  static countWeights(topology) {
    let count = 0;
    for (let l = 1; l < topology.length; l++) {
      // weights: prev_size * curr_size  +  biases: curr_size
      count += topology[l - 1] * topology[l] + topology[l];
    }
    return count;
  }

  /**
   * Forward pass — returns output array.
   * @param {number[]} inputs
   * @returns {number[]}
   */
  forward(inputs) {
    let activation = inputs.slice();
    let wi = 0; // weight index cursor

    for (let l = 1; l < this.topology.length; l++) {
      const prevSize = this.topology[l - 1];
      const currSize = this.topology[l];
      const next = new Array(currSize);

      for (let j = 0; j < currSize; j++) {
        let sum = 0;
        // Weighted sum
        for (let i = 0; i < prevSize; i++) {
          sum += activation[i] * this.weights[wi++];
        }
        // Bias
        sum += this.weights[wi++];
        // Activation: tanh
        next[j] = Math.tanh(sum);
      }

      activation = next;
    }

    return activation;
  }

  /**
   * Return a deep copy of this network.
   */
  copy() {
    return new NeuralNetwork(this.topology.slice(), new Float64Array(this.weights));
  }

  /**
   * Mutate weights in-place with Gaussian noise.
   * @param {number} rate — probability each weight is mutated (0-1)
   * @param {number} sigma — std-dev of Gaussian noise
   */
  mutate(rate = 0.1, sigma = 0.3) {
    for (let i = 0; i < this.weights.length; i++) {
      if (Math.random() < rate) {
        // Box-Muller transform for Gaussian
        this.weights[i] += NeuralNetwork.gaussianRandom() * sigma;
        // Clamp to prevent explosion
        this.weights[i] = Math.max(-3, Math.min(3, this.weights[i]));
      }
    }
  }

  /**
   * Uniform crossover — produce child from two parents.
   * @param {NeuralNetwork} parentA
   * @param {NeuralNetwork} parentB
   * @returns {NeuralNetwork}
   */
  static crossoverUniform(parentA, parentB) {
    const child = new Float64Array(parentA.weights.length);
    for (let i = 0; i < child.length; i++) {
      child[i] = Math.random() < 0.5 ? parentA.weights[i] : parentB.weights[i];
    }
    return new NeuralNetwork(parentA.topology.slice(), child);
  }

  /**
   * Single-point crossover.
   */
  static crossoverSinglePoint(parentA, parentB) {
    const len = parentA.weights.length;
    const point = Math.floor(Math.random() * len);
    const child = new Float64Array(len);
    for (let i = 0; i < len; i++) {
      child[i] = i < point ? parentA.weights[i] : parentB.weights[i];
    }
    return new NeuralNetwork(parentA.topology.slice(), child);
  }

  /**
   * Two-point crossover.
   */
  static crossoverTwoPoint(parentA, parentB) {
    const len = parentA.weights.length;
    let p1 = Math.floor(Math.random() * len);
    let p2 = Math.floor(Math.random() * len);
    if (p1 > p2) [p1, p2] = [p2, p1];
    const child = new Float64Array(len);
    for (let i = 0; i < len; i++) {
      child[i] = (i >= p1 && i < p2) ? parentB.weights[i] : parentA.weights[i];
    }
    return new NeuralNetwork(parentA.topology.slice(), child);
  }

  /**
   * Box-Muller Gaussian random (mean 0, std 1).
   */
  static gaussianRandom() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  /**
   * Get weights as flat array (for genome integration).
   */
  getWeights() {
    return new Float64Array(this.weights);
  }

  /**
   * Set weights from flat array.
   */
  setWeights(w) {
    this.weights = new Float64Array(w);
  }
}
