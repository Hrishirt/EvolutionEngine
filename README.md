# ML Evolution Simulation Engine

A real-time evolution simulation that models hundreds of bacteria with neural-network-driven behavior, genetic algorithms, and a full ecosystem — rendered on an HTML5 canvas with a Python ML backend.

![Python](https://img.shields.io/badge/Python-FastAPI-009688?style=flat-square)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?style=flat-square)
![ML](https://img.shields.io/badge/ML-NumPy%20%7C%20scikit--learn-blue?style=flat-square)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [The Genome](#the-genome)
- [Neural Network](#neural-network)
- [Genetic Algorithm](#genetic-algorithm)
- [Ecosystem Dynamics](#ecosystem-dynamics)
- [Evolution Mechanics](#evolution-mechanics)
- [Analytics & Statistics](#analytics--statistics)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Controls](#controls)

---

## Overview

Each bacterium is an autonomous agent with a **106-dimensional genome** that encodes both its biological traits (40 genes) and the weights of a small feedforward neural network (66 weights). There is no hand-coded behavior — every movement decision comes from the neural network, and the only way a bacterium improves is by surviving long enough for its genome to be passed on.

The simulation models:
- **Natural selection** via environmental pressure (food scarcity, hazard zones, predation)
- **Reproduction** via asexual budding and generational genetic algorithms
- **Mutation** through Gaussian noise, spontaneous de novo mutation, and horizontal gene transfer
- **Speciation** tracked in real time with k-means clustering
- **Population genetics** monitored via Shannon diversity and Hardy-Weinberg equilibrium analysis

---

## Architecture

The project is split into a **JavaScript frontend** and a **Python backend**, each handling what it does best:

| Layer | Technology | Responsibility |
|-------|-----------|----------------|
| **Rendering & real-time logic** | JavaScript (Canvas API) | 60fps animation, per-frame NN forward pass, ecosystem tick, UI |
| **ML / GA computations** | Python (FastAPI + NumPy + scikit-learn) | Genome creation, GA reproduction, k-means clustering, batch statistics |
| **Communication** | REST API (`api.js` client) | JSON payloads at generation boundaries |

The JS frontend runs the real-time simulation loop. At generation boundaries (selection events), it sends population data to the Python backend for heavy computations (crossover, mutation, clustering) and receives offspring genomes back. If the Python server is unavailable, the JS side has a full fallback implementation — the simulation works standalone.

---

## The Genome

Every bacterium carries a `Float64Array` of length **106**:

```
[0 .............. 39 | 40 .............. 105]
 ← 40 trait genes →   ← 66 NN weights →
```

### Trait Genes (indices 0–39)

Each of the 40 genes is a float in `[0, 1]`. A gene is **expressed** if its value exceeds `0.5`, giving the bacterium that trait. There are 40 traits in the pool including:

- **Environmental resistances**: Heat Resistance, Cold Tolerance, UV Shield, Acid Resistance, Radiation Resistance, Pressure Resistance, etc.
- **Metabolic traits**: Fast Metabolism (1.8x energy burn), Slow Metabolism (0.5x energy burn), Photosynthesis
- **Behavioral traits**: Predatory Behavior (can hunt other bacteria), Chemotaxis, Flagella Motility
- **Survival traits**: Spore Formation, Endospore Armor (extends lifespan), Biofilm Producer, Capsule Coating
- **Genetic traits**: DNA Repair (reduces mutation rate to 30%), Gene Transfer (enables horizontal gene transfer)
- **Reproduction**: Rapid Division (increases budding chance)

The threshold-based expression (`gene > 0.5`) means traits can be gained or lost through mutation — a gene drifting from 0.48 to 0.52 will suddenly express, modeling a phenotypic switch.

### Neural Network Weights (indices 40–105)

The remaining 66 floats encode the weights and biases of the bacterium's brain (see [Neural Network](#neural-network) below).

---

## Neural Network

Every bacterium has a small **feedforward neural network** with topology `[6, 8, 2]`:

```
Input Layer (6 neurons)     Hidden Layer (8 neurons)     Output Layer (2 neurons)
─────────────────────       ─────────────────────        ─────────────────────
 Food Angle      ──┐         ┌── h1 ──┐                    ┌── dx (velocity x)
 Food Distance   ──┤         ├── h2 ──┤                    └── dy (velocity y)
 Threat Angle    ──┼────────►├── h3 ──┼──────────────────►
 Threat Distance ──┤         ├── h4 ──┤
 Energy Level    ──┤         ├── h5 ──┤
 Current Speed   ──┘         ├── h6 ──┤
                             ├── h7 ──┤
                             └── h8 ──┘
```

**Weight count**: `(6 x 8) + 8 + (8 x 2) + 2 = 48 + 8 + 16 + 2 = **66 weights + biases**`

### Why no backpropagation?

Traditional supervised learning requires labeled training data — "given this input, produce this output." But bacteria don't have a teacher. They receive **sparse, delayed feedback**: a bacterium that moved well might eat food 30 frames later, or die 200 frames later from starvation. The reward signal is noisy and delayed.

Instead, this simulation uses **neuroevolution**: the neural network weights are embedded directly in the genome and optimized through the genetic algorithm. Bacteria that survive longer and eat more food have higher fitness and are more likely to pass their brain weights to the next generation. Over many generations, the population collectively "learns" to navigate toward food and away from threats — without a single gradient computation.

### Activation Function

All neurons use **tanh** activation, which outputs values in `[-1, 1]` — ideal for directional movement decisions.

### Inputs (computed per frame)

The 6 inputs are computed from the bacterium's local environment:

1. **Food Angle**: `atan2(food.y - b.y, food.x - b.x) / pi` — direction to nearest food, normalized to `[-1, 1]`
2. **Food Distance**: `min(dist / maxDist, 1)` — distance to nearest food, normalized
3. **Threat Angle**: direction to nearest predator or unresisted hazard zone
4. **Threat Distance**: distance to nearest threat, normalized
5. **Energy Level**: `energy / maxEnergy` — current energy as fraction of maximum
6. **Current Speed**: `sqrt(vx^2 + vy^2) / 2` — current velocity magnitude

### Initialization

Weights are initialized using **Xavier-like initialization**: uniform random in `[-0.8, 0.8]`, which helps prevent saturation of tanh neurons at the start.

---

## Genetic Algorithm

The GA operates at generation boundaries (after selection events or when reproduction is triggered). It implements a full evolutionary pipeline:

### 1. Fitness Computation

Fitness is a weighted combination of three survival metrics:

```
fitness = 0.4 * energy_collected + 0.3 * survival_time + 0.3 * food_eaten
```

This multi-objective fitness rewards both efficiency (energy per unit time) and longevity.

### 2. Parent Selection

Two methods are available:

- **Tournament Selection** (default): Pick `k` random individuals, select the fittest. Tournament size `k` controls selection pressure — higher `k` means stronger pressure toward the best.
- **Roulette Wheel Selection**: Probability of selection is proportional to fitness. A bacterium with fitness 10 is twice as likely to be chosen as one with fitness 5.

### 3. Crossover

Three crossover operators are implemented to combine parent genomes:

- **Uniform Crossover** (default): Each gene independently chosen from parent A or B with 50/50 probability. Maximizes recombination.
- **Single-Point Crossover**: A random cut point splits the genome; child gets genes from parent A before the cut and parent B after. Preserves gene linkage.
- **Two-Point Crossover**: Two cut points define a segment from parent B inserted into parent A's genome. Balances recombination and linkage.

### 4. Gaussian Mutation

After crossover, each gene in the offspring genome is mutated with probability `rate` (default 5%):

```
if random() < rate:
    gene += gaussian(mean=0, std=sigma)
```

The Gaussian noise is generated using the **Box-Muller transform**:

```
z = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)     where u1, u2 ~ Uniform(0, 1)
```

This produces normally distributed perturbations — most mutations are small, but occasionally a large mutation occurs, allowing the population to escape local optima.

- Trait genes are clamped to `[0, 1]`
- NN weights are clamped to `[-3, 3]` to prevent weight explosion

### 5. Elitism

The top N% of the population (by fitness) are copied directly into the next generation without modification. This ensures the best solutions are never lost. Default is 10%.

---

## Ecosystem Dynamics

The ecosystem creates environmental pressure that drives natural selection between generation events.

### Energy Economy

The simulation targets a stable equilibrium around ~200 bacteria. The math:

```
Energy drain/sec = population * metabolicCost * 60fps = 200 * 0.04 * 60 = 480/sec
Energy input/sec = foodSpawnRate * 60fps * foodEnergy  = 0.6 * 60 * 25  = 900/sec
```

At ~60% food utilization efficiency, this gives roughly 540 energy in vs 480 energy out — a slightly positive balance that sustains the population.

### Food System

- Green particles spawn at a configurable rate (default 0.6/frame)
- Each food particle gives 25 energy on contact
- Maximum food on screen is capped (default 150) to create scarcity at high populations

### Hazard Zones

Colored circular zones drift across the canvas, dealing damage to bacteria that lack the corresponding resistance trait:

| Zone | Resistance Trait | Color |
|------|-----------------|-------|
| Heat Zone | Heat Resistance | Red |
| Cold Zone | Cold Tolerance | Blue |
| UV Zone | UV Shield | Yellow |
| Acid Zone | Acid Resistance | Green |
| Radiation Zone | Radiation Resistance | Gold |
| Pressure Zone | Pressure Resistance | Teal |
| Metal Zone | Metal Resistance | Silver |
| Salt Zone | Salt Tolerance | Pink |

### Predator-Prey

Bacteria with the **Predatory Behavior** trait can consume other bacteria within a catch range (`size * 2.5` pixels), gaining 40 energy per kill. Predators cannot eat other predators.

### Overcrowding Pressure

When population exceeds the overcrowding threshold (180), metabolic cost scales up:

```
overcrowdingFactor = 1 + (pop - threshold) / threshold * (multiplier - 1)
effectiveMetabolicCost = baseCost * overcrowdingFactor
```

This creates negative density-dependent regulation — as the population grows, resources become scarcer per capita, naturally limiting growth.

### Death Causes

Bacteria can die from:
- **Starvation**: Energy reaches 0
- **Predation**: Caught by a predator bacterium
- **Hazard damage**: Accumulated damage from unresisted hazard zones
- **Selection events**: Failed a natural selection check (see below)

---

## Evolution Mechanics

### Asexual Budding

Bacteria reproduce asexually when conditions are met:
- Energy above threshold (100)
- Random chance per tick (0.6%)
- Population below hard cap (600)
- Bacteria with the **Rapid Division** trait have 2x budding chance

On budding, the parent's energy is split 60/40 with the offspring. The offspring genome is a copy of the parent's, with Gaussian mutation applied — modeling real bacterial binary fission with copy errors.

### Spontaneous Mutation (De Novo)

Modeled as a **Poisson process** where each bacterium has a small probability per tick of a random unexpressed gene flipping to expressed:

```
P(mutation in 1 tick) = 0.0002  (~1.2% per second at 60fps)
```

Bacteria with the **DNA Repair** trait have this rate reduced to 30% of the base rate, modeling the real biological function of DNA repair enzymes.

### Horizontal Gene Transfer (HGT)

Bacteria with the **Gene Transfer** trait can pass expressed genes to nearby bacteria (within 30 pixels). Per tick, there's a 0.1% chance of transfer per eligible donor-recipient pair. The recipient gains the trait by having the corresponding gene value boosted above the expression threshold.

This models real bacterial conjugation — the primary mechanism by which antibiotic resistance spreads in nature.

### Selection Events

Triggered manually via the UI button, simulating a sudden environmental shift. A random trait is chosen, and every bacterium's survival is determined probabilistically:

```
survival_probability = genome[trait_index]    // gene value is directly the survival chance
```

A bacterium with gene value 0.9 has a 90% chance of surviving; one with 0.1 has only 10%. This is more realistic than a binary alive/dead threshold.

A **minimum survivor floor** (15% of population or 10 bacteria, whichever is greater) prevents total extinction — even in the worst case, a remnant population survives to rebuild.

---

## Analytics & Statistics

The real-time analytics panel tracks four charts powered by Chart.js:

### Population Dynamics
Tracks total population, predator count, and prey count over generations.

### Fitness Over Generations
Plots mean, max, and min fitness across the population — shows whether the GA is converging.

### Shannon Diversity Index

Measures genetic diversity using information entropy:

```
H = -sum(p_i * ln(p_i))    for each trait where 0 < p_i < 1
```

Where `p_i` is the frequency of trait `i` being expressed in the population. Higher H means more diverse gene pool; a drop in H after a selection event indicates that the population is becoming more genetically uniform (a bottleneck).

### K-Means Species Clustering

Uses **scikit-learn's KMeans** on the full 106-dimensional genome vectors to identify distinct genetic clusters (species). The algorithm:

1. Initialize k=4 centroids from random individuals
2. Assign each bacterium to nearest centroid (Euclidean distance in genome space)
3. Recompute centroids as the mean of assigned bacteria
4. Repeat for 8-20 iterations

The number of occupied clusters indicates how many distinct "species" exist. If all bacteria cluster into 1-2 groups, the population is becoming homogeneous.

### Hardy-Weinberg Equilibrium

For each trait, compares observed allele frequency against the expected equilibrium frequency:

```
departure = |observed_frequency - 0.5|
```

Large departures indicate that selection is actively pushing that trait's frequency away from equilibrium — a signal that the trait is under strong selective pressure (either positive or negative).

### Live Event Feed

A real-time notification overlay displays ecosystem events as they happen:
- Predator hunts
- Starvation deaths
- Hazard zone kills
- Budding events
- Selection event results
- Mutation and HGT events

Events have a cooldown (1.5s per type) to prevent spam.

---

## Getting Started

### Prerequisites

- Python 3.10+
- A modern web browser

### Installation

```bash
cd evolution-sim
pip install -r requirements.txt
```

### Running

```bash
python server.py
```

The server starts on `http://localhost:8000`. Open that URL in your browser — the frontend is served automatically.

The Python backend handles ML computations (GA, clustering, statistics). If you just want to see the simulation without the backend, you can open `index.html` directly — the JS fallback will handle everything locally.

---

## Project Structure

```
evolution-sim/
├── server.py          # FastAPI backend — GA, k-means, stats (NumPy + scikit-learn)
├── index.html         # Frontend UI — canvas, controls, dashboard, analytics panel
├── sim.js             # Main simulation loop — orchestrates all modules
├── neural.js          # Feedforward neural network (JS) — per-frame forward pass
├── genetics.js        # Genome, crossover, mutation, fitness, diversity (JS fallback)
├── ecosystem.js       # Food, hazards, predation, energy economy
├── analytics.js       # Chart.js integration — population, fitness, diversity charts
├── api.js             # REST client — communicates with Python backend
└── requirements.txt   # Python dependencies (FastAPI, uvicorn, NumPy, scikit-learn)
```

---

## Controls

| Control | Description |
|---------|-------------|
| **Trigger Selection Event** | Randomly selects a trait and probabilistically kills bacteria lacking it |
| **Reset** | Reinitializes the entire population |
| **Pause** | Pauses/resumes the simulation |
| **Speed (1x/2x/4x)** | Simulation speed multiplier |
| **Population Size** | Initial number of bacteria (50–500) |
| **Traits per Bacterium** | Number of traits each bacterium starts with |
| **Mutation Rate** | Probability each gene is mutated during GA reproduction |
| **Mutation Sigma** | Standard deviation of Gaussian mutation noise |
| **Elitism %** | Percentage of top individuals preserved unchanged |
| **Crossover Method** | Uniform / Single-point / Two-point |
| **Tournament Size** | Number of candidates in tournament selection |
| **Food Spawn Rate** | Food particles spawned per frame |
| **Max Food** | Maximum food particles on screen |
| **Metabolic Cost** | Energy drained per tick per bacterium |

Click any bacterium to inspect it — view its genome visualization, expressed traits, neural network diagram, and real-time stats.
