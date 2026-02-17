"""
ML Evolution Simulation — Python Backend
=========================================
FastAPI server handling all ML/GA computations:
  - Genome creation (NumPy)
  - Genetic Algorithm: crossover, mutation, selection (NumPy)
  - K-means species clustering (scikit-learn)
  - Statistics: Shannon entropy, Hardy-Weinberg, fitness (NumPy)
  - Neural network forward pass (NumPy — for batch prediction)

The JavaScript frontend handles rendering + per-frame NN forward pass.
This server is called at generation boundaries for heavy ML operations.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.cluster import KMeans

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

TRAIT_GENE_COUNT = 40
NN_TOPOLOGY = [6, 8, 2]  # 6 inputs → 8 hidden → 2 outputs


def count_nn_weights(topology: list[int]) -> int:
    """Count total weights + biases for a feedforward NN topology."""
    total = 0
    for i in range(1, len(topology)):
        total += topology[i - 1] * topology[i] + topology[i]
    return total


NN_WEIGHT_COUNT = count_nn_weights(NN_TOPOLOGY)
GENOME_LENGTH = TRAIT_GENE_COUNT + NN_WEIGHT_COUNT  # 40 + 66 = 106


# ═══════════════════════════════════════════════════════════════
#  NEURAL NETWORK (NumPy — for batch operations)
# ═══════════════════════════════════════════════════════════════

class NeuralNetNumpy:
    """Feedforward NN using NumPy for vectorised batch operations."""

    def __init__(self, topology: list[int] = NN_TOPOLOGY):
        self.topology = topology

    def forward_batch(self, weights_batch: np.ndarray, inputs_batch: np.ndarray) -> np.ndarray:
        """
        Batch forward pass for N individuals.
        weights_batch: (N, NN_WEIGHT_COUNT)
        inputs_batch:  (N, input_size)
        Returns:       (N, output_size)
        """
        activation = inputs_batch.copy()  # (N, 6)
        wi = 0

        for l in range(1, len(self.topology)):
            prev_size = self.topology[l - 1]
            curr_size = self.topology[l]

            # Extract weights: (N, prev_size, curr_size)
            w_count = prev_size * curr_size
            W = weights_batch[:, wi:wi + w_count].reshape(-1, prev_size, curr_size)
            wi += w_count

            # Extract biases: (N, curr_size)
            B = weights_batch[:, wi:wi + curr_size]
            wi += curr_size

            # Matrix multiply: (N, prev_size) @ (N, prev_size, curr_size) → (N, curr_size)
            activation = np.tanh(np.einsum("ni,nij->nj", activation, W) + B)

        return activation

    def forward_single(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Single forward pass."""
        return self.forward_batch(weights.reshape(1, -1), inputs.reshape(1, -1))[0]


nn_engine = NeuralNetNumpy()


# ═══════════════════════════════════════════════════════════════
#  GENETIC ALGORITHM (NumPy)
# ═══════════════════════════════════════════════════════════════

def create_genomes(n: int, traits_per_bacterium: int = 4) -> np.ndarray:
    """Create n random genomes as (n, GENOME_LENGTH) array."""
    genomes = np.zeros((n, GENOME_LENGTH))

    # Trait genes: bias so roughly `traits_per_bacterium` are expressed
    for i in range(n):
        indices = np.random.permutation(TRAIT_GENE_COUNT)
        # First `traits_per_bacterium` indices get high values
        genomes[i, indices[:traits_per_bacterium]] = np.random.uniform(0.5, 1.0, traits_per_bacterium)
        # Rest get low values
        remaining = TRAIT_GENE_COUNT - traits_per_bacterium
        if remaining > 0:
            genomes[i, indices[traits_per_bacterium:]] = np.random.uniform(0.0, 0.45, remaining)

    # NN weight genes: Xavier-ish init
    genomes[:, TRAIT_GENE_COUNT:] = np.random.uniform(-0.8, 0.8, (n, NN_WEIGHT_COUNT))

    return genomes


def compute_fitness(energy_collected: np.ndarray,
                    survival_time: np.ndarray,
                    food_eaten: np.ndarray) -> np.ndarray:
    """Vectorised fitness: f = 0.4*energy + 0.3*survival + 0.3*food."""
    return (np.maximum(0, energy_collected) * 0.4 +
            np.maximum(0, survival_time) * 0.3 +
            np.maximum(0, food_eaten) * 0.3)


def tournament_select(fitnesses: np.ndarray, k: int = 3) -> int:
    """Tournament selection — pick k random indices, return the fittest."""
    candidates = np.random.randint(0, len(fitnesses), size=k)
    return int(candidates[np.argmax(fitnesses[candidates])])


def crossover_uniform(genome_a: np.ndarray, genome_b: np.ndarray) -> np.ndarray:
    """Uniform crossover — each gene randomly from parent A or B."""
    mask = np.random.random(GENOME_LENGTH) < 0.5
    child = np.where(mask, genome_a, genome_b)
    return child


def crossover_single_point(genome_a: np.ndarray, genome_b: np.ndarray) -> np.ndarray:
    """Single-point crossover."""
    point = np.random.randint(0, GENOME_LENGTH)
    child = np.concatenate([genome_a[:point], genome_b[point:]])
    return child


def crossover_two_point(genome_a: np.ndarray, genome_b: np.ndarray) -> np.ndarray:
    """Two-point crossover."""
    p1, p2 = sorted(np.random.randint(0, GENOME_LENGTH, size=2))
    child = genome_a.copy()
    child[p1:p2] = genome_b[p1:p2]
    return child


def crossover(genome_a: np.ndarray, genome_b: np.ndarray, method: str = "uniform") -> np.ndarray:
    """Dispatch crossover by method name."""
    if method == "single":
        return crossover_single_point(genome_a, genome_b)
    elif method == "two":
        return crossover_two_point(genome_a, genome_b)
    else:
        return crossover_uniform(genome_a, genome_b)


def mutate(genome: np.ndarray, rate: float = 0.05, sigma: float = 0.2) -> np.ndarray:
    """Gaussian mutation — each gene mutated with probability `rate`."""
    mask = np.random.random(GENOME_LENGTH) < rate
    noise = np.random.normal(0, sigma, GENOME_LENGTH) * mask

    genome = genome + noise

    # Clamp trait genes to [0, 1]
    genome[:TRAIT_GENE_COUNT] = np.clip(genome[:TRAIT_GENE_COUNT], 0.0, 1.0)
    # Clamp NN weights to [-3, 3]
    genome[TRAIT_GENE_COUNT:] = np.clip(genome[TRAIT_GENE_COUNT:], -3.0, 3.0)

    return genome


def reproduce(survivor_genomes: np.ndarray,
              fitnesses: np.ndarray,
              target_pop: int,
              config: dict) -> np.ndarray:
    """
    Full GA reproduction cycle:
    1. Elitism — keep top N%
    2. Tournament selection → crossover → mutation for remaining
    Returns offspring genomes (n, GENOME_LENGTH).
    """
    n_survivors = len(survivor_genomes)
    if n_survivors == 0:
        return create_genomes(target_pop)

    mutation_rate = config.get("mutation_rate", 0.05)
    mutation_sigma = config.get("mutation_sigma", 0.2)
    elitism_pct = config.get("elitism_pct", 10)
    crossover_method = config.get("crossover_method", "uniform")
    tournament_size = config.get("tournament_size", 3)

    needed = max(0, target_pop - n_survivors)

    # Elitism — top N% indices
    elite_count = max(1, int(n_survivors * elitism_pct / 100))
    elite_indices = np.argsort(fitnesses)[::-1][:elite_count]

    # Breed offspring
    offspring = np.zeros((needed, GENOME_LENGTH))
    for i in range(needed):
        parent_a_idx = tournament_select(fitnesses, tournament_size)
        parent_b_idx = tournament_select(fitnesses, tournament_size)

        child = crossover(
            survivor_genomes[parent_a_idx],
            survivor_genomes[parent_b_idx],
            crossover_method,
        )
        child = mutate(child, mutation_rate, mutation_sigma)
        offspring[i] = child

    return offspring


# ═══════════════════════════════════════════════════════════════
#  STATISTICS (NumPy + scikit-learn)
# ═══════════════════════════════════════════════════════════════

def shannon_diversity(genomes: np.ndarray) -> float:
    """Shannon diversity index over trait gene frequencies."""
    if len(genomes) == 0:
        return 0.0
    n = len(genomes)
    trait_genes = genomes[:, :TRAIT_GENE_COUNT]
    expressed = (trait_genes > 0.5).astype(float)
    freq = expressed.mean(axis=0)  # frequency of each trait

    # H = -sum(p * ln(p)) for p in (0, 1)
    H = 0.0
    for p in freq:
        if 0 < p < 1:
            H -= p * math.log(p)
    return float(H)


def kmeans_species(genomes: np.ndarray, k: int = 4) -> tuple[list[int], int]:
    """
    K-means clustering on genome vectors using scikit-learn.
    Returns (assignments, n_unique_species).
    """
    if len(genomes) < k:
        return list(range(len(genomes))), len(genomes)

    kmeans = KMeans(n_clusters=k, n_init=3, max_iter=20, random_state=None)
    labels = kmeans.fit_predict(genomes)
    n_species = len(set(labels))
    return labels.tolist(), n_species


def hardy_weinberg(genomes: np.ndarray) -> list[dict]:
    """
    Hardy-Weinberg equilibrium departure for each trait.
    Returns list of {trait_index, observed_freq, departure}.
    """
    if len(genomes) == 0:
        return []

    n = len(genomes)
    results = []
    trait_genes = genomes[:, :TRAIT_GENE_COUNT]

    for i in range(TRAIT_GENE_COUNT):
        count = np.sum(trait_genes[:, i] > 0.5)
        p = count / n
        departure = abs(p - 0.5)  # deviation from equilibrium
        results.append({
            "trait_index": i,
            "observed": round(float(p), 4),
            "expected_het": round(float(2 * p * (1 - p)), 4),
            "departure": round(float(departure), 4),
        })

    return results


def population_stats(genomes: np.ndarray, fitnesses: np.ndarray) -> dict:
    """Compute population-level statistics."""
    if len(fitnesses) == 0:
        return {
            "mean_fitness": 0, "max_fitness": 0, "min_fitness": 0,
            "std_fitness": 0, "diversity": 0,
        }
    return {
        "mean_fitness": round(float(np.mean(fitnesses)), 4),
        "max_fitness": round(float(np.max(fitnesses)), 4),
        "min_fitness": round(float(np.min(fitnesses)), 4),
        "std_fitness": round(float(np.std(fitnesses)), 4),
        "diversity": round(shannon_diversity(genomes), 4),
    }


def trait_frequencies(genomes: np.ndarray) -> list[float]:
    """Frequency of each trait in the population."""
    if len(genomes) == 0:
        return [0.0] * TRAIT_GENE_COUNT
    trait_genes = genomes[:, :TRAIT_GENE_COUNT]
    freq = (trait_genes > 0.5).astype(float).mean(axis=0)
    return [round(float(f), 4) for f in freq]


# ═══════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════

class InitRequest(BaseModel):
    pop_size: int = 200
    traits_per_bacterium: int = 4


class InitResponse(BaseModel):
    genomes: list[list[float]]
    genome_length: int
    trait_gene_count: int
    nn_weight_count: int
    nn_topology: list[int]


class SurvivorData(BaseModel):
    genome: list[float]
    fitness: float


class ReproduceRequest(BaseModel):
    survivors: list[SurvivorData]
    target_pop: int = 200
    mutation_rate: float = 0.05
    mutation_sigma: float = 0.2
    elitism_pct: int = 10
    crossover_method: str = "uniform"
    tournament_size: int = 3


class ReproduceResponse(BaseModel):
    offspring_genomes: list[list[float]]
    elite_indices: list[int]


class StatsRequest(BaseModel):
    genomes: list[list[float]]
    fitnesses: list[float]
    k_species: int = 4


class StatsResponse(BaseModel):
    mean_fitness: float
    max_fitness: float
    min_fitness: float
    std_fitness: float
    diversity: float
    species_assignments: list[int]
    species_count: int
    hardy_weinberg: list[dict]
    trait_frequencies: list[float]


class BatchForwardRequest(BaseModel):
    nn_weights: list[list[float]]
    inputs: list[list[float]]


class BatchForwardResponse(BaseModel):
    outputs: list[list[float]]


# ═══════════════════════════════════════════════════════════════
#  FASTAPI APP
# ═══════════════════════════════════════════════════════════════

app = FastAPI(title="ML Evolution Engine — Python Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Endpoints ────────────────────────────────────────────

@app.post("/api/init", response_model=InitResponse)
async def api_init(req: InitRequest):
    """Create initial population genomes using NumPy."""
    genomes = create_genomes(req.pop_size, req.traits_per_bacterium)
    return InitResponse(
        genomes=genomes.tolist(),
        genome_length=GENOME_LENGTH,
        trait_gene_count=TRAIT_GENE_COUNT,
        nn_weight_count=NN_WEIGHT_COUNT,
        nn_topology=NN_TOPOLOGY,
    )


@app.post("/api/reproduce", response_model=ReproduceResponse)
async def api_reproduce(req: ReproduceRequest):
    """GA reproduction: selection, crossover, mutation via NumPy."""
    if len(req.survivors) == 0:
        # Extinction — create fresh population
        genomes = create_genomes(req.target_pop)
        return ReproduceResponse(
            offspring_genomes=genomes.tolist(),
            elite_indices=[],
        )

    survivor_genomes = np.array([s.genome for s in req.survivors])
    fitnesses = np.array([s.fitness for s in req.survivors])

    config = {
        "mutation_rate": req.mutation_rate,
        "mutation_sigma": req.mutation_sigma,
        "elitism_pct": req.elitism_pct,
        "crossover_method": req.crossover_method,
        "tournament_size": req.tournament_size,
    }

    offspring = reproduce(survivor_genomes, fitnesses, req.target_pop, config)

    # Compute elite indices
    elite_count = max(1, int(len(fitnesses) * req.elitism_pct / 100))
    elite_indices = np.argsort(fitnesses)[::-1][:elite_count].tolist()

    return ReproduceResponse(
        offspring_genomes=offspring.tolist(),
        elite_indices=elite_indices,
    )


@app.post("/api/stats", response_model=StatsResponse)
async def api_stats(req: StatsRequest):
    """Compute all population statistics using NumPy + scikit-learn."""
    genomes = np.array(req.genomes) if req.genomes else np.zeros((0, GENOME_LENGTH))
    fitnesses = np.array(req.fitnesses) if req.fitnesses else np.zeros(0)

    stats = population_stats(genomes, fitnesses)

    # K-means species clustering
    if len(genomes) >= req.k_species:
        species_assignments, species_count = kmeans_species(genomes, req.k_species)
    else:
        species_assignments = list(range(len(genomes)))
        species_count = len(genomes)

    # Hardy-Weinberg
    hw = hardy_weinberg(genomes)

    # Trait frequencies
    tf = trait_frequencies(genomes)

    return StatsResponse(
        mean_fitness=stats["mean_fitness"],
        max_fitness=stats["max_fitness"],
        min_fitness=stats["min_fitness"],
        std_fitness=stats["std_fitness"],
        diversity=stats["diversity"],
        species_assignments=species_assignments,
        species_count=species_count,
        hardy_weinberg=hw,
        trait_frequencies=tf,
    )


@app.post("/api/nn/forward", response_model=BatchForwardResponse)
async def api_nn_forward(req: BatchForwardRequest):
    """
    Batch neural network forward pass using NumPy.
    Used for bulk prediction (not per-frame — JS handles that).
    """
    weights = np.array(req.nn_weights)
    inputs = np.array(req.inputs)
    outputs = nn_engine.forward_batch(weights, inputs)
    return BatchForwardResponse(outputs=outputs.tolist())


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "genome_length": GENOME_LENGTH,
        "nn_weight_count": NN_WEIGHT_COUNT,
        "nn_topology": NN_TOPOLOGY,
        "trait_gene_count": TRAIT_GENE_COUNT,
    }


# ── Static file serving (AFTER API routes) ───────────────────

@app.get("/")
async def serve_index():
    """Serve the frontend HTML."""
    return FileResponse("index.html")


# Mount remaining static files (JS, CSS) at root level
app.mount("/", StaticFiles(directory="."), name="static")


# ═══════════════════════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("ML Evolution Engine -- Python Backend")
    print(f"   Genome length: {GENOME_LENGTH} (traits: {TRAIT_GENE_COUNT}, NN weights: {NN_WEIGHT_COUNT})")
    print(f"   NN topology:   {NN_TOPOLOGY}")
    print(f"   Starting server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
