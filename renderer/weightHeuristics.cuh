#ifndef WEIGHT_HEURISTICS_H

#define WEIGHT_HEURISTICS_H

__device__ float balanceHeuristic(float pdf1, int n1, float pdf2, int n2) {
  float numerator = n1 * pdf1;
  float denominator = numerator + n2 * pdf2;
  float result = (denominator == 0.f) ? 0.f : numerator / denominator;
  return result;
}

__device__ float powerHeuristic(float pdf1, int n1, float pdf2, int n2, float n = 2.0f) {
  float numerator = powf(n1 * pdf1, n);
  float denominator = powf(n1 * pdf1, n) + powf(n2 * pdf2, n);
  return (denominator == 0.f) ? 0.f : numerator / denominator;
}

#endif // WEIGHT_HEURISTICS_H