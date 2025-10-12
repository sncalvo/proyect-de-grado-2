// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef H_INTEGRATOR_COMMON
#define H_INTEGRATOR_COMMON

#define _USE_MATH_DEFINES
#include <cmath>
#include "platform_macros.h"
#include "random_interface.h"

// We can use a much larger depth limit, as RR will handle termination efficiently.
constexpr unsigned int MAX_DEPTH_SAFETY_NET = 128;
constexpr unsigned int MIN_BOUNCES_FOR_RR = 5; // Start Russian Roulette after this many bounces.
constexpr float SIGMA_A = 0.05f;
constexpr float SIGMA_S = 0.95f;
constexpr float MIN_STEP = 0.5f;
constexpr float MAX_STEP = 5.f;
constexpr float PI = 3.14159265358979323846f;
constexpr float DENSITY = 0.5f;

// Common utility functions
template<typename T>
inline __hostdev__ T clamp(T min, T max, T value) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

template<typename Vec3T>
inline __hostdev__ Vec3T cross(const Vec3T& a, const Vec3T& b) {
    return Vec3T(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

template<typename Vec3T>
inline __hostdev__ float dot(const Vec3T& a, const Vec3T& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template<typename Vec3T>
inline __hostdev__ void coordinateSystem(const Vec3T& a, Vec3T& b, Vec3T& c) {
    if (fabs(a[0]) > fabs(a[1])) {
        b = Vec3T(-a[2], 0.0f, a[0]) / sqrtf(a[0] * a[0] + a[2] * a[2]);
    } else {
        b = Vec3T(0.0f, a[2], -a[1]) / sqrtf(a[1] * a[1] + a[2] * a[2]);
    }
    c = cross(a, b);
}

template<typename Vec3T>
inline __hostdev__ Vec3T sampleHenyeyGreenstein(float g, Vec3T w, float random1, float random2) {
    float invHG = (1.f - g * g) / (1.f - g + 2.f * g * random1);
    float cosTheta = (1.f + g * g - invHG * invHG) / (2.f * g);
    float sinTheta = sqrtf(1.f - cosTheta * cosTheta);
    float phi = 2.f * PI * random2;

    Vec3T u, v;
    coordinateSystem(w, u, v);

    return cosTheta * w + sinTheta * cosf(phi) * u + sinTheta * sinf(phi) * v;
}

inline __hostdev__ float henyey_greenstein(const float& g, const float& cos_theta)
{
    float denom = 1 + g * g - 2 * g * cos_theta;
    return 1 / (4 * PI) * (1 - g * g) / (denom * sqrtf(denom));
}

inline __hostdev__ float balanceHeuristic(float pdf1, int n1, float pdf2, int n2) {
    float numerator = n1 * pdf1;
    float denominator = numerator + n2 * pdf2;
    return (denominator == 0.f) ? 0.f : numerator / denominator;
}

// Point-to-point marching algorithm (shadow rays / light transmission)
// This is templated to work with both CPU and GPU grid types
template<typename WorldT, typename Vec3T, typename RayT, typename AccessorT, typename CoordT>
inline __hostdev__ float PointToPointMarching(
    const WorldT& world, 
    const Vec3T& startPos,
    const Vec3T& endPos,
    RandomState* localState, 
    float sigmaMAJ, 
    AccessorT& acc,
    float pfar,
    RayT& piRay
) {
    Vec3T start = Vec3T(piRay(pfar));
    Vec3T iEnd = world.worldToIndex(endPos);
    Vec3T rayDir = iEnd - start;
    float totalDistance = rayDir.length();
    float transmittance = 1.0f;
    
    rayDir.normalize();

    RayT iRay(start, rayDir);

    // Clip ray to bounding box
    if (!world.clipRay(iRay)) {
        return transmittance;
    }

    float far = world.getRayT0(iRay);
    float t1 = world.getRayT1(iRay);
    float distanceTraveled = 0.0f;

    while (true) {
        Vec3T currentPos = iRay(far);
        CoordT coord = world.floorCoord(currentPos);

        Vec3T remainingVec = iEnd - currentPos;
        distanceTraveled = (start - currentPos).length();
        
        if (distanceTraveled >= totalDistance) {
            break;
        }

        float sigma = 0.0f;
        if (world.isActive(coord)) {
            sigma = acc.getValue(coord) * DENSITY;
        }

        float muA = sigma * SIGMA_A;
        float muS = sigma * SIGMA_S;
        float muT = muA + muS;

        float pathLength;
        if (sigma > 0.f) {
            pathLength = -logf(1.0f - curand_uniform(localState)) / sigmaMAJ;
            pathLength = clamp(MIN_STEP, MAX_STEP, pathLength);
        } else {
            pathLength = MIN_STEP * 10.f;
        }

        far += pathLength;
        if (far > t1)
            return transmittance;

        if (sigma <= 0.f) {
            continue;
        }

        float sampleAttenuation = expf(-(pathLength)*muT);
        transmittance *= sampleAttenuation;

        if (transmittance < 0.05f) {
            float q = 0.75f;
            float sample = curand_uniform(localState);
            if (sample < q) {
                transmittance = 0.f;
            } else {
                transmittance /= 1.0f - q;
            }
        }

        if (transmittance <= 0.0f) {
            return 0.f;
        }
    }

    return transmittance;
}


template<typename WorldT, typename Vec3T, typename Vec4T, typename RayT, typename AccessorT, typename CoordT>
inline __hostdev__ Vec4T DeltaTrackingIntegration(
    int i,
    const WorldT& world,
    const Vec3T& rayEye,
    float* image,
    int width,
    int height,
    RandomState* localState,
    float sigmaMAJ,
    AccessorT& acc,
    const Vec3T& lightPosition
) {
    Vec3T rayDir = world.getRayDirection(i);
    RayT wRay(rayEye, rayDir);
    RayT iRay = world.worldToIndexRay(wRay);
    Vec3T backgroundColor(0.36f, 0.702f, 0.98f);

    if (!world.clipRay(iRay)) {
        return Vec4T(backgroundColor[0], backgroundColor[1], backgroundColor[2], 1.0f);
    }

    Vec3T color(0.f, 0.f, 0.f);
    Vec3T pathThroughput(1.f, 1.f, 1.f);
    bool absorbed = false;
    float far = world.getRayT0(iRay);

    // The depth limit is now just a safety net against infinite loops.
    for (unsigned int depth = 0; !absorbed && depth < MAX_DEPTH_SAFETY_NET; ++depth) {
        float pathLength = -logf(1.0f - curand_uniform(localState)) / sigmaMAJ;
        far += pathLength;

        if (far > world.getRayT1(iRay)) {
            break;
        }

        Vec3T currentPos = iRay(far);
        CoordT coord = world.floorCoord(currentPos);

        float sigma = 0.0f;
        if (world.isActive(coord)) {
            sigma = acc.getValue(coord) * DENSITY;
        }

        if (sigma <= 0.f) continue;

        float muA = sigma * SIGMA_A;
        float muS = sigma * SIGMA_S;
        float muT = muA + muS;

        float realCollisionProb = muT / sigmaMAJ;
        if (curand_uniform(localState) >= realCollisionProb) continue;

        // --- A real collision has occurred ---
        float albedo = (muT > 0.f) ? (muS / muT) : 0.f;
        const float g = 0.2f;

        // 4. Next-Event Estimation (Direct Lighting)
        {
            Vec3T scatterPosWorld = wRay(far);
            Vec3T lightDir = lightPosition - scatterPosWorld;
            float lightDistSq = lightDir.lengthSqr();
            lightDir.normalize();

            const float pdf_light = 1.0f;
            float pdf_phase = henyey_greenstein(g, dot(rayDir, lightDir));
            float mis_weight = balanceHeuristic(pdf_light, 1, pdf_phase, 1);

            float lightTransmittance = PointToPointMarching<WorldT, Vec3T, RayT, AccessorT, CoordT>(
                world, scatterPosWorld, lightPosition, localState, sigmaMAJ, acc, far, iRay);

            if (lightTransmittance > 0.0f) {
                Vec3T lightIntensity = Vec3T(1.f, 1.f, 1.f) * 40000.f;
                Vec3T L_i = lightIntensity / lightDistSq;
                float phaseVal = henyey_greenstein(g, dot(rayDir, lightDir));
                color += pathThroughput * L_i * lightTransmittance * phaseVal * mis_weight;
            }
        }

        // 5. Path Continuation - CORRECTED LOGIC
        
        // ✨ Step 1: Apply physical attenuation. The path ALWAYS gets dimmer.
        pathThroughput *= albedo;

        // ✨ Step 2: Apply Russian Roulette for efficiency.
        if (depth >= MIN_BOUNCES_FOR_RR) {
            float survivalProb = fmaxf(pathThroughput[0], fmaxf(pathThroughput[1], pathThroughput[2]));
            if (curand_uniform(localState) > survivalProb) {
                absorbed = true; // Terminate the path
            } else {
                // Compensate the energy of the surviving path
                pathThroughput *= (1.0f / survivalProb);
            }
        }
        
        // If the path was not terminated, scatter it.
        if (!absorbed) {
            float sample1 = curand_uniform(localState);
            float sample2 = curand_uniform(localState);
            rayDir = sampleHenyeyGreenstein(g, rayDir, sample1, sample2);

            iRay = world.createIndexRay(iRay(far), rayDir, wRay.eye());
            if (!world.clipRay(iRay)) {
                absorbed = true;
            }
            far = world.getRayT0(iRay);
        }
    }

    if (!absorbed) {
        color += pathThroughput * backgroundColor;
    }

    return Vec4T(color[0], color[1], color[2], 1.0f);
}
#endif // H_INTEGRATOR_COMMON
