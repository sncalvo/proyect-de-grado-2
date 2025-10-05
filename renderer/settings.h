#pragma once

class Settings {
public:
    static Settings& getInstance() {
        static Settings instance;
        return instance;
    }

    // Delete copy constructor and assignment operator
    Settings(const Settings&) = delete;
    Settings& operator=(const Settings&) = delete;

    // Direct access to arrays for ImGui
    float cameraLocation[3];
    float lightLocation[3];
    float lightColor[3];
    unsigned int pixelSamples;

private:
    // Private constructor for singleton
    Settings() {
        // Initialize with default values
        cameraLocation[0] = 0.0f;
        cameraLocation[1] = 0.0f;
        cameraLocation[2] = 5.0f;

        lightLocation[0] = -0.92f;
        lightLocation[1] = 100.0f;
        lightLocation[2] = 70.97f;

        lightColor[0] = 1.0f;  // R
        lightColor[1] = 1.0f;  // G
        lightColor[2] = 1.0f;  // B (White light)

        pixelSamples = 4;  // Default pixel samples
    }
}; 