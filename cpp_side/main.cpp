#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>

#include "pico/stdlib.h"
#include "model_params.hpp"  // AUTOGEN: CLASSES, FEATS, MU[], SIGMA[], W[][], LABELS[]

#ifndef LED_PIN
#define LED_PIN 25  // Pico (non-W). If using Pico W, LED control differs.
#endif

// ====== Helpers ==============================================================
static inline void zscore(float* x) {
    for (int i = 0; i < model::FEATS; ++i) {
        x[i] = (x[i] - model::MU[i]) / (model::SIGMA[i] + 1e-8f);
    }
}

static inline void logits_softmax(const float* xz, float* probs /* size=CLASSES */) {
    float logits[model::CLASSES];
    for (int c = 0; c < model::CLASSES; ++c) {
        float z = model::W[c][0]; // bias
        for (int f = 0; f < model::FEATS; ++f) z += model::W[c][1 + f] * xz[f];
        logits[c] = z;
    }
    float mx = logits[0];
    for (int c = 1; c < model::CLASSES; ++c) mx = std::max(mx, logits[c]);
    float sum = 0.f;
    for (int c = 0; c < model::CLASSES; ++c) { probs[c] = expf(logits[c] - mx); sum += probs[c]; }
    float inv = 1.f / (sum + 1e-12f);
    for (int c = 0; c < model::CLASSES; ++c) probs[c] *= inv;
}

static inline int argmax(const float* v, int n) {
    int bi = 0; float bv = v[0];
    for (int i = 1; i < n; ++i) if (v[i] > bv) { bv = v[i]; bi = i; }
    return bi;
}

// Simple CSV parse: "f0,f1,...,fN\n"
static bool parse_csv_floats(const char* s, float* out, int expected_n) {
    int n = 0;
    const char* p = s;
    while (*p && n < expected_n) {
        char* endp = nullptr;
        float v = strtof(p, &endp);
        if (endp == p) return false; 
        out[n++] = v;
        if (*endp == ',') p = endp + 1;
        else if (*endp == '\n' || *endp == '\r' || *endp == '\0') { p = endp; break; }
        else p = endp;
    }
    return (n == expected_n);
}

// USB-CDC line reader (until '\n' or timeout_ms)
static bool read_line(char* buf, size_t buflen, uint32_t timeout_ms) {
    size_t i = 0;
    absolute_time_t deadline = make_timeout_time_ms(timeout_ms);
    while (absolute_time_diff_us(get_absolute_time(), deadline) > 0) {
        int ch = getchar_timeout_us(1000); // 1ms
        if (ch == PICO_ERROR_TIMEOUT) continue;
        if (ch == '\r') continue;
        if (ch == '\n') { buf[i] = '\0'; return true; }
        if (i + 1 < buflen) buf[i++] = (char)ch;
    }
    return false;
}

// ====== Hysteresis FSM =======================================================
struct ClassFSM {
    const char* name;
    float th_on;
    float th_off;
    int   need_on;   
    int   count_on;  
    bool  active;
};

// Default thresholds per label
static void init_fsm(ClassFSM* fsm, int n) {
    for (int c = 0; c < n; ++c) {
        fsm[c].name    = model::LABELS[c];
        fsm[c].th_on   = 0.70f;
        fsm[c].th_off  = 0.50f;
        fsm[c].need_on = 3;
        fsm[c].count_on= 0;
        fsm[c].active  = false;

        if (strcmp(model::LABELS[c], "smoke_alarm") == 0) {
            fsm[c].th_on = 0.85f; fsm[c].th_off = 0.65f; fsm[c].need_on = 3;
        } else if (strcmp(model::LABELS[c], "doorbell") == 0) {
            fsm[c].th_on = 0.75f; fsm[c].th_off = 0.55f; fsm[c].need_on = 3;
        } else if (strcmp(model::LABELS[c], "baby") == 0 || strcmp(model::LABELS[c], "baby_cry") == 0) {
            fsm[c].th_on = 0.75f; fsm[c].th_off = 0.55f; fsm[c].need_on = 3;
        }
        // "other" is not considered an active alarm, only logged
    }
}

static void fsm_step(ClassFSM* fsm, const float* probs) {
    for (int c = 0; c < model::CLASSES; ++c) {
        if (probs[c] > fsm[c].th_on) {
            if (fsm[c].count_on < 1000) fsm[c].count_on++;
            if (fsm[c].count_on >= fsm[c].need_on) fsm[c].active = true;
        } else if (probs[c] < fsm[c].th_off) {
            fsm[c].count_on = 0;
            fsm[c].active   = false;
        }
        // in hysteresis band -> keep current state
    }
}

static bool any_alert_active(const ClassFSM* fsm) {
    for (int c = 0; c < model::CLASSES; ++c) {
        if (strcmp(model::LABELS[c], "other") == 0) continue;
        if (fsm[c].active) return true;
    }
    return false;
}

// ====== MAIN =================================================================
int main() {
    stdio_init_all();
    gpio_init(LED_PIN); gpio_set_dir(LED_PIN, GPIO_OUT);

    for (int i = 0; i < 3000; i += 10) {
        if (stdio_usb_connected()) break;
        sleep_ms(10);
    }
    printf("# Pico Edge AI (LR) — FEATS=%d, CLASSES=%d\n", model::FEATS, model::CLASSES);
    for (int c = 0; c < model::CLASSES; ++c) {
        printf("# class[%d]=%s\n", c, model::LABELS[c]);
    }

    ClassFSM fsm[model::CLASSES];
    init_fsm(fsm, model::CLASSES);

    char line[8192];
    float x[model::FEATS];
    float probs[model::CLASSES];

    while (true) {
        // 1) Read line (CSV features)
        bool ok = read_line(line, sizeof(line), 5000 /*ms*/);
        if (!ok) {
            gpio_put(LED_PIN, 0);
            continue;
        }
        if (line[0] == '\0') continue;
        if (strcmp(line, "PING") == 0) { printf("PONG\n"); continue; }
        if (strcmp(line, "RESET") == 0) {
            for (int c = 0; c < model::CLASSES; ++c) { fsm[c].count_on = 0; fsm[c].active = false; }
            printf("# FSM reset\n"); continue;
        }

        // 2) Parse
        if (!parse_csv_floats(line, x, model::FEATS)) {
            printf("!ERR parse (expected %d floats)\n", model::FEATS);
            continue;
        }

        // 3) Z-score
        zscore(x);

        // 4) Softmax
        logits_softmax(x, probs);

        // 5) FSM
        fsm_step(fsm, probs);

        // 6) LED and log
        bool led_on = any_alert_active(fsm);
        gpio_put(LED_PIN, led_on ? 1 : 0);

        printf("probs:");
        for (int c = 0; c < model::CLASSES; ++c) {
            printf(" %s=%.3f", model::LABELS[c], (double)probs[c]);
        }
        printf(" | active:");
        for (int c = 0; c < model::CLASSES; ++c) {
            printf(" %s=%d", model::LABELS[c], fsm[c].active ? 1 : 0);
        }
        int k = argmax(probs, model::CLASSES);
        printf(" | top=%s\n", model::LABELS[k]);

        tight_loop_contents();
    }
    return 0;
}
