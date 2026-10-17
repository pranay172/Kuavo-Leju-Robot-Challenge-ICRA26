/* Offscreen cameras need GL work flushed, but no invisible window presentation.
 * A visible simulator delegates to GLFW unchanged. Loaded only by the evaluation harness (LD_PRELOAD).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

void glfwSwapBuffers(void *window) {
    const char *headless = getenv("MUJOCO_HEADLESS");
    if (headless && strcmp(headless, "1") == 0) {
        void (*flush)(void) = dlsym(RTLD_DEFAULT, "glFlush");
        if (flush) flush();
        return;
    }
    void (*original)(void *) = dlsym(RTLD_NEXT, "glfwSwapBuffers");
    original(window);
}
