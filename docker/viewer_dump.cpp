// Saves the simulator's MuJoCo viewer (which follows the robot from the front) as timestamped
// JPEGs, for demo videos. Used instead of libheadless_render_fix.so by evaluation/record_demo.sh
// FRONT_VIEW=1; behaves like it otherwise (the viewer window stays hidden and is never presented).
//
//   MUJOCO_VIEWER_DUMP=<dir>          where <unix time>.jpg frames go
//   MUJOCO_VIEWER_DUMP_FPS=10         at most this many frames per second
//   MUJOCO_VIEWER_DUMP_CROP=0.317,0.683   horizontal span of the 3D view between the UI panels
//
// Build (in the simulator container):
//   g++ -shared -fPIC -O2 -o viewer_dump.so viewer_dump.cpp $(pkg-config --cflags --libs opencv4) -ldl -lGL
#include <GL/gl.h>
#include <GL/glx.h>
#include <dlfcn.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

const GLenum kReadFramebufferBinding = 0x8CAA, kReadFramebuffer = 0x8CA8;

void Flush() {
  auto gl_flush = reinterpret_cast<void (*)()>(dlsym(RTLD_DEFAULT, "glFlush"));
  if (gl_flush) gl_flush();
}

void Dump(void* window, const char* directory) {
  auto size = reinterpret_cast<void (*)(void*, int*, int*)>(dlsym(RTLD_DEFAULT, "glfwGetFramebufferSize"));
  auto bind = reinterpret_cast<void (*)(GLenum, GLuint)>(
      glXGetProcAddressARB(reinterpret_cast<const GLubyte*>("glBindFramebuffer")));
  if (!size || !bind) return;
  int width = 0, height = 0;
  size(window, &width, &height);
  if (width <= 0 || height <= 0) return;

  double left = 0.0, right = 1.0;
  if (const char* crop = std::getenv("MUJOCO_VIEWER_DUMP_CROP")) std::sscanf(crop, "%lf,%lf", &left, &right);
  const int x0 = static_cast<int>(left * width), w = static_cast<int>(right * width) - x0;
  if (w <= 0) return;

  GLint previous = 0;
  glGetIntegerv(kReadFramebufferBinding, &previous);
  bind(kReadFramebuffer, 0);
  glReadBuffer(GL_BACK);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  cv::Mat rgb(height, w, CV_8UC3);
  glReadPixels(x0, 0, w, height, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
  bind(kReadFramebuffer, previous);

  cv::Mat bgr;
  cv::flip(rgb, rgb, 0);
  cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
  const double now = std::chrono::duration<double>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  char name[64];
  std::snprintf(name, sizeof(name), "/%.3f.jpg", now);
  cv::imwrite(std::string(directory) + name, bgr, {cv::IMWRITE_JPEG_QUALITY, 92});
}

}  // namespace

extern "C" void glfwSwapBuffers(void* window) {
  static const char* directory = std::getenv("MUJOCO_VIEWER_DUMP");
  static const double interval = 1.0 / (std::getenv("MUJOCO_VIEWER_DUMP_FPS")
                                            ? std::atof(std::getenv("MUJOCO_VIEWER_DUMP_FPS")) : 10.0);
  static auto next = std::chrono::steady_clock::now();
  if (directory && std::chrono::steady_clock::now() >= next) {
    next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(interval));
    if (next < std::chrono::steady_clock::now()) next = std::chrono::steady_clock::now();
    Dump(window, directory);
  }
  const char* headless = std::getenv("MUJOCO_HEADLESS");
  if (headless && std::strcmp(headless, "1") == 0) {
    Flush();
    return;
  }
  auto original = reinterpret_cast<void (*)(void*)>(dlsym(RTLD_NEXT, "glfwSwapBuffers"));
  original(window);
}
