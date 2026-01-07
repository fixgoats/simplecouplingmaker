#include "Eigen/Dense"
#include "kdtree.h"
#include "raylib.h"
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <iterator>

typedef uint32_t u32;
typedef double f64;
using Eigen::Vector2d, Eigen::Vector2f, Eigen::MatrixXd;

static Eigen::IOFormat defaultFormat(Eigen::StreamPrecision,
                                     Eigen::DontAlignCols, " ", "\n", "", "",
                                     "", "");
constexpr auto square(auto x) { return x * x; }

struct Line {
  friend std::istream &operator>>(std::istream &is, Line &line) {
    return std::getline(is, line.lineTemp);
  }

  // Output function.
  friend std::ostream &operator<<(std::ostream &os, const Line &line) {
    return os << line.lineTemp;
  }

  // cast to needed result
  operator std::string() const { return lineTemp; }
  // Temporary Local storage for line
  std::string lineTemp{};
};

struct Neighbour {
  size_t i;
  size_t j;
  Vector2d d;

  friend std::ostream &operator<<(std::ostream &os, const Neighbour &nb) {
    return os << '(' << nb.i << ", " << nb.j << ", " << nb.d << ')';
  }
};

struct Point : std::array<double, 2> {
  static constexpr int DIM = 2;
  u32 idx;

  Point() {}
  Point(double x, double y, u32 idx) : idx{idx} {
    (*this)[0] = x;
    (*this)[1] = y;
  }

  constexpr Vector2d asVec() const { return {(*this)[0], (*this)[1]}; }
  constexpr Vector2f asfVec() const { return {(*this)[0], (*this)[1]}; }

  double sqdist(const Point &p) const {
    return square((*this)[0] - p[0]) + square((*this)[1] - p[1]);
  }

  double dist(const Point &p) const {
    return sqrt(square((*this)[0] - p[0]) + square((*this)[1] - p[1]));
  }

  friend std::ostream &operator<<(std::ostream &os, const Point &pt) {
    return os << std::format("({}, {})", pt[0], pt[1]);
  }
};

double avgNNDist(kdt::KDTree<Point> &kdtree, const std::vector<Point> &points) {
  std::vector<double> nn_dist(points.size());
  for (size_t i = 0; i < points.size(); i++) {
    int idx = kdtree.knnSearch(points[i], 2)[1];
    nn_dist[i] = points[i].dist(points[idx]);
  }
  double totalnndist =
      std::accumulate(nn_dist.cbegin() + 1, nn_dist.cend(), nn_dist[0]);
  // double avg_nn_dist = nn_dist[0];
  // for (size_t i = 1; i < nn_dist.size(); i++) {
  //   avg_nn_dist += nn_dist[i];
  // }
  return totalnndist / (double)nn_dist.size();
}

double minNNDist(kdt::KDTree<Point> &kdtree, const std::vector<Point> &points) {
  double min_dist = std::numeric_limits<double>::max();
  for (size_t i = 0; i < points.size(); i++) {
    int idx = kdtree.knnSearch(points[i], 2)[1];
    if (points[i].dist(points[idx]) < min_dist) {
      min_dist = points[i].dist(points[idx]);
    }
  }
  return min_dist;
}

std::vector<Point> readPoints(const std::string &fname) {
  u32 m = 0;
  std::ifstream f(fname);
  if (!f.good()) {
    throw std::runtime_error("Failed to read file: " + fname);
  }
  std::vector<std::string> allLines{std::istream_iterator<Line>(f),
                                    std::istream_iterator<Line>()};
  m = allLines.size();
  std::vector<Point> M(m);
  for (u32 j = 0; j < m; j++) {
    std::istringstream stream(allLines[j]);
    std::vector<double> v{std::istream_iterator<double>(stream),
                          std::istream_iterator<double>()};
    M[j] = {v[0], v[1], j};
  }
  return M;
}

template <class D>
void saveEigen(const std::string &fname, const Eigen::MatrixBase<D> &x) {
  std::ofstream f(fname);
  f << x.format(defaultFormat);
  f.close();
}

MatrixXd finite_hamiltonian(u32 n_points, const std::vector<Neighbour> &nbs,
                            double (*f)(Vector2d)) {
  MatrixXd H = MatrixXd::Zero(n_points, n_points);
  for (const auto &nb : nbs) {
    H(nb.i, nb.j) = f(nb.d);
    H(nb.j, nb.i) = f(nb.d);
  }
  return H;
}

std::vector<Neighbour> pointsToNeighbours(const std::vector<Point> &points,
                                          const kdt::KDTree<Point> &kdtree,
                                          f64 radius) {
  std::vector<Neighbour> nb_info;
  for (size_t i = 0; i < points.size(); i++) {
    auto q = points[i];
    auto nbs = kdtree.radiusSearch(q, radius);
    for (const auto idx : nbs) {
      if ((size_t)idx > i) {
        auto p = points[idx];
        Vector2d d = {p[0] - q[0], p[1] - q[1]};
        nb_info.emplace_back(i, p.idx, d);
      }
    }
  }
  return nb_info;
}

MatrixXd pointsToFiniteHamiltonian(const std::vector<Point> &points,
                                   const kdt::KDTree<Point> &kdtree, f64 radius,
                                   double (*f)(Vector2d)) {
  std::vector<Neighbour> nb_info;
  for (size_t i = 0; i < points.size(); i++) {
    auto q = points[i];
    auto nbs = kdtree.radiusSearch(q, radius);
    for (const auto idx : nbs) {
      if ((size_t)idx > i) {
        auto p = points[idx];
        Vector2d d = {p[0] - q[0], p[1] - q[1]};
        nb_info.emplace_back(i, p.idx, d);
      }
    }
  }
  return finite_hamiltonian(points.size(), nb_info, f);
}

double one(Vector2d) { return 1; }

double exponential(Vector2d d) { return exp(-d.norm()); }

double gauss(Vector2d d) { return exp(-d.dot(d)); }

int toScreenX(double r, double min, double max, int dim) {
  return (int)(((r - min) / (max - min)) * (double)dim);
}

int toScreenY(double r, double min, double max, int dim) {
  return (int)(((r - max) / (min - max)) * (double)dim);
}

static const std::string help_text =
    "Usage: %s [-d] [-w] [-h] [-r radius] [-f function] pointfile\n"
    "    pointfile: name of file containing 2d points in the format:\n"
    "        x1 y1\n"
    "        x2 y2\n"
    "        ...\n"
    "    h: Print this text and exit."
    "    w: Display raylib window to view the resultant graph. Press p to take "
    "a screenshot called graph.png.\n"
    "    d: Print average and minimum distance between nearest neighbours and "
    "exit.\n"
    "    r: Search for neighbours within this radius.\n"
    "    f: Displacement vector dependent coupling strength. Possible values:\n"
    "        one (default): constant coupling strength of 1.\n"
    "        exp: exponentially decaying couplings.\n"
    "        gauss: Gaussian decaying couplings.\n";

int main(int argc, char *argv[]) {
  int opt;
  double (*fn)(Vector2d) = one;

  double radius = 0;
  bool print_nn_dist = false;
  bool radius_set = false;
  bool make_window = false;
  while ((opt = getopt(argc, argv, "whdr:f:")) != -1) {
    switch (opt) {
    case 'f':

      if (strcmp(optarg, "exp") == 0) {
        fn = exponential;
      }
      if (strcmp(optarg, "gauss") == 0) {
        fn = gauss;
      }
      break;
    case 'r':
      radius = atof(optarg);
      radius_set = true;
      break;
    case 'd':
      print_nn_dist = true;
      break;
    case 'w':
      make_window = true;
      break;
    case 'h':
      printf(help_text.c_str(), argv[0]);
      exit(EXIT_SUCCESS);

    default: /* '?' */
      fprintf(stderr, help_text.c_str(), argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  if (optind >= argc) {
    fprintf(stderr, help_text.c_str(), argv[0]);
    fprintf(stderr, "Expected argument after options\n");
    exit(EXIT_FAILURE);
  }

  std::vector<Point> points;
  try {
    points = readPoints(argv[optind]);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    fprintf(stderr, "Expected argument after options\n");
  }
  kdt::KDTree<Point> tree(points);
  size_t xmin_idx = tree.axisFindMin(0);
  size_t xmax_idx = tree.axisFindMax(0);
  size_t ymin_idx = tree.axisFindMin(1);
  size_t ymax_idx = tree.axisFindMax(1);
  double xmin = points[xmin_idx][0];
  double xmax = points[xmax_idx][0];
  double ymin = points[ymin_idx][1];
  double ymax = points[ymax_idx][1];
  double exmin = xmin - 0.05 * (xmax - xmin);
  double eymin = ymin - 0.05 * (ymax - ymin);
  double exmax = xmax + 0.05 * (xmax - xmin);
  double eymax = ymax + 0.05 * (ymax - ymin);
  auto neighbours = pointsToNeighbours(points, tree, radius);
  if (make_window) {
    int width = 1080;
    int height = 1080;

    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE |
                   FLAG_WINDOW_TRANSPARENT);
    InitWindow(width, height, "raylib test");
    SetTargetFPS(10);
    while (!WindowShouldClose()) {
      width = GetScreenWidth();
      height = GetScreenHeight();
      BeginDrawing();
      ClearBackground(WHITE);
      for (const auto &point : points) {
        DrawCircle(toScreenX(point[0], exmin, exmax, width),
                   toScreenY(point[1], eymin, eymax, height), 3.0f, RED);
      }
      for (const auto &nb : neighbours) {
        auto startx = toScreenX(points[nb.i][0], exmin, exmax, width);
        auto starty = toScreenY(points[nb.i][1], eymin, eymax, height);
        auto endx = toScreenX(points[nb.j][0], exmin, exmax, width);
        auto endy = toScreenY(points[nb.j][1], eymin, eymax, height);
        DrawLine(startx, starty, endx, endy, BLUE);
      }
      EndDrawing();
      if (IsKeyPressed(KEY_P)) {
        TakeScreenshot("graph.png");
      }
    }
    CloseWindow();
  }
  // printf("name argument = %s\n", argv[optind]);
  if (print_nn_dist) {
    std::cout << "Average distance between nearest neighbours is: "
              << avgNNDist(tree, points)
              << "\n"
                 "Minimum distance between nearest neighbours is: "
              << minNNDist(tree, points) << '\n';
    exit(EXIT_SUCCESS);
  }
  if (!radius_set) {
    fprintf(stderr, help_text.c_str(), argv[0]);
    exit(EXIT_FAILURE);
  }
  auto H = pointsToFiniteHamiltonian(points, tree, radius, fn);
  saveEigen("hamiltonian.txt", H);

  exit(EXIT_SUCCESS);
}
