#include "Eigen/Dense"
#include "kdtree.h"
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

std::vector<Point> readPoints(const std::string &fname) {
  u32 m = 0;
  std::ifstream f(fname);
  if (!f.good()) {
    throw std::exception();
  }
  std::vector<std::string> allLines{std::istream_iterator<Line>(f),
                                    std::istream_iterator<Line>()};
  m = allLines.size();
  // f.clear();
  // f.seekg(0, std::ios::beg);
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

MatrixXd finite_hamiltonian(u32 n_points, const std::vector<Neighbour> &nbs) {
  MatrixXd H = MatrixXd::Zero(n_points, n_points);
  for (const auto &nb : nbs) {
    H(nb.i, nb.j) = 1;
    H(nb.j, nb.i) = 1;
  }
  return H;
}

MatrixXd pointsToFiniteHamiltonian(const std::vector<Point> &points,
                                   const kdt::KDTree<Point> &kdtree,
                                   f64 radius) {
  /* This function creates a hamiltonian for a simple finite lattice.
   * Can't exactly do a dispersion from this.
   */
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
  std::cout << "got here\n";
  return finite_hamiltonian(points.size(), nb_info);
}

int main(int argc, char *argv[]) {
  auto points = readPoints("points.txt");
  kdt::KDTree<Point> tree(points);
  auto H = pointsToFiniteHamiltonian(points, tree, 1.01);
  saveEigen("hamiltonian.txt", H);
  return 0;
}
