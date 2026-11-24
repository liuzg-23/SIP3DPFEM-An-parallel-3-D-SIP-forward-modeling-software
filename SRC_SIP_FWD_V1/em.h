
#ifndef _EM_H
#define _EM_H

#include <Eigen/Dense>  // Linear Algebra Lib.
#include <Eigen/Sparse> // Sparse Lib
#include <Eigen/StdVector>
#include <complex>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sys/resource.h>
#include <unistd.h> // Add this line

#include "mfem.hpp"
using namespace mfem;

namespace EM {
typedef std::complex<double> Dcomplex;
static const Dcomplex II = Dcomplex(0., 1.);

static const double PI = 3.1415926535897932384626433832795;
#define MAX_INT std::numeric_limits<int>::max()

// 角度制和幅度制的转换损失
const double ARC_TO_DEG = 57.29577951308238;
const double DEG_TO_ARC = 0.0174532925199433;

// Tolerance used in our package.
static const double TOLERANCE = 1e-6;

#define TRANSTOL 1e-8 // modellingbase.h

// A invalid big number
static const unsigned int INVALID_UNIT = static_cast<unsigned int>(-1);

// Using eigen matrix and vector
typedef Eigen::SparseMatrix<Dcomplex, Eigen::RowMajor> DESM;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> RESM;
typedef Eigen::SparseMatrix<Dcomplex, Eigen::ColMajor> DESM_ColMajor;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> RESM_ColMajor;
typedef Eigen::Matrix<Dcomplex, Eigen::Dynamic, Eigen::Dynamic> DEM;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> REM;
typedef Eigen::VectorXd REV;
typedef Eigen::VectorXd DenseVector; // double
typedef Eigen::VectorXcd DEV;
typedef Eigen::SparseVector<Dcomplex> DSEV;
typedef Eigen::SparseVector<double> RSEV;
typedef Eigen::Matrix<double, 3, 3> Matrix3D;    // 3*3
typedef Eigen::Matrix<double, 1, 3> RowVector3D; // 1*3
typedef Eigen::Vector3d Vector3D; // 3*1 //默认是列向量,只有一列
typedef Eigen::Matrix<double, 4, 3> Matrix43D;
typedef Eigen::Matrix<double, 1, 3> Matrix13D;
typedef Eigen::Matrix<double, 1, 4> Matrix14D;
typedef Eigen::Matrix<double, 4, 4> Matrix4D;

#define WHERE bert_str(__FILE__) + ": " + bert_str(__LINE__) + "\t"
#define WHERE_AM_I WHERE + "\t" + bert_str(__ASSERT_FUNCTION) + " "

enum LogType { Verbose, Info, Warning, Error, Debug, Critical };

} // namespace EM
using namespace EM;

/* ----------------Basic and common function----------------------- */

// customize assert function for replacing assert()---------------------start
#define MY_ENABLE_ASSERT
void static assert_fail(const char *file, int line, const char *func,
                        const char *expr, const std::string &msg) {
  (void)file;
  (void)line;
  (void)func;
  (void)expr;
  (void)msg;
  std::string err_msg = std::string("Assertion fails:") + expr +
                        ", message:" + msg + ", at " + file + ":" +
                        std::to_string(line) + ":" + func;
  std::cout << err_msg << std::endl;
  std::terminate();
}

#ifdef MY_ENABLE_ASSERT
#define myassert(expr)                                                         \
  do {                                                                         \
    if (!(expr)) {                                                             \
      assert_fail(__FILE__, __LINE__, __func__, #expr, "");                    \
    }                                                                          \
  } while (0)

#define MY_ASSERT_EQ(a, b)                                                     \
  do {                                                                         \
    using std::to_string;                                                      \
    if ((a) != (b)) {                                                          \
      assert_fail(__FILE__, __LINE__, __func__, #a " == " #b,                  \
                  std::to_string(a) + " != " + std::to_string(b));             \
    }                                                                          \
  } while (0)

#define MY_ASSERT_NE(a, b)                                                     \
  do {                                                                         \
    using std::to_string;                                                      \
    if ((a) == (b)) {                                                          \
      assert_fail(__FILE__, __LINE__, __func__, #a " == " #b,                  \
                  std::to_string(a) + " != " + std::to_string(b));             \
    }                                                                          \
  } while (0)
#else
#define MY_ASSERT(expr) (void)0
#define MY_ASSERT_EQ(expr) (void)0
#define MY_ASSERT_NE(expr) (void)0
#endif

/*
 * @Description: This class is from Zhengyong Ren to describe point of 3D.
 */

#ifndef _Point_double_H
#define _Point_double_H

// C++ includes
#include <cassert>
#include <cmath>
#include <iostream>

class Point_double {
public:
  Point_double(const double x = 0., const double y = 0., const double z = 0.);
  Point_double(const Point_double &p);
  virtual ~Point_double();

  friend std::ostream &operator<<(std::ostream &os,
                                  const Point_double &Point_double);

  // Overload operators.
  Point_double &operator=(const Point_double &p);
  double operator()(const unsigned int i) const;
  double &operator()(const unsigned int i);
  Point_double operator+(const Point_double &v) const;
  Point_double operator-(const Point_double &v) const;
  Point_double operator*(const double a) const;
  double operator*(const Point_double &v) const;
  bool operator==(const Point_double &v) const;
  bool operator<(const Point_double &v) const;
  void operator=(const double a);

  // Math functions
  Point_double cross(const Point_double &v) const;
  Point_double unit() const;
  double size() const;
  void zero();
  double *get_xyz() { return _coords; }

protected:
  double _coords[3];
};

//-----------------------------------------------------------------------
inline Point_double::Point_double(const double x, const double y,
                                  const double z) {
  _coords[0] = x;
  _coords[1] = y;
  _coords[2] = z;
}

inline Point_double::Point_double(const Point_double &p) {
  for (unsigned int i = 0; i < 3; i++)
    _coords[i] = p._coords[i];
}

inline Point_double &Point_double::operator=(const Point_double &p) {
  _coords[0] = p._coords[0];
  _coords[1] = p._coords[1];
  _coords[2] = p._coords[2];

  return (*this);
}

inline Point_double::~Point_double() {
  // no space is allocated by the new operator.
  // so,there is no delete [].
}

inline double Point_double::operator()(const unsigned int i) const {
  assert(i < 3);
  return _coords[i];
}

inline double &Point_double::operator()(const unsigned int i) {
  assert(i < 3);
  return _coords[i];
}

inline Point_double Point_double::operator+(const Point_double &p) const {
  return Point_double(_coords[0] + p._coords[0], _coords[1] + p._coords[1],
                      _coords[2] + p._coords[2]);
}

inline Point_double Point_double::operator-(const Point_double &p) const {
  return Point_double(_coords[0] - p._coords[0], _coords[1] - p._coords[1],
                      _coords[2] - p._coords[2]);
}

inline Point_double Point_double::operator*(const double factor) const {
  return Point_double(_coords[0] * factor, _coords[1] * factor,
                      _coords[2] * factor);
}

inline double Point_double::operator*(const Point_double &p) const {
  return (_coords[0] * p(0) + _coords[1] * p(1) + _coords[2] * p(2));
}

inline bool Point_double::operator==(const Point_double &rhs) const {
  return ((fabs(_coords[0] - rhs._coords[0]) +
           fabs(_coords[1] - rhs._coords[1]) +
           fabs(_coords[2] - rhs._coords[2])) < 3 * 1e-6);
}

inline bool Point_double::operator<(const Point_double &rhs) const {
  // First we assume (this)<rhs true
  if (*this == rhs)
    return false;
  if ((*this)(0) < rhs(0))
    return true; //  <
  else if ((*this)(0) > rhs(0))
    return false;                                  //  >
  else if (std::abs((*this)(0) - rhs(0)) < 1e-6) { // vx=rhsx
    if ((*this)(1) < rhs(1))
      return true;
    else if ((*this)(1) > rhs(1))
      return false;
    else if (std::abs((*this)(1) - rhs(1)) < 1e-6) { // vy=rhsy
      if ((*this)(2) < rhs(2))
        return true;
      else if ((*this)(2) > rhs(2))
        return false;
    }
  }
  return false;
}

inline void Point_double::operator=(const double a) {
  _coords[0] = a;
  _coords[1] = a;
  _coords[2] = a;
}

inline Point_double Point_double::cross(const Point_double &p) const {
  return Point_double(_coords[1] * p._coords[2] - _coords[2] * p._coords[1],
                      -_coords[0] * p._coords[2] + _coords[2] * p._coords[0],
                      _coords[0] * p._coords[1] - _coords[1] * p._coords[0]);
}

inline Point_double Point_double::unit() const {
  const double length = size();
  return Point_double(_coords[0] / length, _coords[1] / length,
                      _coords[2] / length);
}

inline double Point_double::size() const {
  double value = std::pow(_coords[0], 2) + std::pow(_coords[1], 2) +
                 std::pow(_coords[2], 2);

  return std::sqrt(value);
}

inline void Point_double::zero() {
  _coords[0] = 0.;
  _coords[1] = 0.;
  _coords[2] = 0.;
}

inline std::ostream &operator<<(std::ostream &os,
                                const Point_double &Point_double) {
  os << Point_double(0) << "\t" << Point_double(1) << "\t" << Point_double(2);
  return os;
}

#endif // _Point_double_H

//-----------------------------------Public
//Function--------------------------------

// overload operator << for eigen vector  // static also work
inline std::ostream &operator<<(std::ostream &output, REV &v) {
  for (int i = 0; i < v.size(); i++) {
    std::cout << v(i) << " ";
  }
  return output; // necessary!
}

// overload operator << for eigen const vector
inline std::ostream &operator<<(std::ostream &output, const REV &v) {
  for (int i = 0; i < v.size(); i++) {
    std::cout << v(i) << " ";
  }
  return output; // necessary!
}

inline void throwError(const std::string &errString) {
  std::cerr << errString << std::endl;
}

template <class T, class U> T max(const T &a, const U &b) {
  return std::max(a, T(b));
}

template <typename Value, typename... Values>
std::string bert_str(Value v, Values... vs) {
  std::ostringstream os;
  using expander = int[];
  os << v; // first
  (void)expander{0, (os << " " << vs, void(), 0)...};
  return os.str();
}

void display(REV x);
void display(Vertex x);
void display(std::vector<Vertex> x);

std::vector<int> sort(const std::vector<int> &a);
std::vector<int> unique(const std::vector<int> &a);
template <class T> void print_std_vector(T a) {
  for (auto e : a) {
    std::cout << e << " " << std::endl;
  }
}
void print_time(double time_in_seconds);

//默认值只能出现在声明
void print_value(double value, int dim1 = -1, int dim2 = -1,
                 std::string fn = "value.txt");

double median(const REV &a);
REV logspace(double a, double b, int n);
REV normalRandn(int n);

template <class T> bool isInfNaN(const T &a) {
  return (std::isinf(a) || std::isnan(a));
}

inline bool haveInfNaN(const REV &v) {
  for (int i = 0; i < v.size(); i++) {
    if (isInfNaN(v(i)))
      return true;
  }
  return false;
}

#define VECTORASCSUFFIX ".vector"
#define VECTORBINSUFFIX ".bvec"
typedef int64_t int64;
enum IOFormat { Ascii, Binary };
bool save(const REV &a, const std::string &filename, IOFormat format = Ascii);
bool saveVec(const REV &a, const std::string &filename,
             IOFormat format = Ascii);

template <typename... Values> void log(LogType type, Values... vs) {
  return log(type, bert_str(vs...));
}

// Calculate the length between two points: coord1 and coord2
double length_two_point(double *coord1, double *coord2);
double length_two_point(Vector &coord1, Vector &coord2);
double length_two_point(Point_double &coord1, Point_double &coord2);

// 按照坐标排序x，y，z
bool cmp(std::vector<double> a, std::vector<double> b);

// 按照坐标排序x1,y1,z1 x2,y2,z2
bool cmp2(std::vector<double> a, std::vector<double> b);

// 按照两个点的坐标进行排序，位置分别是1,5
bool cmpMN15(std::vector<double> a, std::vector<double> b);

bool cmpN(std::vector<double> a, std::vector<double> b, int N);
bool cmp3(std::vector<double> a, std::vector<double> b);
bool cmp4(std::vector<double> a, std::vector<double> b);

void remove_duplicate_vertex(std::vector<mfem::Vertex> &all_point);

double GetCurrentMemoryUsage();

// whether a==b, note width == height for a and b
bool equalDM(DenseMatrix a, DenseMatrix b);
bool equalVector(Vector a, Vector b);

#endif // _EM_H
