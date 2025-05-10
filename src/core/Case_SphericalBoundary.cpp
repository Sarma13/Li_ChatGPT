// File: Bubble_Dynamics/src/core/Case_SphericalBoundary.cpp

#include "Case_SphericalBoundary.hpp"
#include "cubic_spline.hpp"
#include <armadillo>
#include <cmath>
#include <vector>

using namespace std;
using namespace arma;

static constexpr double pi = 3.14159265358979323846;

Case_SphericalBoundary::Case_SphericalBoundary(const Input &data)
  : BoundaryData(data)
{}

void Case_SphericalBoundary::initialize() {
  // Nb+1 nodes from north to south pole on a perfect sphere of radius gamma
  for (int i = 0; i <= Ns; ++i) {
    
    
    double alpha = double(i) * pi / double(Ns);
    r_nodes[i]    = xi * sin(alpha);
    z_nodes[i]    = xi * cos(alpha) - gamma;
    F_nodes[i]    = 0.0;
    curv_nodes[i] = 2.0 / 2;
  }
  // enforce r=0 at the poles
  r_nodes.front() = r_nodes.back() = 0.0;
}

void Case_SphericalBoundary::boundary_endpoints_derivatives() {
  // at north pole, tangent is horizontal to +r
  drds1 = +1.0; dzds1 = 0.0;
  // at south pole, same horizontal tangent
  drds2 = -1.0; dzds2 = 0.0;
  dphi1ds1 = dphi1ds2 = dphi2ds1 = dphi2ds2 = 0.0;
}

/*! \brief Remesh the spherical boundary using geometric progression along arc-length. Copy-paste from BubbleData::remesh_bubble.
    Similar to BubbleData::remesh_bubble, but applied to the fluid-fluid interface.
*/
void Case_SphericalBoundary::remesh_boundary() {
  // generate current arc-length distribution for creating later evenly spaced nodes
  // copy original nodes
  vec rs = conv_to<vec>::from(r_nodes);
  vec zs = conv_to<vec>::from(z_nodes);

  vec drs = diff(rs);
  vec dzs = diff(zs);
  vec seg = sqrt(drs % drs + dzs % dzs);  
  //cumsum stands for cumulative sum See my notes:Compute the cumulative sum of the elements of a vector. Last element is total length
  vec distance_s = join_cols(vec(1, fill::zeros), cumsum(seg));

  vector<double> distance = conv_to<vector<double>>::from(distance_s);

  // build clamped-end splines for r(s), z(s), and F(s)
  boundary_endpoints_derivatives();

  cubic_spline spR, spZ, spF;
  spR.set_spline(distance, r_nodes, drds1, drds2);
  spZ.set_spline(distance, z_nodes, dzds1, dzds2);
  spF.set_spline(distance, F_nodes, 0.0, 0.0);

  // geometric progression spacing along total length L
  
  int n = Ns + 1;
  double r = 0.9999;  // nearly uniform, adjust for clustering
  double h = (r - 1.0) / (pow(r, n - 1) - 1.0);

  vec e(Ns);

  for (int i = 0; i < Ns; ++i) {
      e(i) = pow(r, 1.0*i);
  }

  vec s_new = h * cumsum(e) * distance[Ns];
  s_new.insert_rows(0, vec(1, fill::zeros));      // s_new size = Ns+1
  s_new(Ns) = distance[Ns];              // ensure last point

  // interpolate to new nodes
  for (int i = 0; i <= Ns; ++i) {
      double s = s_new(i);
      r_nodes[i] = spR.interpolate(s);
      z_nodes[i] = spZ.interpolate(s);
      F_nodes[i] = spF.interpolate(s);
  }
  // enforce axisymmetry
  r_nodes.front() = r_nodes.back() = 0.0;
}

/*! \brief Compute axisymmetric curvature at each boundary node.
  Fits a polynomial in a local 9-point stencil, computes meridional and azimuthal terms.
*/




void Case_SphericalBoundary::boundary_curvature() {
  
  //Solution Vecotors
  vector<double> r_copy = r_nodes;
  vector<double> z_copy = z_nodes;
  vector<double> F_copy = F_nodes;

  r_copy.insert(r_copy.begin(), -r_nodes[0+1] );
  r_copy.insert(r_copy.begin(), -r_nodes[0+2]);
  r_copy.insert(r_copy.begin(), -r_nodes[0+3]);
  r_copy.insert(r_copy.begin(), -r_nodes[0+4]);
  r_copy.push_back(-r_nodes[Ns - 1]);
  r_copy.push_back(-r_nodes[Ns - 2]);
  r_copy.push_back(-r_nodes[Ns - 3]);
  r_copy.push_back(-r_nodes[Ns - 4]);

  //Same for z
  z_copy.insert(z_copy.begin(), z_nodes[0+1]);
  z_copy.insert(z_copy.begin(), z_nodes[0+2]);
  z_copy.insert(z_copy.begin(), z_nodes[0+3]);
  z_copy.insert(z_copy.begin(), z_nodes[0+4]);
  z_copy.push_back(z_nodes[Ns - 1]);
  z_copy.push_back(z_nodes[Ns - 2]);
  z_copy.push_back(z_nodes[Ns - 3]);
  z_copy.push_back(z_nodes[Ns - 4]);

  vec rs = conv_to<vec>::from(r_copy);
  vec zs = conv_to<vec>::from(z_copy);
  vec drs = diff(rs);
  vec dzs = diff(zs);
  vec pointss = sqrt(drs % drs + dzs % dzs);
  vec distance_s = cumsum(pointss);
  vec begin(1, fill::zeros);
  distance_s = join_cols(begin, distance_s);

  int degree = 4; // polynomial degree
  int n_elem = 9; // number of nodes points over which the polynomial curve fitting is conducted
  double numerator, denominator;

  double drds_i, d2rds2_i, dzds_i, d2zds2_i;


  vec r_i(n_elem, fill::zeros);
  vec z_i(n_elem, fill::zeros);
  vec distance_i(n_elem, fill::zeros);

  for (int i = 0; i < Ns + 1; ++i) {
      for (int j = 0; j < n_elem; ++j) {
          distance_i(j) = distance_s(i + j);
          r_i(j) = r_copy[i + j];
          z_i(j) = z_copy[i + j];
      }
      double s_tmp = distance_s(i + 4);
      // polynomial curve fitting f(x)=a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
      vec r_coeff = polyfit(distance_i, r_i, degree);
      vec z_coeff = polyfit(distance_i, z_i, degree);

      drds_i =
              4.0 * s_tmp * s_tmp * s_tmp * r_coeff(0) + 3.0 * s_tmp * s_tmp * r_coeff(1) + 2.0 * s_tmp * r_coeff(2) +
              r_coeff(3);
      d2rds2_i = 12.0 * s_tmp * s_tmp * r_coeff(0) + 6.0 * s_tmp * r_coeff(1) + 2.0 * r_coeff(2);
      dzds_i =
              4.0 * s_tmp * s_tmp * s_tmp * z_coeff(0) + 3.0 * s_tmp * s_tmp * z_coeff(1) + 2.0 * s_tmp * z_coeff(2) +
              z_coeff(3);
      d2zds2_i = 12.0 * s_tmp * s_tmp * z_coeff(0) + 6.0 * s_tmp * z_coeff(1) + 2.0 * z_coeff(2);


      numerator = drds_i * d2zds2_i - dzds_i * d2rds2_i;
      denominator = pow((drds_i * drds_i + dzds_i * dzds_i), 1.5);
      curv_nodes[i] = numerator / denominator + dzds_i / (r_nodes[i] * pow((drds_i * drds_i + dzds_i * dzds_i), 0.5));

      // at the symmetry axis r = 0.0, the curvature is defined as
      // K = 2 * d2z/ds2 / (dr/ds)^3

      if (i == Ns || i == 0) {
          curv_nodes[i] = 2 * d2zds2_i / (drds_i * drds_i * drds_i);
      }

  }

}




void Case_SphericalBoundary::filter_boundary() {
  // copy original nodes
  vector<double> r_copy(r_nodes.begin(), r_nodes.end());
  vector<double> z_copy(z_nodes.begin(), z_nodes.end());
  vector<double> f_copy(F_nodes.begin(), F_nodes.end());

  // mirror two nodes at each end for 5-point smoothing
  r_copy.insert(r_copy.begin(), -r_nodes[1]);
  r_copy.insert(r_copy.begin(), -r_nodes[2]);
  r_copy.push_back(-r_nodes[Ns - 1]);
  r_copy.push_back(-r_nodes[Ns - 2]);

  z_copy.insert(z_copy.begin(), z_nodes[1]);
  z_copy.insert(z_copy.begin(), z_nodes[2]);
  z_copy.push_back(z_nodes[Ns - 1]);
  z_copy.push_back(z_nodes[Ns - 2]);

  f_copy.insert(f_copy.begin(), F_nodes[1]);
  f_copy.insert(f_copy.begin(), F_nodes[2]);
  f_copy.push_back(F_nodes[Ns - 1]);
  f_copy.push_back(F_nodes[Ns - 2]);

  // compute the arclength for the padded nodes
  vec rb = conv_to<vec>::from(r_copy);
  vec zb = conv_to<vec>::from(z_copy);
  vec drb = diff(rb), dzb = diff(zb);
  vec seg = sqrt(drb % drb + dzb % dzb);
  vec scum = join_cols(vec{0.0}, cumsum(seg));
  vector<double> dist = conv_to<vector<double>>::from(scum);

  double s1, s2, s3, s4, s5;
  double c1, c2, c3, c4, c5;

  // Longuet-Higgins & Cokelet smoothing
  for (int i = 0; i <= Ns; ++i) {
      s1 = dist[i];   s2 = dist[i+1];
      s3 = dist[i+2]; s4 = dist[i+3];
      s5 = dist[i+4];
      c1 = 0.5 * ((s3 - s2) * (s3 - s4)) / ((s1 - s3) * (s1 - s5));
      c2 = 0.5 * ((s4 - s3)) /       ((s4 - s2));
      c3 = 0.5 * (1.0 + ((s3 - s2)*(s3 - s4))/((s3 - s1)*(s3 - s5)));
      c4 = 0.5 * ((s2 - s3)) /       ((s2 - s4));
      c5 = 0.5 * ((s3 - s2)*(s3 - s4))/((s5 - s1)*(s5 - s3));

      r_nodes[i] = c1*r_copy[i]   + c2*r_copy[i+1] + c3*r_copy[i+2]
                 + c4*r_copy[i+3] + c5*r_copy[i+4];
      z_nodes[i] = c1*z_copy[i]   + c2*z_copy[i+1] + c3*z_copy[i+2]
                 + c4*z_copy[i+3] + c5*z_copy[i+4];
      F_nodes[i] = c1*f_copy[i]   + c2*f_copy[i+1] + c3*f_copy[i+2]
                 + c4*f_copy[i+3] + c5*f_copy[i+4];
  }
  // enforce axisymmetry at poles
  r_nodes.front() = r_nodes.back() = 0.0;
}





