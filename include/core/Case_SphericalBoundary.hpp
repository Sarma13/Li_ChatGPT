#ifndef CASE_SPHERICALBOUNDARY_HPP
#define CASE_SPHERICALBOUNDARY_HPP

#include "BoundaryData.hpp"
#include "Inputs.hpp"

/**
 * \class Case_SphericalBoundary
 * \brief Derived class: handles the dynamics of a fluid-fluid interface
 *        initialized as a spherical surface (e.g., droplet boundary).
 *
 *  This class inherits from BoundaryData and overrides the methods
 *  needed to initialize, remesh, filter, compute curvature, and set
 *  endpoint derivatives for a sphere of radius R_d (from Inputs.gamma).
 */
class Case_SphericalBoundary : public BoundaryData {
public:
    /**
     * \brief Constructor: passes input data to BoundaryData base.
     */
    explicit Case_SphericalBoundary(const Input &data);

    /**
     * \brief Initialize nodes on a sphere of radius gamma
     *        centered appropriately in (r,z).
     */
    void initialize() override;

    /**
     * \brief Remesh the spherical boundary nodes along arc length
     *        preserving a nearly-uniform discretization.
     */
    void remesh_boundary() override;

    /**
     * \brief Apply smoothing filter to spherical interface nodes
     *        to prevent numerical instabilities.
     */
    void filter_boundary() override;

    /**
     * \brief Compute the curvature at each node of the spherical interface.
     */
    void boundary_curvature() override;

    /**
     * \brief Estimate endpoint derivatives (for clamped spline) at the
     *        two poles of the sphere.
     */
    void boundary_endpoints_derivatives() override;
};

#endif // CASE_SPHERICALBOUNDARY_HPP
