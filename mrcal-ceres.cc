// Ceres-based solver backend for mrcal.
//
// Architecture:
//
// 1. Before the solve, optimizer_callback() is called ONCE at the initial
//    point to discover the Jacobian sparsity structure (which state variables
//    affect which measurements).
//
// 2. Consecutive measurements that depend on the same set of parameter blocks
//    are grouped together.  One ceres::CostFunction is created per group.
//    Each cost function is registered with only the parameter blocks it
//    actually uses, so Ceres can build a sparse J^T J and run
//    SPARSE_NORMAL_CHOLESKY efficiently.
//
// 3. A ceres::EvaluationCallback calls optimizer_callback() exactly ONCE per
//    outer iteration, filling shared residual and Jacobian buffers.
//
// 4. Each per-group CostFunction::Evaluate() simply reads its slice of the
//    shared buffers; it does not call optimizer_callback() itself.
//
// 5. All Ceres parameter blocks are sub-ranges of the same contiguous
//    packed_state array.  Ceres updates them in-place via
//    StateVectorToParameterBlocks(), so packed_state always reflects the
//    current (or trial) parameter values when the EvaluationCallback fires.
//
// Build requirements: link against libceres, libcholmod.

#include <algorithm>
#include <cstring>
#include <set>
#include <vector>

#include <ceres/cost_function.h>
#include <ceres/evaluation_callback.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <ceres/types.h>
#include <suitesparse/cholmod.h>

extern "C" {
#include "mrcal.h"
#include "internal.h"
}

// Function pointer type matching optimizer_callback_ceres_shim() in mrcal.c.
typedef void (*mrcal_optimizer_callback_fn_t)(const double*   packed_state,
                                               double*         x,
                                               cholmod_sparse* Jt,
                                               const void*     ctx);

// ---------------------------------------------------------------------------
// Shared buffers: filled once per outer iteration by the EvaluationCallback,
// then read (not written) by each per-group CostFunction::Evaluate().
// ---------------------------------------------------------------------------
struct MrcalSharedBuffers
{
    std::vector<double> x;       // residuals,             length Nmeasurements
    std::vector<int>    colptr;  // Jt CSC column ptrs,    length Nmeasurements+1
    std::vector<int>    rowidx;  // Jt CSC row indices,    length N_j_nonzero
    std::vector<double> val;     // Jt CSC values,         length N_j_nonzero
    cholmod_sparse      Jt;      // shell; p/i/x point into above
};

// ---------------------------------------------------------------------------
// EvaluationCallback: drives the single optimizer_callback() call per step
// ---------------------------------------------------------------------------
class MrcalEvaluationCallback : public ceres::EvaluationCallback
{
public:
    MrcalEvaluationCallback(mrcal_optimizer_callback_fn_t callback,
                              const void*                   ctx,
                              double*                       packed_state,
                              MrcalSharedBuffers*           shared)
        : callback_(callback), ctx_(ctx),
          packed_state_(packed_state), shared_(shared) {}

    void PrepareForEvaluation(bool evaluate_jacobians,
                               bool new_evaluation_point) override
    {
        // All Ceres parameter blocks are sub-ranges of packed_state_ and are
        // updated in-place before this callback fires, so packed_state_
        // already holds the current trial values.
        if (evaluate_jacobians)
        {
            // Need the Jacobian.  Call the callback unconditionally: either
            // the point is new (must recompute everything), or the point is
            // the same as the previous residuals-only call (Jt not yet filled).
            callback_(packed_state_, shared_->x.data(), &shared_->Jt, ctx_);
        }
        else if (new_evaluation_point)
        {
            // New point, residuals only.
            callback_(packed_state_, shared_->x.data(), nullptr, ctx_);
        }
        // else: same point, residuals-only again — nothing to do.
    }

private:
    mrcal_optimizer_callback_fn_t callback_;
    const void*                   ctx_;
    double*                       packed_state_;
    MrcalSharedBuffers*           shared_;
};

// ---------------------------------------------------------------------------
// Per-group cost function: reads pre-computed data from shared buffers
// ---------------------------------------------------------------------------
// A "group" is a maximal run of consecutive measurements that all depend on
// the same set of parameter blocks.  For a calibration problem these groups
// correspond naturally to board observations (all corners of one
// camera+frame pair share the same intrinsics, extrinsics, and frame blocks).
class MrcalGroupCostFunction : public ceres::CostFunction
{
public:
    // meas_start    – index of first measurement in this group
    // meas_count    – number of measurements
    // local_blocks  – sorted list of parameter-block indices (into blk[])
    //                 that this group depends on
    // blk           – block_offsets array (Nblocks+1 entries):
    //                 block b covers packed_state[blk[b]..blk[b+1])
    // shared        – shared residual/Jacobian buffers
    MrcalGroupCostFunction(int                        meas_start,
                            int                        meas_count,
                            const std::vector<int>&    local_blocks,
                            const std::vector<int>&    blk,
                            const MrcalSharedBuffers*  shared)
        : meas_start_(meas_start),
          local_blocks_(local_blocks),
          blk_(blk),
          shared_(shared)
    {
        set_num_residuals(meas_count);
        for (int b : local_blocks_)
            mutable_parameter_block_sizes()->push_back(blk_[b + 1] - blk_[b]);
    }

    bool Evaluate(const double* const* /*parameters*/,
                  double*              residuals,
                  double**             jacobians) const override
    {
        // Residuals were filled by the EvaluationCallback; just copy our slice.
        int num_res = num_residuals();
        std::memcpy(residuals, &shared_->x[meas_start_],
                    static_cast<size_t>(num_res) * sizeof(double));

        if (jacobians == nullptr)
            return true;

        const int*    colptr = shared_->colptr.data();
        const int*    rowidx = shared_->rowidx.data();
        const double* val    = shared_->val.data();

        // For each local parameter block, scatter the relevant Jt entries
        // into the dense row-major sub-Jacobian Ceres expects:
        //   jacobians[ib][i * blk_size + j] = dresidual[meas_start+i] / dparam_b[j]
        for (int ib = 0; ib < static_cast<int>(local_blocks_.size()); ib++)
        {
            if (jacobians[ib] == nullptr) continue;

            int    b         = local_blocks_[ib];
            int    blk_start = blk_[b];
            int    blk_size  = blk_[b + 1] - blk_start;
            double* Jb       = jacobians[ib];

            std::memset(Jb, 0,
                        static_cast<size_t>(num_res) * blk_size * sizeof(double));

            for (int i = 0; i < num_res; i++)
            {
                int m = meas_start_ + i;
                for (int k = colptr[m]; k < colptr[m + 1]; k++)
                {
                    int s = rowidx[k];
                    if (s >= blk_start && s < blk_start + blk_size)
                        Jb[i * blk_size + (s - blk_start)] = val[k];
                }
            }
        }
        return true;
    }

private:
    int                       meas_start_;
    std::vector<int>          local_blocks_;
    std::vector<int>          blk_;      // full block_offsets (Nblocks+1 entries)
    const MrcalSharedBuffers* shared_;
};

// ---------------------------------------------------------------------------
// Iteration callback: real-time progress and libdogleg-compatible termination
// ---------------------------------------------------------------------------
class MrcalIterationCallback : public ceres::IterationCallback
{
public:
    explicit MrcalIterationCallback(bool verbose) : verbose_(verbose) {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& s) override
    {
        if (verbose_)
            fprintf(stderr, "iter %4d  cost %f  cost_change %e  trust_region %e\n",
                    s.iteration, 2.0 * s.cost, 2.0 * s.cost_change,
                    s.trust_region_radius);
        return ceres::SOLVER_CONTINUE;
    }
private:
    bool verbose_;
};

// ---------------------------------------------------------------------------
// C entry point called from mrcal_optimize_ceres() in mrcal.c
//
// block_offsets   – array of Nblocks+1 ints; block b covers
//                   packed_state[block_offsets[b]..block_offsets[b+1]).
//                   Blocks are ordered: camera intrinsics, camera extrinsics,
//                   frames, variable points, calobject_warp.
// Nblocks_e_start – index of the first Schur e-block (= number of camera blocks)
// Nblocks_e_count – number of e-blocks (frames + variable points)
//                   When > 0, SPARSE_SCHUR is used; otherwise SPARSE_NORMAL_CHOLESKY.
// ---------------------------------------------------------------------------
extern "C"
double _mrcal_ceres_solve(double*                       packed_state,
                           int                           Nstate,
                           int                           Nmeasurements,
                           int                           N_j_nonzero,
                           const void*                   ctx,
                           mrcal_optimizer_callback_fn_t callback,
                           const int*                    block_offsets,
                           int                           Nblocks,
                           int                           Nblocks_e_start,
                           int                           Nblocks_e_count,
                           bool                          verbose)
{
    // ---- Shared buffers ------------------------------------------------
    MrcalSharedBuffers shared;
    shared.x.resize(Nmeasurements);
    shared.colptr.resize(Nmeasurements + 1);
    shared.rowidx.resize(N_j_nonzero);
    shared.val.resize(N_j_nonzero);

    std::memset(&shared.Jt, 0, sizeof(shared.Jt));
    shared.Jt.nrow   = Nstate;
    shared.Jt.ncol   = Nmeasurements;
    shared.Jt.nzmax  = N_j_nonzero;
    shared.Jt.p      = shared.colptr.data();
    shared.Jt.i      = shared.rowidx.data();
    shared.Jt.x      = shared.val.data();
    shared.Jt.packed = 1;
    shared.Jt.stype  = 0;
    shared.Jt.itype  = CHOLMOD_INT;
    shared.Jt.xtype  = CHOLMOD_REAL;
    shared.Jt.dtype  = CHOLMOD_DOUBLE;

    // ---- Pre-call: discover Jacobian sparsity at the initial point ------
    // This is also the initial residual/Jacobian evaluation that seeds the
    // shared buffers before the first EvaluationCallback fires.
    callback(packed_state, shared.x.data(), &shared.Jt, ctx);

    if (verbose)
    {
        int nnz = shared.colptr[Nmeasurements];
        fprintf(stderr, "Jacobian: %d x %d,  nonzeros %d  (dense would be %d,  fill %.1f%%)\n",
                Nmeasurements, Nstate, nnz,
                Nmeasurements * Nstate,
                100.0 * nnz / (double)(Nmeasurements * Nstate));
    }

    // ---- Block layout ---------------------------------------------------
    std::vector<int> blk(block_offsets, block_offsets + Nblocks + 1);

    // Return the block index for state variable s (binary search in blk[]).
    auto state_to_block = [&](int s) -> int {
        auto it = std::upper_bound(blk.begin(), blk.end(), s);
        return static_cast<int>(it - blk.begin()) - 1;
    };

    // Return the sorted set of block indices that measurement m depends on.
    auto blocks_for_meas = [&](int m) -> std::vector<int> {
        std::set<int> bs;
        for (int k = shared.colptr[m]; k < shared.colptr[m + 1]; k++)
            bs.insert(state_to_block(shared.rowidx[k]));
        return std::vector<int>(bs.begin(), bs.end());
    };

    // ---- Group consecutive measurements with identical block sets --------
    struct Group { int start, count; std::vector<int> blocks; };
    std::vector<Group> groups;

    if (Nmeasurements > 0)
    {
        Group cur { 0, 1, blocks_for_meas(0) };
        for (int m = 1; m < Nmeasurements; m++)
        {
            auto bm = blocks_for_meas(m);
            if (bm == cur.blocks)
                cur.count++;
            else
            {
                groups.push_back(std::move(cur));
                cur = { m, 1, std::move(bm) };
            }
        }
        groups.push_back(std::move(cur));
    }

    // ---- Build Ceres problem --------------------------------------------
    // evaluation_callback lives in Problem::Options in this version of Ceres.
    MrcalEvaluationCallback eval_cb(callback, ctx, packed_state, &shared);

    ceres::Problem::Options problem_options;
    problem_options.evaluation_callback = &eval_cb;
    ceres::Problem problem(problem_options);

    // Each parameter block is a contiguous sub-range of packed_state.
    // Ceres will update these in-place during the solve, keeping packed_state
    // current so the EvaluationCallback can pass it straight to the callback.
    for (int b = 0; b < Nblocks; b++)
        problem.AddParameterBlock(packed_state + blk[b], blk[b + 1] - blk[b]);

    for (const auto& g : groups)
    {
        auto* cf = new MrcalGroupCostFunction(g.start, g.count,
                                               g.blocks, blk, &shared);
        std::vector<double*> ptrs;
        for (int b : g.blocks)
            ptrs.push_back(packed_state + blk[b]);
        problem.AddResidualBlock(cf, /*loss=*/nullptr, ptrs);
    }

    // ---- Solver ---------------------------------------------------------
    ceres::Solver::Options options;
    options.trust_region_strategy_type   = ceres::DOGLEG;
    options.dogleg_type                  = ceres::TRADITIONAL_DOGLEG;
    options.max_num_iterations           = 300;
    options.function_tolerance           = 1e-8;  // stop when relative cost change < 1e-8
    options.gradient_tolerance           = 0.0;
    options.parameter_tolerance          = 0.0;
    options.minimizer_progress_to_stdout = false;

    // Use SPARSE_SCHUR when there are e-blocks (frames/points): eliminates
    // them via Schur complement, leaving a much smaller camera-only system.
    if (Nblocks_e_count > 0)
    {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        auto* ordering = new ceres::ParameterBlockOrdering;
        for (int b = 0; b < Nblocks; b++)
        {
            bool is_e = (b >= Nblocks_e_start && b < Nblocks_e_start + Nblocks_e_count);
            ordering->AddElementToGroup(packed_state + blk[b], is_e ? 0 : 1);
        }
        options.linear_solver_ordering.reset(ordering);
    }
    else
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

    // Always register: handles both real-time printing and the absolute
    // step-norm stopping criterion that mirrors libdogleg's update_threshold.
    MrcalIterationCallback iter_cb(verbose);
    options.callbacks.push_back(&iter_cb);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (verbose)
        fprintf(stderr, "Termination: %s\n", summary.message.c_str());

    if (summary.termination_type == ceres::FAILURE ||
        summary.termination_type == ceres::USER_FAILURE)
        return -1.0;

    // Ceres minimises 0.5*||f||^2; return ||f||^2 to match mrcal convention.
    return summary.final_cost * 2.0;
}
