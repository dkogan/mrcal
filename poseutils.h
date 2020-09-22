#pragma once

// Unless specified all arrays stored in contiguous matrices in row-major order.
// The mrcal_..._noncontiguous() functions support a noncontiguous memory
// arrangement
//
// I have two different representations of pose transformations:
//
// - Rt is a concatenated (4,3) array: Rt = nps.glue(R,t, axis=-2). The
//   transformation is R*x+t
//
// - rt is a concatenated (6,) array: rt = nps.glue(r,t, axis=-1). The
//   transformation is R*x+t where R = R_from_r(r)

// Make an identity rotation or transformation
#define mrcal_identity_R(R) mrcal_identity_R_noncontiguous(R,0,0)
void mrcal_identity_R_noncontiguous(double* R,      // (3,3) array
                                    int R_stride0,  // in bytes. <= 0 means "contiguous"
                                    int R_stride1   // in bytes. <= 0 means "contiguous"
                                    );

#define mrcal_identity_r(r) mrcal_identity_r_noncontiguous(r,0)
void mrcal_identity_r_noncontiguous(double* r,      // (3,) array
                                    int r_stride0   // in bytes. <= 0 means "contiguous"
                                    );

#define mrcal_identity_Rt(Rt) mrcal_identity_Rt_noncontiguous(Rt,0,0)
void mrcal_identity_Rt_noncontiguous(double* Rt,      // (4,3) array
                                     int Rt_stride0,  // in bytes. <= 0 means "contiguous"
                                     int Rt_stride1   // in bytes. <= 0 means "contiguous"
                                     );

#define mrcal_identity_rt(rt)  mrcal_identity_rt_noncontiguous(rt,0)
void mrcal_identity_rt_noncontiguous(double* rt,      // (6,) array
                                     int rt_stride0   // in bytes. <= 0 means "contiguous"
                                     );


// Applying rotations. The R version is simply a matrix-vector multiplication
#define mrcal_rotate_point_R(x_out,J_R,J_x,R,x_in) mrcal_rotate_point_R_noncontiguous(x_out,0,J_R,0,0,0,J_x,0,0,R,0,0,x_in,0)
void mrcal_rotate_point_R_noncontiguous( // output
                                        double* x_out,      // (3,) array
                                        int x_out_stride0,  // in bytes. <= 0 means "contiguous"
                                        double* J_R,        // (3,3,3) array. May be NULL
                                        int J_R_stride0,    // in bytes. <= 0 means "contiguous"
                                        int J_R_stride1,    // in bytes. <= 0 means "contiguous"
                                        int J_R_stride2,    // in bytes. <= 0 means "contiguous"
                                        double* J_x,        // (3,3) array. May be NULL
                                        int J_x_stride0,    // in bytes. <= 0 means "contiguous"
                                        int J_x_stride1,    // in bytes. <= 0 means "contiguous"

                                        // input
                                        const double* R,    // (3,3) array. May be NULL
                                        int R_stride0,      // in bytes. <= 0 means "contiguous"
                                        int R_stride1,      // in bytes. <= 0 means "contiguous"
                                        const double* x_in, // (3,) array. May be NULL
                                        int x_in_stride0    // in bytes. <= 0 means "contiguous"
                                         );

// This one is implemented in C++ in poseutils-uses-autodiff.cc
#define mrcal_rotate_point_r(x_out,J_r,J_x,r,x_in) mrcal_rotate_point_r_noncontiguous(x_out,0,J_r,0,0,J_x,0,0,r,0,x_in,0)
void mrcal_rotate_point_r_noncontiguous( // output
                                        double* x_out,      // (3,) array
                                        int x_out_stride0,  // in bytes. <= 0 means "contiguous"
                                        double* J_r,        // (3,3) array. May be NULL
                                        int J_r_stride0,    // in bytes. <= 0 means "contiguous"
                                        int J_r_stride1,    // in bytes. <= 0 means "contiguous"
                                        double* J_x,        // (3,3) array. May be NULL
                                        int J_x_stride0,    // in bytes. <= 0 means "contiguous"
                                        int J_x_stride1,    // in bytes. <= 0 means "contiguous"

                                        // input
                                        const double* r,    // (3,) array. May be NULL
                                        int r_stride0,      // in bytes. <= 0 means "contiguous"
                                        const double* x_in, // (3,) array. May be NULL
                                        int x_in_stride0    // in bytes. <= 0 means "contiguous"
                                         );

// Apply a transformation to a point
#define mrcal_transform_point_Rt(x_out,J_Rt,J_x,Rt,x_in) mrcal_transform_point_Rt_noncontiguous(x_out,0,J_Rt,0,0,0,J_x,0,0,Rt,0,0,x_in,0)
void mrcal_transform_point_Rt_noncontiguous( // output
                                            double* x_out,      // (3,) array
                                            int x_out_stride0,  // in bytes. <= 0 means "contiguous"
                                            double* J_Rt,       // (3,4,3) array. May be NULL
                                            int J_Rt_stride0,   // in bytes. <= 0 means "contiguous"
                                            int J_Rt_stride1,   // in bytes. <= 0 means "contiguous"
                                            int J_Rt_stride2,   // in bytes. <= 0 means "contiguous"
                                            double* J_x,        // (3,3) array. May be NULL
                                            int J_x_stride0,    // in bytes. <= 0 means "contiguous"
                                            int J_x_stride1,    // in bytes. <= 0 means "contiguous"

                                            // input
                                            const double* Rt,   // (4,3) array. May be NULL
                                            int Rt_stride0,     // in bytes. <= 0 means "contiguous"
                                            int Rt_stride1,     // in bytes. <= 0 means "contiguous"
                                            const double* x_in, // (3,) array. May be NULL
                                            int x_in_stride0    // in bytes. <= 0 means "contiguous"
                                             );

#define mrcal_transform_point_rt(x_out,J_rt,J_x,rt,x_in) mrcal_transform_point_rt_noncontiguous(x_out,0,J_rt,0,0,J_x,0,0,rt,0,x_in,0)
void mrcal_transform_point_rt_noncontiguous( // output
                                            double* x_out,      // (3,) array
                                            int x_out_stride0,  // in bytes. <= 0 means "contiguous"
                                            double* J_rt,       // (3,6) array. May be NULL
                                            int J_rt_stride0,   // in bytes. <= 0 means "contiguous"
                                            int J_rt_stride1,   // in bytes. <= 0 means "contiguous"
                                            double* J_x,        // (3,3) array. May be NULL
                                            int J_x_stride0,    // in bytes. <= 0 means "contiguous"
                                            int J_x_stride1,    // in bytes. <= 0 means "contiguous"

                                            // input
                                            const double* rt,   // (6,) array. May be NULL
                                            int rt_stride0,     // in bytes. <= 0 means "contiguous"
                                            const double* x_in, // (3,) array. May be NULL
                                            int x_in_stride0    // in bytes. <= 0 means "contiguous"
                                             );

// This one is implemented in C++ in poseutils-uses-autodiff.cc
// Convert a rotation representation from a matrix to a Rodrigues vector
#define mrcal_r_from_R(r,J,R) mrcal_r_from_R_noncontiguous(r,0,J,0,0,0,R,0,0)
void mrcal_r_from_R_noncontiguous( // output
                                  double* r,       // (3,) vector
                                  int r_stride0,   // in bytes. <= 0 means "contiguous"
                                  double* J,       // (3,3,3) array. Gradient. May be NULL
                                  int J_stride0,   // in bytes. <= 0 means "contiguous"
                                  int J_stride1,   // in bytes. <= 0 means "contiguous"
                                  int J_stride2,   // in bytes. <= 0 means "contiguous"

                                  // input
                                  const double* R, // (3,3) array
                                  int R_stride0,   // in bytes. <= 0 means "contiguous"
                                  int R_stride1    // in bytes. <= 0 means "contiguous"
                                   );

// Convert a rotation representation from a Rodrigues vector to a matrix
#define mrcal_R_from_r(R,J,r) mrcal_R_from_r_noncontiguous(R,0,0,J,0,0,0,r,0)
void mrcal_R_from_r_noncontiguous( // outputs
                                  double* R,       // (3,3) array
                                  int R_stride0,   // in bytes. <= 0 means "contiguous"
                                  int R_stride1,   // in bytes. <= 0 means "contiguous"
                                  double* J,       // (3,3,3) array. Gradient. May be NULL
                                  int J_stride0,   // in bytes. <= 0 means "contiguous"
                                  int J_stride1,   // in bytes. <= 0 means "contiguous"
                                  int J_stride2,   // in bytes. <= 0 means "contiguous"

                                  // input
                                  const double* r, // (3,) vector
                                  int r_stride0    // in bytes. <= 0 means "contiguous"
                                   );

// Convert a transformation representation from Rt to rt
#define mrcal_rt_from_Rt(rt,Rt) mrcal_rt_from_Rt_noncontiguous(rt,0,NULL,0,0,0,Rt,0,0)
void mrcal_rt_from_Rt_noncontiguous(   // output
                                    double* rt,      // (6,) vector
                                    int rt_stride0,  // in bytes. <= 0 means "contiguous"
                                    double* J_R,     // (3,3,3) array. Gradient. May be NULL
                                    // No J_t. It's always the identity
                                    int J_R_stride0, // in bytes. <= 0 means "contiguous"
                                    int J_R_stride1, // in bytes. <= 0 means "contiguous"
                                    int J_R_stride2, // in bytes. <= 0 means "contiguous"

                                    // input
                                    const double* Rt,  // (4,3) array
                                    int Rt_stride0,    // in bytes. <= 0 means "contiguous"
                                    int Rt_stride1     // in bytes. <= 0 means "contiguous"
                                       );

// Convert a transformation representation from Rt to rt
#define mrcal_Rt_from_rt(Rt,rt) mrcal_Rt_from_rt_noncontiguous(Rt,0,0,NULL,0,0,0,rt,0)
void mrcal_Rt_from_rt_noncontiguous(   // output
                                    double* Rt,      // (4,3) array
                                    int Rt_stride0,  // in bytes. <= 0 means "contiguous"
                                    int Rt_stride1,  // in bytes. <= 0 means "contiguous"
                                    double* J_r,     // (3,3,3) array. Gradient. May be NULL
                                    // No J_t. It's just the identity
                                    int J_r_stride0, // in bytes. <= 0 means "contiguous"
                                    int J_r_stride1, // in bytes. <= 0 means "contiguous"
                                    int J_r_stride2, // in bytes. <= 0 means "contiguous"

                                    // input
                                    const double* rt, // (6,) vector
                                    int rt_stride0    // in bytes. <= 0 means "contiguous"
                                       );

// Invert an Rt transformation
//
// b = Ra + t  -> a = R'b - R't
#define mrcal_invert_Rt(Rt_out,Rt_in) mrcal_invert_Rt_noncontiguous(Rt_out,0,0,Rt_in,0,0)
void mrcal_invert_Rt_noncontiguous( // output
                                   double* Rt_out,      // (4,3) array
                                   int Rt_out_stride0,  // in bytes. <= 0 means "contiguous"
                                   int Rt_out_stride1,  // in bytes. <= 0 means "contiguous"

                                   // input
                                   const double* Rt_in, // (4,3) array
                                   int Rt_in_stride0,   // in bytes. <= 0 means "contiguous"
                                   int Rt_in_stride1    // in bytes. <= 0 means "contiguous"
                                    );

// Invert an rt transformation
//
// b = rotate(a) + t  -> a = invrotate(b) - invrotate(t)
//
// drout_drin is not returned: it is always -I
// drout_dtin is not returned: it is always 0
#define mrcal_invert_rt(rt_out,dtout_drin,dtout_dtin,rt_in) mrcal_invert_rt_noncontiguous(rt_out,0,dtout_drin,0,0,dtout_dtin,0,0,rt_in,0)
void mrcal_invert_rt_noncontiguous( // output
                                   double* rt_out,          // (6,) array
                                   int rt_out_stride0,      // in bytes. <= 0 means "contiguous"
                                   double* dtout_drin,      // (3,3) array
                                   int dtout_drin_stride0,  // in bytes. <= 0 means "contiguous"
                                   int dtout_drin_stride1,  // in bytes. <= 0 means "contiguous"
                                   double* dtout_dtin,      // (3,3) array
                                   int dtout_dtin_stride0,  // in bytes. <= 0 means "contiguous"
                                   int dtout_dtin_stride1,  // in bytes. <= 0 means "contiguous"

                                   // input
                                   const double* rt_in,     // (6,) array
                                   int rt_in_stride0        // in bytes. <= 0 means "contiguous"
                                    );


// Compose two Rt transformations
//   R0*(R1*x + t1) + t0 =
//   (R0*R1)*x + R0*t1+t0
#define mrcal_compose_Rt(Rt_out,Rt_0,Rt_1) mrcal_compose_Rt_noncontiguous(Rt_out,0,0,Rt_0,0,0,Rt_1,0,0)
void mrcal_compose_Rt_noncontiguous( // output
                                    double* Rt_out,      // (4,3) array
                                    int Rt_out_stride0,  // in bytes. <= 0 means "contiguous"
                                    int Rt_out_stride1,  // in bytes. <= 0 means "contiguous"

                                    // input
                                    const double* Rt_0,  // (4,3) array
                                    int Rt_0_stride0,    // in bytes. <= 0 means "contiguous"
                                    int Rt_0_stride1,    // in bytes. <= 0 means "contiguous"
                                    const double* Rt_1,  // (4,3) array
                                    int Rt_1_stride0,    // in bytes. <= 0 means "contiguous"
                                    int Rt_1_stride1     // in bytes. <= 0 means "contiguous"
                                     );
// Compose two rt transformations. It is assumed that we're getting no gradients
// at all or we're getting ALL the gradients: only dr_r0 is checked for NULL
//
// dr_dt0 is not returned: it is always 0
// dr_dt1 is not returned: it is always 0
// dt_dr1 is not returned: it is always 0
// dt_dt0 is not returned: it is always the identity matrix
#define mrcal_compose_rt(rt_out,dr_r0,dr_r1,dt_r0,dt_t1,rt_0,rt_1) mrcal_compose_rt_noncontiguous(rt_out,0,dr_r0,0,0,dr_r1,0,0,dt_r0,0,0,dt_t1,0,0,rt_0,0,rt_1,0)
void mrcal_compose_rt_noncontiguous( // output
                                    double* rt_out,       // (6,) array
                                    int rt_out_stride0,   // in bytes. <= 0 means "contiguous"
                                    double* dr_r0,        // (3,3) array; may be NULL
                                    int dr_r0_stride0,    // in bytes. <= 0 means "contiguous"
                                    int dr_r0_stride1,    // in bytes. <= 0 means "contiguous"
                                    double* dr_r1,        // (3,3) array; may be NULL
                                    int dr_r1_stride0,    // in bytes. <= 0 means "contiguous"
                                    int dr_r1_stride1,    // in bytes. <= 0 means "contiguous"
                                    double* dt_r0,        // (3,3) array; may be NULL
                                    int dt_r0_stride0,    // in bytes. <= 0 means "contiguous"
                                    int dt_r0_stride1,    // in bytes. <= 0 means "contiguous"
                                    double* dt_t1,        // (3,3) array; may be NULL
                                    int dt_t1_stride0,    // in bytes. <= 0 means "contiguous"
                                    int dt_t1_stride1,    // in bytes. <= 0 means "contiguous"

                                    // input
                                    const double* rt_0,   // (6,) array
                                    int rt_0_stride0,     // in bytes. <= 0 means "contiguous"
                                    const double* rt_1,   // (6,) array
                                    int rt_1_stride0      // in bytes. <= 0 means "contiguous"
                                     );
