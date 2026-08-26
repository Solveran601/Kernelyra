! Shared numerical bounds.  Keeping them in one module prevents binary and
! regression loops from quietly diverging on clipping or zero-variance rules.
module kernelyra_numeric_constants
  use, intrinsic :: iso_c_binding, only: c_float
  implicit none

  real(c_float), parameter :: kr_logit_limit = 30.0_c_float
  real(c_float), parameter :: kr_probability_epsilon = 1.0e-7_c_float
  real(c_float), parameter :: kr_minimum_std = 1.0e-12_c_float
end module kernelyra_numeric_constants
