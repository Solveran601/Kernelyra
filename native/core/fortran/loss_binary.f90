! Numerically bounded binary cross-entropy used by the logistic trainer.
module kernelyra_binary_loss
  use, intrinsic :: iso_c_binding, only: c_double, c_float
  use kernelyra_numeric_constants, only: kr_probability_epsilon
  implicit none
contains
  pure function kr_binary_cross_entropy(target, probability) result(loss)
    real(c_float), value, intent(in) :: target, probability
    real(c_double) :: loss
    real(c_float) :: bounded

    bounded = max(kr_probability_epsilon, min(1.0_c_float - kr_probability_epsilon, probability))
    ! Public classification adapters encode hard labels as exactly 0/1.  In
    ! that common case evaluate only the logarithm that contributes to the
    ! loss.  Soft-label callers retain the complete expression below.
    if (target <= 0.0_c_float) then
      loss = -log(real(1.0_c_float - bounded, c_double))
    else if (target >= 1.0_c_float) then
      loss = -log(real(bounded, c_double))
    else
      loss = -real(target, c_double) * log(real(bounded, c_double)) - &
          real(1.0_c_float - target, c_double) * log(real(1.0_c_float - bounded, c_double))
    end if
  end function kr_binary_cross_entropy
end module kernelyra_binary_loss
