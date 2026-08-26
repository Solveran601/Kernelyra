! Cross-entropy for the already-normalized class probabilities.
module kernelyra_multiclass_loss
  use, intrinsic :: iso_c_binding, only: c_double, c_float, c_size_t
  use kernelyra_numeric_constants, only: kr_probability_epsilon
  implicit none
contains
  pure function kr_multiclass_cross_entropy(probabilities, truth) result(loss)
    real(c_float), intent(in) :: probabilities(*)
    integer(c_size_t), value, intent(in) :: truth
    real(c_double) :: loss
    real(c_float) :: probability

    probability = max(kr_probability_epsilon, min(1.0_c_float, probabilities(truth + 1_c_size_t)))
    loss = -log(real(probability, c_double))
  end function kr_multiclass_cross_entropy
end module kernelyra_multiclass_loss
