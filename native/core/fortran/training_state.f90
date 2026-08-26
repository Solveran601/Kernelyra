! Parameter-state updates that are common to binary and regression loops.
module kernelyra_training_state
  use, intrinsic :: iso_c_binding, only: c_float
  implicit none
contains
  subroutine kr_update_bias(bias, gradient, learning_rate, inverse)
    real(c_float), intent(inout) :: bias
    real(c_float), value, intent(in) :: gradient, learning_rate, inverse

    bias = bias - learning_rate * gradient * inverse
  end subroutine kr_update_bias
end module kernelyra_training_state
