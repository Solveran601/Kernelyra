module kernelyra_numeric_precision
  use, intrinsic :: iso_c_binding, only: c_float
  use kernelyra_numeric_constants, only: kr_logit_limit, kr_minimum_std
  implicit none
contains
  pure function kr_safe_std(value) result(result_value)
    real(c_float), value, intent(in) :: value
    real(c_float) :: result_value

    result_value = value
    if (abs(result_value) <= kr_minimum_std) result_value = 1.0_c_float
  end function kr_safe_std

  pure function kr_sigmoid(score) result(probability)
    real(c_float), value, intent(in) :: score
    real(c_float) :: probability

    probability = 1.0_c_float / (1.0_c_float + exp(-max(-kr_logit_limit, min(kr_logit_limit, score))))
  end function kr_sigmoid
end module kernelyra_numeric_precision
