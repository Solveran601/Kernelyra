! Squared error is isolated so the regression objective is explicit and tested
! independently from parameter updates.
module kernelyra_regression_loss
  use, intrinsic :: iso_c_binding, only: c_double, c_float
  implicit none
contains
  pure function kr_squared_error(error) result(loss)
    real(c_float), value, intent(in) :: error
    real(c_double) :: loss

    loss = real(error, c_double) * real(error, c_double)
  end function kr_squared_error
end module kernelyra_regression_loss
