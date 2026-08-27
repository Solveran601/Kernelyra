! Row-major layout helpers shared by gradient and training kernels.
module kernelyra_gradient_layout
  use, intrinsic :: iso_c_binding, only: c_size_t
  implicit none
contains
  pure function kr_row_offset(row, features) result(offset)
    integer(c_size_t), value, intent(in) :: row, features
    integer(c_size_t) :: offset

    offset = (row - 1_c_size_t) * features
  end function kr_row_offset
end module kernelyra_gradient_layout
