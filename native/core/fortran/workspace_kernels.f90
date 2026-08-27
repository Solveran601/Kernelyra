! Bounded-workspace helpers for the dense Fortran training kernels.
!
! These routines never allocate.  They choose a row tile whose input footprint
! is capped near 256 KiB, then fuse the gradient accumulation with each tile.
! This keeps the hot row-major path cache-local without changing the C ABI.
module kernelyra_workspace_kernels
  use, intrinsic :: iso_c_binding, only: c_float, c_size_t
  use kernelyra_gradient_layout, only: kr_row_offset
  implicit none

  integer(c_size_t), parameter :: kr_workspace_target_bytes = 262144_c_size_t
contains
  pure function kr_fortran_workspace_tile_rows(rows, features) result(tile_rows)
    integer(c_size_t), value, intent(in) :: rows, features
    integer(c_size_t) :: tile_rows, bytes_per_row, candidate

    if (rows == 0_c_size_t .or. features == 0_c_size_t) then
      tile_rows = 0_c_size_t
      return
    end if

    ! One input row plus the scalar error.  The result is deliberately a
    ! conservative upper bound: callers retain ownership of x and errors.
    bytes_per_row = features * 4_c_size_t + 4_c_size_t
    if (bytes_per_row >= kr_workspace_target_bytes) then
      tile_rows = 1_c_size_t
      return
    end if

    candidate = kr_workspace_target_bytes / bytes_per_row
    tile_rows = min(rows, max(1_c_size_t, candidate))
  end function kr_fortran_workspace_tile_rows

  subroutine kr_fortran_zero_gradient_f32(gradient, features)
    integer(c_size_t), value, intent(in) :: features
    real(c_float), intent(out) :: gradient(*)
    integer(c_size_t) :: feature

    !$omp simd
    do feature = 1_c_size_t, features
      gradient(feature) = 0.0_c_float
    end do
  end subroutine kr_fortran_zero_gradient_f32

  subroutine kr_fortran_accumulate_gradient_tile_f32( &
      x, errors, first_row, last_row, features, gradient)
    integer(c_size_t), value, intent(in) :: first_row, last_row, features
    real(c_float), intent(in) :: x(*), errors(*)
    real(c_float), intent(inout) :: gradient(*)
    integer(c_size_t) :: row, feature, offset

    if (last_row < first_row .or. features == 0_c_size_t) return

    ! Rows are contiguous in x.  The only live per-row value is errors(row),
    ! so this path needs no temporary matrix or second batch-sized workspace.
    do row = first_row, last_row
      offset = kr_row_offset(row, features)
      !$omp simd
      do feature = 1_c_size_t, features
        gradient(feature) = gradient(feature) + x(offset + feature) * errors(row)
      end do
    end do
  end subroutine kr_fortran_accumulate_gradient_tile_f32
end module kernelyra_workspace_kernels
