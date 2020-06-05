!MATH96012 Project 2

!This module contains two module variables and three subroutines;
!two of these routines must be developed for this assignment.
!Module variables--
! lr_x: training images, typically n x d with n=784 and d<=15000
! lr_y: labels for training images, d-element array containing 0s and 1s
!   corresponding to images of even and odd integers, respectively.
!lr_lambda: l2-penalty parameter, should be set to be >=0.
!Module routines---
! data_init: allocate lr_x and lr_y using input variables n and d. May be used if/as needed.
! clrmodel: compute cost function and gradient using classical logistic
!   regression model (CLR) with lr_x, lr_y, and
!   fitting parameters provided as input
! mlrmodel: compute cost function and gradient using MLR model with m classes
!   and with lr_x, lr_y, and fitting parameters provided as input

module lrmodel
  implicit none
  real(kind=8), allocatable, dimension(:,:) :: lr_x
  integer, allocatable, dimension(:) :: lr_y
  real(kind=8) :: lr_lambda !penalty parameter

contains

!---allocate lr_x and lr_y deallocating first if needed (used by p2_main)--
! ---Use if needed---
subroutine data_init(n,d)
  implicit none
  integer, intent(in) :: n,d
  if (allocated(lr_x)) deallocate(lr_x)
  if (allocated(lr_y)) deallocate(lr_y)
  allocate(lr_x(n,d),lr_y(d))
end subroutine data_init


!Compute cost function and its gradient for CLR model
!for d images (in lr_x) and d labels (in lr_y) along with the
!fitting parameters provided as input in fvec.
!The weight vector, w, corresponds to fvec(1:n) and
!the bias, b, is stored in fvec(n+1)
!Similarly, the elements of dc/dw should be stored in cgrad(1:n)
!and dc/db should be stored in cgrad(n+1)
!Note: lr_x and lr_y must be allocated and set before calling this subroutine.
subroutine clrmodel(fvec,n,d,c,cgrad)
  implicit none
  integer, intent(in) :: n
  integer, intent(in) :: d
  real(kind=8), dimension(n+1), intent(in) :: fvec !fitting parameters
  real(kind=8), intent(out) :: c !cost
  real(kind=8), dimension(n+1), intent(out) :: cgrad !gradient of cost
  integer :: k1,j,k2
  real(kind=8) :: c_sum_1,c_sum_2,c_grad_b1,c_grad_b2,c_grad_b,dc_by_dwj1,dc_by_dwj2,a

  !Declare other variables as needed

  c_sum_1  = 0.d0
  c_sum_2 = 0.d0
  c_grad_b1 = 0.d0
  c_grad_b2 = 0.d0



  do k1 = 1,d
    ! First calculate a1 for the kth image
    a = 1.d0 - (exp(0.d0)/(exp(0.d0) + exp(dot_product(fvec(1:n),lr_x(:,k1)) + fvec(n+1))))
    ! We now add the contribution of the kth image to the first sum in expression for C
    c_sum_1 = c_sum_1 + lr_y(k1)*log(a + 1.0d-12) + (1.d0 - dble(lr_y(k1)))*log((1.d0 - a) + 1.0d-12)
    ! We now add the contribution of the kth image to del(C)/del(b) & del(C)/del(w)
    if (lr_y(k1) == 0.d0) then
    c_grad_b1 = c_grad_b1 + a
    else
    c_grad_b2 = c_grad_b2 + (1.d0 - a)
    end if
  end do



  do j = 1,n
    ! Computes second sum in expression of c
    c_sum_2 = c_sum_2 + (fvec(j)**2)
    dc_by_dwj1 = 0.d0
    dc_by_dwj2 = 0.d0
    ! Calculates del(C)/del(wj)
    do k2 = 1,d
      a = 1.d0 - (exp(0.d0)/(exp(0.d0) + exp(dot_product(fvec(1:n),lr_x(:,k2)) + fvec(n+1))))
      if (lr_y(k2) == 0.d0) then
      dc_by_dwj1 = dc_by_dwj1 + (a*lr_x(j,k2) + 2*lr_lambda*fvec(j))
      else
      dc_by_dwj2 = dc_by_dwj2 + (1.d0 - a)*lr_x(j,k2)
      end if
    end do
    cgrad(j) = dc_by_dwj1 + dc_by_dwj2
  end do

  cgrad(n+1) = c_grad_b1 + c_grad_b2
  c = lr_lambda*c_sum_2 - c_sum_1
end subroutine clrmodel


!!Compute cost function and its gradient for MLR model
!for d images (in lr_x) and d labels (in lr_y) along with the
!fitting parameters provided as input in fvec. The labels are integers
! between 0 and m-1.
!fvec contains the elements of the weight matrix w and the bias vector, b
! Code has been provided below to "unpack" fvec
!The elements of dc/dw and dc/db should be stored in cgrad
!and should be "packed" in the same order that fvec was unpacked.
!Note: lr_x and lr_y must be allocated and set before calling this subroutine.
subroutine mlrmodel(fvec,n,d,m,c,cgrad)
  implicit none
  integer, intent(in) :: n,d,m !training data sizes and number of classes
  real(kind=8), dimension((m-1)*(n+1)), intent(in) :: fvec !fitting parameters
  real(kind=8), intent(out) :: c !cost
  real(kind=8), dimension((m-1)*(n+1)), intent(out) :: cgrad !gradient of cost
  real(kind=8), dimension((m-1),(n+1)) :: Wij
  integer :: i1,j1,i,l1,k3,k1,k,a1,a2,k4,l2,k5,k6,k7,j2,j3,j4,t,p,i2
  real(kind=8), dimension(m,d) :: Am
  real(kind=8), dimension(m-1,n) :: w
  real(kind=8), dimension(m-1) :: b,Z1,Z2
  real(kind=8) :: cbi_sum1,cbi_sum2a,cbi_sum2b,cw_sum1,cw_sum2a,cw_sum2b,sumw1,sumw2,sum3,sum3a
  !Declare other variables as needed


  !unpack fitting parameters (use if needed)
  do i1=1,n
    j1 = (i1-1)*(m-1)+1
    w(:,i1) = fvec(j1:j1+m-2) !weight matrix
  end do
  b = fvec((m-1)*n+1:(m-1)*(n+1)) !bias vector

  ! Basic structure to calculate dc/bi:
  do i = 1,m-1
    !Generate an (m-1)xd matrix of ai^k = Am
    do k = 1,d ! For the d images
    Z1(1) = dble(exp(0.d0))
      do a2 = 2,m
        Z1(a2) = exp(dot_product(w(i,:),lr_x(:,k)) + b(i))
      end do
      do a1 = 1,m
        Am(a1,k) = dble(Z1(a1)/(exp(0.d0) + dble(sum(Z1))))
      end do
    end do ! Am has now been generated
    cbi_sum1 = 0.d0
    do k1 = 1,d
      if (lr_y(k1) .eq. i) then
      cbi_sum1 = cbi_sum1 + (1.d0 - Am(i,k1))
      end if
    end do ! First sum in dc/dbi

    cbi_sum2a = 0.d0
    do l1 = 1,m
      if (l1 .ne. i) then
      cbi_sum2b = 0.d0
      do k3 = 1,d
        if (lr_y(k3) .eq. l1) then
        cbi_sum2b = cbi_sum2b + Am(i,k3)
        end if
      end do
      cbi_sum2a = cbi_sum2a + cbi_sum2b
      end if
    end do
    Wij(i,n+1) = cbi_sum2a - cbi_sum1  ! dc/db computed.


    do j2 = 1,n
      cw_sum1 = 0.d0
      do k4 = 1,d
        if (lr_y(k4) .eq. i) then
        cw_sum1 = cw_sum1 + (1.d0 - Am(i,k4))*lr_x(j2,k4)
        end if
      end do
      cw_sum2a = 0.d0
      do l2 = 1,m
        if (l2 .ne. i) then
        cw_sum2b = 0.d0
        do k5 = 1,d
          if (lr_y(k5) .eq. l2) then
          cw_sum2b = cw_sum2b + Am(i,k5)*lr_x(j2,k5)
          end if
        end do
        end if
        cw_sum2a = cw_sum2a + cw_sum2b
      end do

      Wij(i,j2) = cw_sum2a - cw_sum1 + 2*lr_lambda*w(i,j2)

    end do

  end do  ! dc/dwij computed.


! C 2nd sum
  sumw1 = 0.d0
  do i2 = 1,m
    sumw2 = 0.d0
    do j3 = 1,n
      sumw2 = sumw2 + w(i2,j3)**2
    end do
    sumw1 = sumw1 + sumw2

    do k6 = 1,d
      Z2(1) = dble(exp(1.d0))
      do t = 2,m
        Z2(t) = exp(dot_product(w(i2,:),lr_x(:,k6)) + b(i2))
      end do
      do a1 = 1,m
        Am(a1,k) = dble(Z2(a1)/(exp(1.d0) + dble(sum(Z2))))
      end do ! A generated
    end do
  end do
  sum3 = 0.d0
  do j4 = 1,m
    sum3a = 0.d0
    do k7 = 1,d
      if (lr_y(k7) .eq. j4) then
      sum3a = sum3a + log(Am(j4,k7))
      end if
    end do
    sum3 = sum3 + sum3a
  end do

  c = lr_lambda*sumw1 - sum3

  do p = 1,n
    cgrad((m-1)*(p-1)+1 : p*(m-1)) = Wij(:,p) !Packing cgrad in the same order fvec was unpacked
  end do

end subroutine mlrmodel


end module lrmodel
