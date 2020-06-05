!M3C FINAL PROJECT
!Final project part 1
!Module for flow simulations of liquid through tube
!This module contains a few module variables (see comments below)
!and four subroutines:
!jacobi: Uses jacobi iteration to compute solution
! to flow through tube
!sgisolve: To be completed. Use sgi method to
! compute flow through tube
!mvec: To be completed; matrix-vector multiplication z = Ay
!mtvec: To be completed; matrix-vector multiplication z = A^T y
!--------------------------------------------------
!p41.f90 COMPILED AS m2 USING THE UNIX COMMAND:
!   python -m numpy.f2py --f90flags='-fopenmp' -c p41_template.f90 -m m2 -lgomp
!--------------------------------------------------
module flow
    use omp_lib
    implicit none
    real(kind=8), parameter :: pi = acos(-1.d0)
    integer :: numthreads !number of threads used in parallel regions
    integer :: fl_kmax=10000 !max number of iterations
    real(kind=8) :: fl_tol=0.00000001d0 !convergence criterion
    real(kind=8), allocatable, dimension(:) :: fl_deltaw !|max change in w| each iteration
    real(kind=8) :: fl_s0=0.1d0 !deformation magnitude

contains
!-----------------------------------------------------
!Solve 2-d tube flow problem with Jacobi iteration
subroutine jacobi(n,w)
    !input  n: number of grid points (n+2 x n+2) grid
    !output w: final velocity field
    !Should also compute fl_deltaw(k): max(|w^k - w^k-1|)
    !A number of module variables can be set in advance.

    integer, intent(in) :: n
    real(kind=8), dimension(0:n+1,0:n+1), intent(out) :: w
    integer :: i1,j1,k1
    real(kind=8) :: del_r,del_t,del_r2,del_t2
    real(kind=8), dimension(0:n+1) :: s_bc,fac_bc
    real(kind=8), dimension(0:n+1,0:n+1) :: r,r2,t,RHS,w0,wnew,fac,fac2,facp,facm

    if (allocated(fl_deltaw)) then
      deallocate(fl_deltaw)
    end if
    allocate(fl_deltaw(fl_kmax))


    !grid--------------
    del_t = 0.5d0*pi/dble(n+1)
    del_r = 1.d0/dble(n+1)
    del_r2 = del_r**2
    del_t2 = del_t**2


    do i1=0,n+1
        r(i1,:) = i1*del_r
    end do

    do j1=0,n+1
        t(:,j1) = j1*del_t
    end do
    !-------------------

    !Update-equation factors------
    r2 = r**2
    fac = 0.5d0/(r2*del_t2 + del_r2)
    facp = r2*del_t2*fac*(1.d0+0.5d0*del_r/r) !alpha_p/gamma
    facm = r2*del_t2*fac*(1.d0-0.5d0*del_r/r) !alpha_m/gamma
    fac2 = del_r2 * fac !beta/gamma
    RHS = fac*(r2*del_r2*del_t2) !1/gamma
    !----------------------------

    !set initial condition/boundary deformation
    w0 = (1.d0-r2)/4.d0
    w = w0
    wnew = w0
    s_bc = fl_s0*exp(-10.d0*((t(0,:)-pi/2.d0)**2))/del_r
    fac_bc = s_bc/(1.d0+s_bc)


    !Jacobi iteration
    do k1=1,fl_kmax
        wnew(1:n,1:n) = RHS(1:n,1:n) + w(2:n+1,1:n)*facp(1:n,1:n) + w(0:n-1,1:n)*facm(1:n,1:n) + &
                                         (w(1:n,0:n-1) + w(1:n,2:n+1))*fac2(1:n,1:n)

        !Apply boundary conditions
        wnew(:,0) = wnew(:,1) !theta=0
        wnew(:,n+1) = wnew(:,n) !theta=pi/2
        wnew(0,:) = wnew(1,:) !r=0
        wnew(n+1,:) = wnew(n,:)*fac_bc !r=1s

        fl_deltaw(k1) = maxval(abs(wnew-w)) !compute relative error

        w=wnew    !update variable
        if (fl_deltaw(k1)<fl_tol) exit !check convergence criterion
        if (mod(k1,1000)==0) print *, k1,fl_deltaw(k1)
    end do

    print *, 'k,error=',k1,fl_deltaw(min(k1,fl_kmax))

end subroutine jacobi
!-----------------------------------------------------

!Solve 2-d tube flow problem with sgi method
subroutine sgisolve(n,w)
  implicit none
  !input  n: number of grid points (n+2 x n+2) grid
  !output w: final velocity field stored in a column vector
  !Should also compute fl_deltaw(k): max(|w^k - w^k-1|)
  !A number of module variables can be set in advance.
  integer, intent(in) :: n
  real(kind=8), dimension((n+2)*(n+2)), intent(out) :: w
  real(kind=8) :: del_t,del_r,k,mu,del_r2,del_t2
  real(kind=8), allocatable, dimension(:,:) :: M
  real(kind=8), dimension((n+2)*(n+2)) :: b,v,d,x,e,enew,A_d,dnew,M_d,xnew
  real(kind=8), dimension(0:n+1) :: fac2,facm,facp,fac_bc,fac
  real(kind=8), dimension(0:n+1) :: s,s_j
  integer :: l1,p1,s1,b1,p2
  !add other variables as needed
  if (allocated(fl_deltaw)) then
    deallocate(fl_deltaw)
  end if
  allocate(fl_deltaw(fl_kmax))

  !-------------------------------------------
  !WE USE OPENMP TO PARALLISE THE SET UP OF THE COEFFICIENT VECTORS.
  !AS MVEC & MTVEC ARE REPEATEDLY USED IN THE MAIN LOOP, WE CANNOT PARALLELISE
  !THESE SUBROUTINES AS THIS WOULD CREATE PARALLEL REGIONS WITHIN PARALLEL
  !REGIONS AND CRITICAL VARAIABLE INFORMATION WOULD BE LOST.
  !-------------------------------------------

  !$ call omp_set_num_threads(numthreads)

  del_t = 0.5d0*pi/dble(n+1)  !grid spacings
  del_r = 1.d0/dble(n+1)
  del_t2 = del_t**2
  del_r2 = del_r**2

  !now set up coefficient vectors

  !$OMP parallel do private(s1)
  do s1 = 0,n+1
    s(s1) = fl_s0*(exp(-10*(s1*del_t - (pi/2.d0))**2) + exp(-10*(s1*del_t + (pi/2.d0))**2))
  end do
  !$OMP end parallel do

  !$OMP parallel do private(p2)
  do p2 = 1,n
    fac(p2) = 1.d0/((2.d0/del_r2) + (2.d0/(del_t2*(p2*del_r)**2)))
  end do
  !$OMP end parallel do

  !$OMP parallel do private(p1)
  do p1 = 1,n
    facm(p1) = (1.d0/del_r2 - 1.d0/(2.d0*p1*del_r2))*fac(p1)
    facp(p1) = (1.d0/del_r2 + 1.d0/(2.d0*p1*del_r2))*fac(p1)
    fac2(p1) = (1.d0/(del_t2*(p1*del_r)**2))*fac(p1)
  end do
  !$OMP end parallel do

  fac_bc = s/dble(del_r + s)

  b = 0.d0
  !$OMP parallel do private(b1)
  do b1 = 1,n
    b((b1*(n+2))+2:(b1*(n+2))+n+1) = -1.d0*fac(b1)
  end do
!$OMP end parallel do

  !initial states
  call mtvec(n,fac,fac2,facp,facm,fac_bc,b,v) !compute v

  d = v
  x = 0.d0
  e = v
  !main loop
  !------------------------------------------------------
  !------------------------------------------------------
  !THE MAIN ITERATIVE LOOP OF SGISOLVE IS NOT THREADSAFE, SO
  !IT IS NOT POSSIBLE TO PARALLISE EFFICIENTLY.
  !------------------------------------------------------
  !------------------------------------------------------
  do l1 = 1,fl_kmax
    call mvec(n,fac,fac2,facp,facm,fac_bc,d,A_d) !computes matrix vector product A d
    call mtvec(n,fac,fac2,facp,facm,fac_bc,A_d,M_d) !computes mvecprod Md = A^T A d
    k = dble(dot_product(e,e))/dble(dot_product(d,M_d))

    xnew = x + k*d
    enew = e - k*M_d
    mu = dble(dot_product(enew,enew))/dble(dot_product(e,e))
    dnew = enew + mu*d

    fl_deltaw(l1) = maxval(abs(enew-e))

    if (fl_deltaw(l1)<fl_tol) exit !check convergence criterion
    if (mod(l1,1000)==0) print *, l1,fl_deltaw(l1)

    !update variables
    e = enew
    d = dnew
    x = xnew

  end do
  w = x

end subroutine sgisolve

!------------------------------------------------------
!------------------------------------------------------

!Compute matrix-vector multiplication, z = Ay
subroutine mvec(n,fac,fac2,facp,facm,fac_bc,y,z)
    !input n: grid is (n+2) x (n+2)
    ! fac,fac2,facp,facm,fac_bc: arrays that appear in
    !   discretized equations
    ! y: vector multipled by A
    !output z: result of multiplication Ay
    implicit none
    integer, intent(in) :: n
    real(kind=8), dimension(n+2), intent(in) :: fac,fac2,facp,facm,fac_bc
    real(kind=8), dimension((n+2)*(n+2)), intent(in) :: y
    real(kind=8), dimension((n+2)*(n+2)), intent(out) :: z
    integer :: i1,j1,j2,i2,k

    !From the governing equation
    do i1 = 1,n
      do j1 = 2,n+1
        k = i1*(n+2) + j1
        z(k) = y(k-(n+2))*facm(i1+1) + y(k-1)*fac2(i1+1) + y(k)*(-1.d0) + y(k+1)*fac2(i1+1) + y(k+n+2)*facp(i1+1)
      end do
    end do

    !Boundary conditions
    do i2 = 1,n
      z((i2*(n+2))+1) = y((i2*(n+2))+1)*(-1.d0) + 1.d0*y((i2*(n+2))+2)  !for w_i,1 - w_i,0 = 0 in row i,0
      z((i2*(n+2))+n+2) = y((i2*(n+2))+n+2)*(1.d0) + (-1.d0)*y((i2*(n+2))+n+1)  !for w_i,n+1 - w_i,n = 0 in row i,n+1
    end do
    do j2 = 1,(n+2)
      z(j2) = y(n+2+j2) + (-1.d0)*y(j2) !w_1,j - w0,j = 0 in row 0,j
      z(((n+1)*(n+2))+j2) = 1.d0*y(((n+1)*(n+2))+j2) + (-1.d0)*fac_bc(j2)*y((n*(n+2))+j2)  ! for w_n+1,j - w_n,j*fac_bc(j) in row w_n+1,j
    end do

end subroutine mvec

!------------------------------------------------------
!------------------------------------------------------


!------------------------------------------------------
!------------------------------------------------------


!Compute matrix-vector multiplication, z = A^T y
subroutine mtvec(n,fac,fac2,facp,facm,fac_bc,y,z)
    !input n: grid is (n+2) x (n+2)
    ! fac,fac2,facp,facm,fac_bc: arrays that appear in
    !   discretized equations
    ! y: vector multipled by A^T
    !output z: result of multiplication A^T y
    implicit none
    integer, intent(in) :: n
    real(kind=8), dimension(n+2), intent(in) :: fac,fac2,facp,facm,fac_bc
    real(kind=8), dimension((n+2)*(n+2)), intent(in) :: y
    real(kind=8), dimension((n+2)*(n+2)), intent(out) :: z
    real(kind=8), dimension((n+2)*(n+2)) :: s,s1,finalvec
    !real(kind=8), dimension((n+2)**2,(n+2)**2) :: A,A_t
    integer :: i1,j1,j2,i2,k,k2,k3,k4

    !add other variables as needed
    finalvec = 0.d0

    !Governing equations

    do i1 = 1,n
      do j1 = 2,n+1
        k = i1*(n+2) + j1
        s = 0.d0

        s(k-(n+2)) = facm(i1+1)
        s(k-1) = fac2(i1+1)
        s(k) = -1.d0
        s(k+1) = fac2(i1+1)
        s(k+n+2) = facp(i1+1)

        finalvec = finalvec + y(k)*s
      end do
    end do

    !Boundary conditions
    do i2 = 1,n
      s = 0.d0
      s1 = 0.d0
      k2 = (i2*(n+2))+1

      s(k2) = -1.d0
      s(k2+1) = 1.d0

      finalvec = finalvec + y(k2)*s

      k3 = (i2*(n+2))+n+2

      s1(k3) = 1.d0
      s1(k3-1) = -1.d0

      finalvec = finalvec + y(k3)*s1
    end do

    do j2 = 1,(n+2)
      s = 0.d0
      s1 = 0.d0

      s(n+2+j2) = 1.d0  !w_1,j - w0,j = 0 in row 0,j
      s(j2) = -1.d0
      finalvec = finalvec + y(j2)*s

      k4 = ((n+1)*(n+2))+j2
      s1(k4) = 1.d0 ! for w_n+1,j - w_n,j*fac_bc(j) in row w_n+1,j
      s1(k4-(n+2)) = -1.d0*fac_bc(j2)
      finalvec = finalvec + y(k4)*s1
    end do

    z = finalvec

end subroutine mtvec

end module flow
