
!MATH 96012 Project 3
!This module contains four module variables and two subroutines;
!one of these routines must be developed for this assignment.
!Module variables--
! b.bm_l, b.bm_s0, b.bm_r0, b.bm_a: the parameters l, s0, r0, and a in the particle dynamics model
! numthreads: The number of threads that should be used in parallel regions within simulate2_omp
!
!Module routines---
! simulate2_f90: Simulate particle dynamics over m trials. Return: all x-y positions at final time
! and alpha at nt+1 times averaged across the m trials.
! simulate2_omp: Same input/output functionality as simulate2.f90 but parallelized with OpenMP
! f2py compiled with python -m numpy.f2py --f90flags='-fopenmp' -c p3dev.f90 -m m1 -lgomp

module bmotion
  use omp_lib
  implicit none
  integer :: NumThreads
  real(kind=8) :: bm_l,bm_s0,bm_r0,bm_a
  real(kind=8), parameter :: pi = acos(-1.d0)
  complex(kind=8), parameter :: ii = complex(0.d0,1.d0)
contains

!Compute m particle dynamics simulations using the parameters,bm_l,bm_s0,bm_r0,bm_a.
!Input:
!m: number of simulations
!n: number of particles
!nt: number of time steps
!Output:
! x: x-positions at final time step for all m trials
! y: y-positions at final time step for all m trials
! alpha_ave: alignment parameter at each time step (including initial condition)
! averaged across the m trials
subroutine simulate2_f90(m,n,nt,x,y,alpha_ave)
  implicit none
  integer, intent(in) :: m,n,nt
  real(kind=8), dimension(m,n), intent(out) :: x,y
  real(kind=8), dimension(nt+1), intent(out) :: alpha_ave
  integer :: i1,j1,k1
  real(kind=8), dimension(m,n) :: nn !neighbors
  real(kind=8) :: r0sq !r0^2
  real(kind=8), dimension(m,n) :: phi_init,r_init,theta !used for initial conditions
  real(kind=8), dimension(m,n,n) :: dist2 !distance squared
  real(kind=8), allocatable, dimension(:,:,:) :: temp
  complex(kind=8), dimension(m,n) :: phase
  complex(kind=8), dimension(m,n,nt+1) :: exp_theta,AexpR

!---Set initial condition and initialize variables----
  allocate(temp(m,n,nt+1))
  call random_number(phi_init)
  call random_number(r_init)
  call random_number(theta)
  call random_number(temp)
  phi_init = phi_init*(2.d0*pi)
  r_init = sqrt(r_init)
  theta = theta*(2.d0*pi) !initial direction of motion
  x = r_init*cos(phi_init)+0.5d0*bm_l !initial positions
  y = r_init*sin(phi_init)+0.5d0*bm_l

  alpha_ave=0.d0
  r0sq = bm_r0*bm_r0
  exp_theta(:,:,1) = exp(ii*theta)
  AexpR = bm_a*exp(ii*temp*2.d0*pi) !noise term
  deallocate(temp)
  nn = 0

!----------------------------------------------
  !Time marching
  do i1 = 2,nt+1

    phase=0.d0
    dist2 = 0.d0

    !Compute distances
    do j1=1,n-1
      do k1 = j1+1,n
        dist2(:,j1,k1) = (x(:,j1)-x(:,k1))**2 + (y(:,j1)-y(:,k1))**2
        where (dist2(:,j1,k1)>r0sq)
          dist2(:,j1,k1)=0
        elsewhere
          dist2(:,j1,k1)=1
        end where
       dist2(:,k1,j1) =dist2(:,j1,k1)
      end do
    end do

    nn = sum(dist2,dim=3)+1

    !Update phase
    phase =  exp_theta(:,:,i1-1) +nn*AexpR(:,:,i1-1)
    do j1=1,m
      phase(j1,:) = phase(j1,:) + matmul(dist2(j1,:,:),exp_theta(j1,:,i1-1))
    end do

    !Update Theta
    theta = atan2(aimag(phase),real(phase))

    !Update X,Y
    x = x + bm_s0*cos(theta)
    y = y + bm_s0*sin(theta)

    x = modulo(x,bm_l)
    y = modulo(y,bm_l)

    exp_theta(:,:,i1) = exp(ii*theta)

  end do

  alpha_ave = (1.d0/dble(m*n))*sum(abs(sum(exp_theta,dim=2)),dim=1)

end subroutine simulate2_f90


!Same functionality as simulate2_f90, but parallelized with OpenMP
!Parallel regions should use numthreads threads.
!Compute m particle dynamics simulations using the parameters,bm_l,bm_s0,bm_r0,bm_a.
!Same functionality as simulate2_f90, but parallelized with OpenMP
!Parallel regions should use numthreads threads.
!Input:
!m: number of simulations
!n: number of particles
!nt: number of time steps
!Output:
! x: x-positions at final time step for all m trials
! y: y-positions at final time step for all m trials
! alpha_ave: alignment parameter at each time step (including initial condition)
! averaged across the m trials



!-----------------------------------------------------
! MY APPROACH TO PARALLLELIZATION WITH OpenMP
!
!In simulate2_f90 above, the subroutine works on all of the
!m simulations simultaneously at each time step. Here, the time step
!iterations are not independent of each other, so it is not possible
!to parallelize this do loop.

!Since all of the m simulations are independent of each other,
!I have adapted the subroutine to iterate through the m simulations
!as the outer loop, and then have the time step do loop as the inner loop.
!The m independent simulations do loop can then efficiently be parallelised,
!as is confirmed in my performance analysis.
!-----------------------------------------------------


subroutine simulate2_omp(m,n,nt,xfinal,yfinal,alpha_ave)
  implicit none
  integer, intent(in) :: m,n,nt
  real(kind=8), dimension(n) :: x,y
  real(kind=8), dimension(nt+1), intent(out) :: alpha_ave
  real(kind=8), dimension(m,n), intent(out) :: xfinal,yfinal
  integer :: i1,j1,k1,threadID,m1,p1
  real(kind=8), dimension(n) :: nn !neighbors
  real(kind=8) :: r0sq !r0^2
  real(kind=8), dimension(n) :: phi_init,r_init,theta !used for initial conditions
  real(kind=8), dimension(n,n) :: dist2 !distance squared
  real(kind=8), dimension(n,nt+1) :: temp
  complex(kind=8), dimension(n) :: phase
  complex(kind=8), dimension(n,nt+1) :: exp_theta,AexpR
  complex(kind=8), dimension(m,nt+1) :: almost_alpha
  call omp_set_num_threads(2)

  !---Set initial condition and initialize variables----
  !allocate(temp(n,nt+1))
  call random_number(phi_init)
  call random_number(r_init)
  call random_number(theta)
  call random_number(temp)
  phi_init = phi_init*(2.d0*pi)
  r_init = sqrt(r_init)
  theta = theta*(2.d0*pi) !initial direction of motion
  x = r_init*cos(phi_init)+0.5d0*bm_l !initial positions
  y = r_init*sin(phi_init)+0.5d0*bm_l

  alpha_ave=0.d0
  r0sq = bm_r0*bm_r0
  exp_theta(:,1) = exp(ii*theta)
  AexpR = bm_a*exp(ii*temp*2.d0*pi) !noise term
  !deallocate(temp)
  nn = 0
  !$OMP parallel do private(j1,p1)
  !----------------------------------------------
  !Time marching
  do m1 = 1,m
    do i1 = 2,nt+1
      phase = 0.d0
      dist2 = 0.d0

      do j1=1,n-1
        do k1=j1+1,n
          dist2(j1,k1) = (x(j1)-x(k1))**2 + (y(j1)-y(k1))**2
          if (dist2(j1,k1)>r0sq) then
            dist2(j1,k1)=0
          else
            dist2(j1,k1)=1
          end if
         dist2(k1,j1) =dist2(j1,k1)
        end do
      end do


      nn = sum(dist2,dim=2)+1
      phase = exp_theta(:,i1-1) + nn*AexpR(:,i1)
      do p1=1,n
        phase(p1) = phase(p1) + dot_product(dist2(p1,:),exp_theta(:,i1-1))
      end do

      theta = atan2(aimag(phase),real(phase))

      x = x + bm_s0*cos(theta)
      y = y + bm_s0*sin(theta)

      x = modulo(x,bm_l)
      y = modulo(y,bm_l)

      exp_theta(:,i1) = exp(ii*theta)

    end do
    xfinal(m1,:) = x
    yfinal(m1,:) = y
    almost_alpha(m1,:) = (1.d0/dble(n))*abs(sum(exp_theta,dim=1))
  end do
  !$OMP end parallel do
  alpha_ave = (0.5d0/dble(m))*sum(almost_alpha,dim=1)

end subroutine

end module bmotion
